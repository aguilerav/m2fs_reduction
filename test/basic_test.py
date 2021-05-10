import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from m2fsredux import basic, cosmics, tracer


TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_FITS = os.path.join(TESTS_DIR, 'assets', 'raw_images', 'b0147c1.fits')


def test_str_to_npslice():
    slice_string = '[2049:2176,2057:2184]'
    expected_output = (2048, 2176, 2056, 2184)
    output = basic.str_slice_to_corners(slice_string)
    assert expected_output == output, "Incorrect slicing"


class BasicThatNeed1Image(unittest.TestCase):

    def setUp(self):
        self.c1 = fits.open(TEST_FITS)

    def tearDown(self):
        self.c1.close()

    def test_debias(self):
        expected_bias = self.c1[0].data[1028:, :]
        expected_bias = np.median(expected_bias, axis=0)
        expected_output = self.c1[0].data - expected_bias

        # Now apply debias to fits file
        basic.debias(self.c1[0])
        np.testing.assert_array_equal(expected_output, self.c1[0].data)

        assert('debias: Substracted bias on a per column basis' in
               self.c1[0].header['HISTORY'])

    def test_trim(self):
        basic.trim(self.c1[0])
        assert self.c1[0].data.shape == (1028, 1024)
        assert 'BIASSEC' not in self.c1[0].header
        assert("trim: bias section trimmed away" in
               self.c1[0].header['HISTORY'])

    def test_gain_correct(self):
        expected_data = self.c1[0].data * 0.68
        basic.gain_correct(self.c1[0])
        np.testing.assert_array_equal(expected_data, self.c1[0].data)

        assert self.c1[0].header['BUNIT'] == 'E-/PIXEL'
        assert 'EGAIN' not in self.c1[0].header


class BasicThatNeed4Images(unittest.TestCase):

    def setUp(self):
        self.c1 = fits.open(TEST_FITS)
        self.f1 = fits.open(TEST_FITS)
        self.f2 = fits.open(TEST_FITS.replace('c1', 'c2'))
        self.f3 = fits.open(TEST_FITS.replace('c1', 'c3'))
        self.f4 = fits.open(TEST_FITS.replace('c1', 'c4'))

        basic.trim(self.f1[0])
        basic.trim(self.f2[0])
        basic.trim(self.f3[0])
        basic.trim(self.f4[0])

    def tearDown(self):
        self.c1.close()
        self.f1.close()
        self.f2.close()
        self.f3.close()
        self.f4.close()

    def test_merge_data(self):
        expected_output = np.zeros((1028*2, 1024*2))
        expected_output[   0:1028,    0:1024] = self.f1[0].data
        expected_output[   0:1028, 1024:2048] = np.fliplr(self.f2[0].data)
        expected_output[1028:2056,    0:1024] = np.flipud(self.f4[0].data)
        expected_output[1028:2056, 1024:2048] = np.flip(self.f3[0].data,
                                                        (0, 1))

        output = basic.merge_data(self.f1[0], self.f2[0], self.f3[0],
                                  self.f4[0])

        np.testing.assert_array_equal(expected_output, output.data)
        merge_msg = "merge: Merged 4 amplifiers into single frame"
        assert merge_msg in output.header['HISTORY']
        assert 'GAIN' not in output.header

        assert output.header['FILENAME'] == "b0147"
        assert output.header['DATASEC'] == "[1:2048,1:2056]"
        assert output.header['TRIMSEC'] == "[1:2048,1:2056]"
        assert output.header['ENOISE'] == 2.5

    def test_merge_error(self):
        expected_err = np.zeros((1028*2, 1024*2))
        expected_err[   0:1028,    0:1024] = np.sqrt(self.f1[0].data + 2.7**2)
        expected_err[   0:1028, 1024:2048] = np.sqrt(
                                           np.fliplr(self.f2[0].data) + 2.3**2)
        expected_err[1028:2056,    0:1024] = np.sqrt(
                                           np.flipud(self.f4[0].data) + 2.6**2)
        expected_err[1028:2056, 1024:2048] = np.sqrt(
                                           np.flip(self.f3[0].data, (0, 1)) +
                                           2.4**2)

        output = basic.merge_error(self.f1[0], self.f2[0], self.f3[0],
                                   self.f4[0])

        np.testing.assert_array_equal(expected_err, output.data)


def test_grow_mask():
    input_arr = np.array([[1, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
    expected = np.array([[1, 1, 0],
                         [1, 1, 0],
                         [0, 0, 0]])
    output = basic.grow_mask(input_arr, ngrow=1)
    np.testing.assert_array_equal(expected, output,
                                  "Corner (0,0), incorrect")

    input_arr = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 1]])
    expected = np.array([[0, 0, 0],
                         [0, 1, 1],
                         [0, 1, 1]])
    output = basic.grow_mask(input_arr, ngrow=1)
    np.testing.assert_array_equal(expected, output,
                                  "Corner (-1, -1), incorrect")

    input_arr = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]])

    expected = np.array([[1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1],
                         [0, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0]])
    output = basic.grow_mask(input_arr, ngrow=1)
    np.testing.assert_array_equal(expected, output,
                                  "More general case failed")


def test_cosray():
    '''This test takes a while, it may be good idea to exclude it'''
    test_mask_fname = os.path.join(TESTS_DIR, 'assets', 'temp_products',
                                   'mask-b0147btgm.npy')
    exptected_mask = np.load(test_mask_fname)

    test_fits_fname = os.path.join(TESTS_DIR, 'assets', 'temp_products',
                                   'b0147btgm.fits')
    test_fits = fits.open(test_fits_fname)
    try:
        basic.cosray(test_fits, ngrow=1)

        np.testing.assert_array_equal(test_fits[0].data[exptected_mask],
                                      np.nan)
        np.testing.assert_array_equal(test_fits[1].data[exptected_mask],
                                      np.nan)

        msg = 'cosray: Masked Cosmic Rays with LA Cosmic'
        assert msg in test_fits[0].header['HISTORY']
    finally:
        test_fits.close()
