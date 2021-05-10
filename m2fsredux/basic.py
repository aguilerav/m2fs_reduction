import os
import re
import numpy as np
from astropy.io import fits

from . import cosmics


def str_slice_to_corners(str_section):
    """
    Headers contain bias, data, and trim sections. They are like numpy slices
    but they are strings. They are also flipped and 1 based. This routine
    parses the values and return proper numpy slices.

    Example:

    '[2049:2176,2057:2184]' -> [2056:2184, 2048:2176]
    """
    corners = re.findall("[0-9]+", str_section)
    col_start, col_end = int(corners[0])-1, int(corners[1])
    row_start, row_end = int(corners[2])-1, int(corners[3])
    return col_start, col_end, row_start, row_end


def debias(fits_ext):
    """ Subtract bias column wise
    """
    (col_start, col_end,
     row_start, row_end) = str_slice_to_corners(fits_ext.header['BIASSEC'])
    column_wise_bias_section = np.s_[row_start:row_end, :]

    bias_value = np.median(fits_ext.data[column_wise_bias_section], axis=0)
    fits_ext.data = fits_ext.data - bias_value
    msg = "debias: Substracted bias on a per column basis"
    fits_ext.header['HISTORY'] = msg


def trim(fits_ext):
    (col_start, col_end,
     row_start, row_end) = str_slice_to_corners(fits_ext.header['DATASEC'])
    data_section = np.s_[row_start:row_end, col_start:col_end]
    fits_ext.data = fits_ext.data[data_section].copy()
    del fits_ext.header['BIASSEC']
    fits_ext.header['HISTORY'] = "trim: bias section trimmed away"


def gain_correct(fits_ext):
    gain = fits_ext.header['EGAIN']
    fits_ext.data = fits_ext.data * gain
    fits_ext.header['BUNIT'] = 'E-/PIXEL'
    del fits_ext.header['EGAIN']
    fits_ext.header['HISTORY'] = "gain: gain corrected, now in E-/PIXEL"


def merge_data(fits_ext1, fits_ext2, fits_ext3, fits_ext4):
    rows, cols = fits_ext1.data.shape
    shape_output = (rows*2, cols*2)
    data_output = np.zeros(shape_output)

    data_output[     0:rows,    0:  cols] = fits_ext1.data
    data_output[     0:rows, cols:2*cols] = np.fliplr(fits_ext2.data)
    data_output[rows:2*rows,    0:  cols] = np.flipud(fits_ext4.data)
    data_output[rows:2*rows, cols:2*cols] = np.flip(fits_ext3.data, (0, 1))

    # Update header
    header_output = fits_ext1.header.copy()
    header_output['HISTORY'] = "merge: Merged 4 amplifiers into single frame"
    header_output['FILENAME'] = header_output['FILENAME'].replace('c1', '')
    header_output['DATASEC'] = '[{}:{},{}:{}]'.format(1, 2*cols, 1, 2*rows)
    header_output['TRIMSEC'] = '[{}:{},{}:{}]'.format(1, 2*cols, 1, 2*rows)

    header_output['ENOISE'] = np.mean((fits_ext1.header['ENOISE'],
                                       fits_ext2.header['ENOISE'],
                                       fits_ext3.header['ENOISE'],
                                       fits_ext4.header['ENOISE']))

    data_hdu = fits.PrimaryHDU(data_output, header=header_output)

    return data_hdu


def merge_error(fits_ext1, fits_ext2, fits_ext3, fits_ext4):
    rows, cols = fits_ext1.data.shape
    shape_output = (rows*2, cols*2)
    err_output = np.zeros(shape_output)

    err_output[     0:rows,    0:  cols] = np.sqrt(
                                            abs(fits_ext1.data) +
                                            fits_ext1.header['ENOISE']**2)

    err_output[     0:rows, cols:2*cols] = np.sqrt(
                                            abs(np.fliplr(fits_ext2.data)) +
                                            fits_ext2.header['ENOISE']**2)

    err_output[rows:2*rows,    0:  cols] = np.sqrt(
                                            abs(np.flipud(fits_ext4.data)) +
                                            fits_ext4.header['ENOISE']**2)

    err_output[rows:2*rows, cols:2*cols] = np.sqrt(
                                            abs(np.flip(fits_ext3.data, (0, 1))) +
                                            fits_ext3.header['ENOISE']**2)

    err_hdu = fits.ImageHDU(err_output)

    return err_hdu


def merge(fits_ext1, fits_ext2, fits_ext3, fits_ext4):
    data_hdu = merge_data(fits_ext1, fits_ext2, fits_ext3, fits_ext4)
    error_hdu = merge_error(fits_ext1, fits_ext2, fits_ext3, fits_ext4)
    output = fits.HDUList([data_hdu, error_hdu])
    return output


def grow_mask(arr, ngrow=1):
    nrows, ncols = arr.shape

    rows, cols = np.where(arr == 1)
    for r, c in zip(rows, cols):
        r_start = max(0, r-ngrow)
        r_end = min(nrows, r+ngrow+1)

        c_start = max(0, c-ngrow)
        c_end = min(ncols, c+ngrow+1)

        arr[r_start:r_end, c_start:c_end] = 1

    return arr


def cosray(fits_ext, ngrow=1, maxiter=4):
    rdnoise = fits_ext[0].header['ENOISE']
    img = fits_ext[0].data
    err = fits_ext[1].data

    c = cosmics.cosmicsimage(img, gain=1.0, readnoise=rdnoise, sigclip=30.0)
    c.run(maxiter=maxiter)

    mask_grown = grow_mask(c.mask, ngrow=ngrow)

    img[mask_grown] = np.nan
    err[mask_grown] = np.nan

    msg = 'cosray: Masked Cosmic Rays with LA Cosmic'
    fits_ext[0].header['HISTORY'] = msg


def dark_correction(fits_science, fits_dark):
    img_science = fits_science[0].data
    err_science = fits_science[1].data
    hdr_science = fits_science[0].header

    img_dark = fits_dark[0].data
    err_dark = fits_dark[1].data
    hdr_dark = fits_dark[0].header

    img = img_science - hdr_science['EXPTIME'] * (img_dark/hdr_dark['EXPTIME'])
    err = np.sqrt((err_science**2 + 
                   (hdr_science['EXPTIME']*err_dark/hdr_dark['EXPTIME'])**2))

    msg = 'dark: dark corrected'
    hdr_science['HISTORY'] = msg

    data_hdu = fits.PrimaryHDU(img, header=hdr_science)
    err_hdu = fits.ImageHDU(err)
    output = fits.HDUList([data_hdu, err_hdu])

    return output


def perform_basic_steps(filename):
    """
    - Bias subtraction, overscan columnwise.
    - Trims overscan region
    - Gain correct
    - Merge
    - Cosmic rays rejection

    filename should be the base filename, e.g., '<path>/b0001'.
    
    This should be used with lamps and twilights
    """
    c1 = fits.open(filename + 'c1.fits')
    c2 = fits.open(filename + 'c2.fits')
    c3 = fits.open(filename + 'c3.fits')
    c4 = fits.open(filename + 'c4.fits')
    try:
        input_fits = [c1[0], c2[0], c3[0], c4[0]]

        for i, c in enumerate(input_fits):
            print("Working in frame c{}".format(i+1))
            print("- Removing bias")
            debias(c)
            print("- Trimming overscan")
            trim(c)
            print("- Gain correcting")
            gain_correct(c)
            print("")
        print("- Stitching frames together")
        output = merge(*input_fits)
        print("- Removing cosmic rays")
        cosray(output)
    finally:
        c1.close()
        c2.close()
        c3.close()
        c4.close()

    return output


def basic_dark_steps(darkname):
    """
    Perform basic steps to create dark
    - Gain correct
    - Cosmic rays rejection

    filename should be the base filename, e.g., '<path>/bdark'.
    
    This should be used to create dark (before basic steps to science)
    """
    d1 = fits.open(darkname + 'c1.fits')
    d2 = fits.open(darkname + 'c2.fits')
    d3 = fits.open(darkname + 'c3.fits')
    d4 = fits.open(darkname + 'c4.fits')
    
    try:
        input_darks = [d1[0], d2[0], d3[0], d4[0]]
        for i, c in enumerate(input_darks):
            print("Working in dark frame c{}".format(i+1))
            print("- Gain correcting")
            gain_correct(c)
            print("")
        print("- Stitching dark frames together")
        output_dark = merge(*input_darks)
    finally:
        d1.close()
        d2.close()
        d3.close()
        d4.close()
    
    return output_dark


def basic_steps(science_filename, dark_filename):
    """
    - Bias subtraction, overscan columnwise.
    - Trims overscan region
    - Gain correct
    - Merge
    - Dark correction
    - Cosmic rays rejection

    science_filename should be the base filename, e.g., '<path>/b0001'.
    dark_filename should be the base filename, e.g., '<path>/dark'.
    
    This should be the first step for sciences
    """
    c1 = fits.open(science_filename + 'c1.fits')
    c2 = fits.open(science_filename + 'c2.fits')
    c3 = fits.open(science_filename + 'c3.fits')
    c4 = fits.open(science_filename + 'c4.fits')
    try:
        input_fits = [c1[0], c2[0], c3[0], c4[0]]

        for i, c in enumerate(input_fits):
            print("Working in frame c{}".format(i+1))
            print("- Removing bias")
            debias(c)
            print("- Trimming overscan")
            trim(c)
            print("- Gain correcting")
            gain_correct(c)
            print("")
        print("- Stitching frames together")
        output = merge(*input_fits)
    finally:
        c1.close()
        c2.close()
        c3.close()
        c4.close()

    try:
        dark = fits.open(dark_filename + '.fits')
        print("- Dark correcting")
        output = dark_correction(output, dark)
        print("- Removing cosmic rays")
        cosray(output)
    finally:
        dark.close()

    return output


def basic(sciences, dark_file, output_dir):
    output_dark = basic_dark_steps(dark_file)
    dark_basename = os.path.basename(dark_file)
    darkname = os.path.join(output_dir, dark_basename)
    output_dark.writeto(darkname + '.fits', overwrite=True)
    for i in range(len(sciences)):
        output = basic_steps(sciences[i], darkname)
        basename = os.path.basename(sciences[i])
        science_name = os.path.join(output_dir, basename + 'btgmdc.fits')
        output.writeto(science_name, overwrite=True)


