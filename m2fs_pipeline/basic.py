import os
import re
import numpy as np
from astropy.io import fits

from m2fs_pipeline import cosmics


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
                                            abs(np.flip(fits_ext3.data,
                                                        (0, 1))) +
                                            fits_ext3.header['ENOISE']**2)

    err_hdu = fits.ImageHDU(err_output)

    return err_hdu


def merge(fits_ext1, fits_ext2, fits_ext3, fits_ext4):
    data_hdu = merge_data(fits_ext1, fits_ext2, fits_ext3, fits_ext4)
    error_hdu = merge_error(fits_ext1, fits_ext2, fits_ext3, fits_ext4)
    output = fits.HDUList([data_hdu, error_hdu])
    return output


def grow_mask(arr, ngrow=2):
    nrows, ncols = arr.shape

    rows, cols = np.where(arr == 1)
    for r, c in zip(rows, cols):
        r_start = max(0, r-ngrow)
        r_end = min(nrows, r+ngrow+1)

        c_start = max(0, c-ngrow)
        c_end = min(ncols, c+ngrow+1)

        arr[r_start:r_end, c_start:c_end] = 1

    return arr


def cosray(fits_ext, ngrow=2, maxiter=4):
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


def gain_bias_trim_merge(filename):
    """
    - Gain correction
    - Bias substraction, overscan columnwise
    - Trims overscan region
    - Rotate and merge

    filename should be the base filename, e.g., '<path>/b0001'.

    This is for all the files except dark
    """
    c1 = fits.open(filename + 'c1.fits')
    c2 = fits.open(filename + 'c2.fits')
    c3 = fits.open(filename + 'c3.fits')
    c4 = fits.open(filename + 'c4.fits')
    try:
        input_fits = [c1[0], c2[0], c3[0], c4[0]]
        for i, c in enumerate(input_fits):
            print("Working in frame c{}".format(i+1))
            print("- Gain correcting")
            gain_correct(c)
            print("- Removing bias")
            debias(c)
            print("- Trimming overscan")
            trim(c)
            print("")
        print("- Stitching frames together")
        output = merge(*input_fits)
    finally:
        c1.close()
        c2.close()
        c3.close()
        c4.close()

    return output


def gain_merge(darkname):
    """
    - Gain correction
    - Rotate and merge

    darkname should be the base filename, e.g., '<path>/bdark'.

    This is for darks only
    """
    c1 = fits.open(darkname + 'c1.fits')
    c2 = fits.open(darkname + 'c2.fits')
    c3 = fits.open(darkname + 'c3.fits')
    c4 = fits.open(darkname + 'c4.fits')
    try:
        input_fits = [c1[0], c2[0], c3[0], c4[0]]
        for i, c in enumerate(input_fits):
            print("Working in frame c{}".format(i+1))
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

    return output


def basic_dark(dark, output_dir):
    """
    Basic steps reduction for dark file.
    - Gain
    - Merge

    It saves the resulting fits in output_dir.

    Parameters
    ----------
    dark : str
        dark base filename
    output_dir : str
        output directory
    Returns
    -------
    None
    """
    output_dark = gain_merge(dark)
    darkname = os.path.basename(dark)
    output_dark.writeto(os.path.join(output_dir, darkname + '.fits'),
                        overwrite=True)


def basic_steps(fits_fname, dark_fname, ngrow=2):
    """
    Basic steps reduction for the science observations.
    - Gain
    - Bias substraction
    - Trim
    - Merge
    - Dark correction
    - Cosmic rays rejection

    Parameters
    ----------
    science_fname : str
        science base filename
    dark_fname : str
        dark fits name, e.g. <path>/bdark.fits
    output_dir : str
        output directory
    Returns
    -------
    HDUList
    """
    dark_fits = fits.open(dark_fname)
    output = gain_bias_trim_merge(fits_fname)
    try:
        print("- Dark correcting")
        output = dark_correction(output, dark_fits)
        print("- Removing cosmic rays")
        cosray(output, ngrow=ngrow)
    finally:
        dark_fits.close()

    return output


def basic(sciences, twilights, lamps, dark, output_dir):
    basic_dark(dark, output_dir)
    darkname = os.path.join(output_dir, os.path.basename(dark) + '.fits')
    for i in range(len(sciences)):
        science_name = os.path.basename(sciences[i])
        output_science = basic_steps(sciences[i], darkname)
        output_science.writeto(os.path.join(output_dir,
                                            science_name + 'b.fits'),
                               overwrite=True)

    for i in range(len(twilights)):
        twilight_name = os.path.basename(twilights[i])
        output_twilight = basic_steps(twilights[i], darkname)
        output_twilight.writeto(os.path.join(output_dir,
                                             twilight_name + 'b.fits'),
                                overwrite=True)

    for i in range(len(lamps)):
        lamp_name = os.path.basename(lamps[i])
        output_lamp = basic_steps(lamps[i], darkname)
        output_lamp.writeto(os.path.join(output_dir,
                                         lamp_name + 'b.fits'),
                            overwrite=True)
