import os
import sys
import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def median_nwidth_cols(arr, center_col, ncols_each_side=1):
    """
    Median combines an array taking ncols_each_side cols around a central
    column
    """
    start_col = center_col - ncols_each_side
    end_col = center_col + ncols_each_side + 1
    tmp = arr[:, start_col:end_col].copy()
    output = np.nanmedian(tmp, axis=1)
    #ref_value = np.nanmin(tmp) - 1e6
    #tmp[~np.isfinite(tmp)] = ref_value
    #output = np.median(tmp, axis=1)
    #output[output == ref_value] = np.nan
    return output


def median_nwidth_cols_multiple_centers(arr, center_cols, ncols_each_side=1):
    output = np.zeros((arr.shape[0], len(center_cols)))
    for i, center_col in enumerate(center_cols):
        median_at_center_col = median_nwidth_cols(
                                            arr, center_col,
                                            ncols_each_side=ncols_each_side)
        output[:, i] = median_at_center_col
    return output


def peak_thresh(arr):
    """ Determine the threshold for identification of a peak.
    """
    first_cut = np.nanmedian(arr)
    with np.errstate(invalid='ignore'):
        bkg = arr[arr < first_cut]
    #bkg = arr[arr < first_cut]
    threshold = bkg.mean() + 10 * bkg.std()
    return threshold


def peak_row_indices(arr, peak_thresh, ystart=0):
    peaks = find_peaks(arr, height=peak_thresh, distance=6)[0]
    peaks = peaks[ystart:]
    return peaks


def gaussian(x, *pars):
    A, mu, sigma, offset = pars
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset


def gaussian_peak(arr, ref_peak, half_width=4, sigma_ref=2.):
    ref_index = int(round(ref_peak))
    x = np.arange(ref_index-half_width, ref_index+half_width+1)
    y = arr[x]

    finite = np.isfinite(y)
    y = y[finite]
    x = x[finite]

    # print(x)

    p0 = [arr[ref_index], ref_index, sigma_ref, 0]
    # print(p0)

    bounds_inf = [0, ref_index-half_width, 0, 0]
    bounds_sup = [np.inf, ref_index+half_width+1, 2*sigma_ref, np.inf]
    try:
        pstar, pcov = curve_fit(gaussian, x, y, p0=p0,
                                bounds=(bounds_inf, bounds_sup))
    except:
        pstar = [np.nan, np.nan, np.nan, np.nan]
    return pstar[1]


def parabola_peak(arr, ref_peak, half_width=4):
    ref_index = int(round(ref_peak))
    x = np.arange(ref_index-half_width, ref_index+half_width+1)
    y = arr[x]

    finite = np.isfinite(y)
    y = y[finite]
    x = x[finite]

    p2 = np.polyfit(x, y, 2)
    peak = -0.5 * p2[1] / p2[0]
    return peak


def all_peaks(func, collapsed_rows_arr, ref_peaks, half_widths=4):
    """ Determine precise centers for each fiber.

    Fit a peaky function (gaussian, parabola) to get row coordinate for the
    peak of each detected fiber. The max values is passed and a width to fit
    the function that will find a more precise center.
    """
    peaks = np.zeros(len(ref_peaks))

    for i, ref_peak in enumerate(ref_peaks):
        # print(ref_peak)
        peaks[i] = func(collapsed_rows_arr, ref_peak, half_width=half_widths)
    return peaks


def sweep_mapping_peaks(arr_slices, starting_peaks, sequence, output):
    ref_peaks = starting_peaks.copy()
    i_prev = sequence[0]
    for i in sequence[1:]:
        output[:, i] = all_peaks(gaussian_peak, arr_slices[:, i], ref_peaks)
        ref_peaks = output[:, i].copy()
        # But replace the failed peaks with the values in the previous set
        # If those also failed we are in trouble
        failed_peaks = ~np.isfinite(ref_peaks)
        ref_peaks[failed_peaks] = output[:, i_prev][failed_peaks]


def peak_map(arr, ref_col_distance=60, bin_width=35):
    """ Determine fiber row-centers for the full array at multiple columns.
    """
    nrows, ncols = arr.shape
    ncols_each_side = int((bin_width-1)/2)

    ref_cols = np.arange(ref_col_distance, ncols, ref_col_distance)
    center_col_indx = int(len(ref_cols)/2)

    median_rows = median_nwidth_cols_multiple_centers(
                                arr, ref_cols, ncols_each_side=ncols_each_side)

    # Reference peaks in the center reference column
    thresh = peak_thresh(median_rows[:, center_col_indx])
    ref_peaks_center = peak_row_indices(median_rows[:, center_col_indx],
                                        thresh)

    row_positions = np.zeros((len(ref_peaks_center), len(ref_cols)))
    row_positions[:, center_col_indx] = all_peaks(
                                            gaussian_peak,
                                            median_rows[:, center_col_indx],
                                            ref_peaks_center)

    # From center to right
    ref_peaks_start = row_positions[:, center_col_indx]
    sequence_swipe_right = range(center_col_indx, len(ref_cols))
    sweep_mapping_peaks(median_rows, ref_peaks_start, sequence_swipe_right,
                        row_positions)

    # From center to left
    sequence_swipe_left = range(center_col_indx, -1, -1)
    sweep_mapping_peaks(median_rows, ref_peaks_start, sequence_swipe_left,
                        row_positions)

    return ref_cols, row_positions


def trace(fits_fname, output_dir, step_size=40,
          bin_width=35, degree=4):
    data = fits.getdata(fits_fname)

    basename = os.path.basename(fits_fname).replace('.fits', '')

    col_positions, row_positions = peak_map(data, ref_col_distance=step_size,
                                            bin_width=bin_width)
    nfibers = len(row_positions)

    # Write peak positions to file
    peaks_to_write = np.vstack((col_positions, row_positions))
    peakmap_fname = os.path.join(output_dir, basename+'_peakmap.dat')
    with open(peakmap_fname, 'w') as peakmap_file:
        np.savetxt(peakmap_fname, peaks_to_write)

    # Calculate polynomial fits
    coeff = np.zeros((nfibers, degree+1))
    for fiber in range(nfibers):
        valid = np.isfinite(row_positions[fiber])
        fiber_coeff = np.polyfit(col_positions[valid],
                                 row_positions[fiber][valid],
                                 deg=degree)
        coeff[fiber] = fiber_coeff

    # Write coeffs to file
    trace_fname = os.path.join(output_dir, basename+'_trace_coeffs.out')
    with open(trace_fname, 'w') as coeff_file:
        np.savetxt(trace_fname, coeff)


def get_tracing_row(tracing, fiber, cols):
    """
    It returns the row indexes of the complete fiber
    """
    coeff = tracing[fiber, :]

    if (all(np.isnan(coeff))):
        return cols*np.nan
    else:
        return np.polyval(coeff, cols)