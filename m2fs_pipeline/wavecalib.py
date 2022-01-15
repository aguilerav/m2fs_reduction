import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.stats import biweight
from astropy.io import fits
from scipy.signal import correlate
from m2fs_pipeline import extract
from m2fs_pipeline import tracer

"""
This script makes the wavelength calibration.
It need the observation of two arc lamps (ThAr and NeHgArXe).
The routine is to make two different fits. First for the NeHgArXe lamp and then
it mix both lamps and it calibrates again.
The output is a file that contains the polynomial coefficients of the
calibration.
"""


def get_1Dspectrum(arr, tracing, fiber, apsize=4, method='biweight'):
    """
    It returns the spectrum of a specific fiber using extract module functions
    """
    nrows, ncols = arr.shape
    cols = np.arange(ncols)
    tracing_fiber = tracer.get_tracing_row(tracing, fiber, cols)
    if (all(np.isnan(tracing_fiber))):
        fiber_spectrum = cols*np.nan
    else:
        fiber_spectrum = extract.extract1D(arr, tracing, fiber,
                                           yaper=apsize,
                                           method=method, output=0)

    return fiber_spectrum


def extract_information(hg_data, th_data, hg_tracing, th_tracing, fiber,
                        nthresh=3):
    """
    It returns the spectrum and the peaks given a threshold for both arc lamps.
    """
    hg_spectrum = get_1Dspectrum(hg_data, hg_tracing, fiber)
    if (all(np.isnan(hg_spectrum))):
        hg_thresh = np.nan
        hg_peaks = np.nan
    else:
        hg_thresh = nthresh*tracer.peak_thresh(hg_spectrum)
        hg_peaks = find_peaks(hg_spectrum, hg_thresh)

    th_spectrum = get_1Dspectrum(th_data, th_tracing, fiber)
    if (all(np.isnan(th_spectrum))):
        th_thresh = np.nan
        th_peaks = np.nan
    else:
        th_thresh = (nthresh+2)*tracer.peak_thresh(th_spectrum)
        th_peaks = find_peaks(th_spectrum, th_thresh)

    return hg_spectrum, hg_peaks, th_spectrum, th_peaks


def find_peaks(lamp_spectrum, thresh, ncols_each_side=5):
    """
    Determine precise peaks for each emission given a threshold
    """
    pre_peaks = tracer.peak_row_indices(lamp_spectrum, thresh)

    peaks = []

    # There is an error when the peak is in the finals or in the firsts pixels
    # If that is the  case, the func take a smaller window
    for i, ref_peak in enumerate(pre_peaks):
        if ref_peak + ncols_each_side > len(lamp_spectrum)-1:
            continue
        elif ref_peak - ncols_each_side < 0:
            continue
        else:
            peak = tracer.gaussian_peak(lamp_spectrum, ref_peak,
                                        half_width=ncols_each_side)
            if ~np.isnan(peak):
                peaks.append(peak)

    peaks = np.array(peaks)
    return peaks


def merit(coeff, peaks, linelist):
    """
    Return the square distance between lines and the wavelenght peaks modeled
    """
    transformed_pixels = np.polyval(coeff, peaks)
    distances = transformed_pixels[:, np.newaxis] - linelist[np.newaxis, :]
    min_distances = np.min(np.fabs(distances), axis=1)
    output = np.sum(min_distances**2)
    return output


def peaks_distance(pixel_peaks, linelist, coeff):
    """
    It returns the distance between the peaks wavelength and the line
    wavelength
    """
    wavelength_peaks = np.polyval(coeff, pixel_peaks)
    distances = wavelength_peaks[:, np.newaxis] - linelist[np.newaxis, :]
    idx_of_identified_lines = np.argmin(np.fabs(distances), axis=1)
    separation = wavelength_peaks - linelist[idx_of_identified_lines]
    return separation


def rms(peaks_distances):
    """
    Standard deviation of all the distances between the peaks wavelength and
    the lines wavelength
    """
    rms = np.std(peaks_distances)
    return rms


def create_peaks_wavelength(hg_pixel_peaks, hg_coeff, hg_lines, th_pixel_peaks,
                            th_lines):
    """
    It takes all the peaks of the NeHgArXe arc lamp and it adds the ThAr peaks
    in the wavelength range that is not covered by the NeHgArXe lamp. It returns
    the peaks in pixel value and in wavelength (the closest line) using the
    calibration of the NeHgArXe lamp.
    """
    hg_wave_peaks = np.polyval(hg_coeff, hg_pixel_peaks)
    th_wave_peaks = np.polyval(hg_coeff, th_pixel_peaks)
    peaks = []
    for i in range(len(th_pixel_peaks)):
        if th_pixel_peaks[i] < min(hg_pixel_peaks):
            pixel_peak = th_pixel_peaks[i]
            closest_line_index = (np.abs(th_lines-th_wave_peaks[i])).argmin()
            wave_peak = th_lines[closest_line_index]
            peaks.append([pixel_peak, wave_peak])
    
    for i in range(len(hg_pixel_peaks)):
        pixel_peak = hg_pixel_peaks[i]
        closest_line_index = (np.abs(hg_lines-hg_wave_peaks[i])).argmin()
        wave_peak = hg_lines[closest_line_index]
        peaks.append([pixel_peak, wave_peak])

    return np.array(peaks)


def blind_calibration(peaks, lines_list, sigma=2.5,
                         ptest=[0, 0, -5e-6, 1.01, 4520]):
    """
    It takes some pixel peaks and a list of lines. It calculates the 
    polynomial coefficients for the peaks and their closes line in the list.
    It eliminates peaks above 2.5 sigma. (it can be change with the sigma
    variable)
    """
    coeffs = minimize(merit, ptest, args=(peaks, lines_list),
                      method='Nelder-Mead').x
    distance = peaks_distance(peaks, lines_list, coeffs)
    dispersion = rms(distance)
    abs_distance = np.fabs(distance)
    filter_peaks = peaks[abs_distance < sigma*dispersion]
    if (len(peaks) == len(filter_peaks)):
        return coeffs, dispersion
    else:
        return blind_calibration(filter_peaks, lines_list, ptest=coeffs)


def forced_calibration(peaks, wavelength, sigma=2.5,
                       ptest=[0, 0, -5e-6, 1.01, 4520]):
    """
    It makes the same as blind_calibration but this time it is not a list of
    lines, it is the expected wavelengths that is given.
    It returns the wavelength calibration for a list of peaks and corresponding
    wavelength. It leaves out the peaks that has more than 2.5*sigma distance.
    """
    coeffs = minimize(merit, ptest, args=(peaks, wavelength),
                           method='Nelder-Mead').x
    distance = peaks_distance(peaks, wavelength, coeffs)
    rms = np.std(distance)
    clean_peaks = peaks[np.fabs(distance) < sigma*rms]
    if len(peaks) == len(clean_peaks):
        return coeffs, rms
    else:
        return forced_calibration(clean_peaks, wavelength, ptest=coeffs)


def fiber_calibration(hg_peaks, hg_lines, th_peaks, th_lines, sigma=2.5,
                      ptest=[0, 0, -5e-6, 1.01, 4520]):
    """
    It calibrates a fiber using the NeHgArXe and the ThAr lamps. First it
    calibrates the NeHgArXe lamp, then it takes the lines in ThAr that are
    in the range not covered by the NeHgArXe lamp. It adds this and then it
    calibrates again with this information.
    """
    hg_coeffs, hg_rms = blind_calibration(hg_peaks, hg_lines, sigma=sigma,
                                          ptest=ptest)
    pixel_wave = create_peaks_wavelength(hg_peaks, hg_coeffs, hg_lines,
                                         th_peaks, th_lines)
    coeffs, rms = forced_calibration(pixel_wave[:,0], pixel_wave[:,1],
                                     sigma=sigma, ptest=hg_coeffs)
    return coeffs, rms


def crosscorrelate_offset(arr1, arr2):
    """
    The first array is offset in the return value according the second array
    """
    arr1_notNan = arr1[~np.isnan(arr1)]
    arr2_notNan = arr2[~np.isnan(arr2)]
    delta_pixels = np.arange(1-len(arr1_notNan), len(arr2_notNan))

    xcorr = correlate(arr1_notNan, arr2_notNan)
    offset = delta_pixels[xcorr.argmax()]

    return offset


def sweep_mapping_coeffs(hg_data, th_data, hg_lines, th_lines, hg_trace,
                         th_trace, sequence, prefiber_spectrum, prefiber_coeff,
                         output, all_rms, sigma=2.5, nthresh=3):
    """
    Calibrates fibers following a sequence. It needs the first fiber already
    calibrated
    """
    for fiber in sequence[1:]:
        sys.stdout.write('\rCalibrating fiber: ' + str(fiber))
        sys.stdout.flush()
        fiber_info = extract_information(hg_data, th_data, hg_trace, th_trace,
                                         fiber, nthresh=nthresh)
        if (all(np.isnan(fiber_info[0]))):
            fiber_coeff = prefiber_coeff*np.nan
            fiber_rms = np.nan
            output[fiber] = fiber_coeff
            all_rms[fiber] = fiber_rms
        else:
            offset = crosscorrelate_offset(fiber_info[0], prefiber_spectrum)
            prefiber_coeff[-1] = (prefiber_coeff[-1] -
                                 (offset*prefiber_coeff[len(prefiber_coeff)-2]))
            fiber_coeff, fiber_rms = fiber_calibration(fiber_info[1], hg_lines,
                                                       fiber_info[3], th_lines,
                                                       sigma=sigma, 
                                                       ptest=prefiber_coeff)
            output[fiber] = fiber_coeff
            all_rms[fiber] = fiber_rms
            prefiber_spectrum = np.copy(fiber_info[0])
            prefiber_coeff = np.copy(fiber_coeff)
    
    print('')


def coeffs_map(hg_data, th_data, hg_lines, th_lines, hg_trace, th_trace,
               nthresh=3, sigma=2.5, ptest=[0, 0, -5e-6, 1.01, 4520]):
    """
    It calibrates all the fibers starting from the center with a first guess
    """
    nfibers = len(hg_trace)
    center_fiber = round(nfibers/2)
    center_info = extract_information(hg_data, th_data, hg_trace, th_trace,
                                      center_fiber, nthresh=nthresh)
    while (all(np.isnan(center_info[0]))):
        center_fiber = center_fiber + 1
        center_info = extract_information(hg_data, th_data, hg_trace, th_trace,
                                          center_fiber, nthresh=nthresh)
    center_coeffs, center_rms = fiber_calibration(center_info[1], hg_lines,
                                                  center_info[3], th_lines,
                                                  sigma=sigma, ptest=ptest)
    
    #Wavelength calibration in the center fiber
    prefiber_coeff = np.copy(center_coeffs)
    prefiber_spectrum = np.copy(center_info[0])
    output = [0]*nfibers
    all_rms = np.zeros(nfibers)
    output[center_fiber] = center_coeffs
    all_rms[center_fiber] = center_rms

    #From center down
    sequence_swipe_down = range(center_fiber, nfibers)
    sweep_mapping_coeffs(hg_data, th_data, hg_lines, th_lines, hg_trace,
                         th_trace, sequence_swipe_down, prefiber_spectrum,
                         prefiber_coeff, output, all_rms, sigma=sigma,
                         nthresh=nthresh)
    
    #From center up
    prefiber_coeff = np.copy(center_coeffs)
    prefiber_spectrum = np.copy(center_info[0])
    sequence_swipe_up = range(center_fiber, -1, -1)
    sweep_mapping_coeffs(hg_data, th_data, hg_lines, th_lines, hg_trace,
                         th_trace, sequence_swipe_up, prefiber_spectrum,
                         prefiber_coeff, output, all_rms, sigma=sigma,
                         nthresh=nthresh)
    
    return output, all_rms


def calibration(NeHgArXe_fits_fname, ThAr_fits_fname, output_dir,
               NeHgArXe_lines_fname, ThAr_lines_fname,
               NeHgArXe_tracing_fname, ThAr_tracing_fname, nthresh=3, sigma=2.5,
               ptest=[0, 0, -5e-6, 1.01, 4520]):
    
    NeHgArXe_data = fits.getdata(NeHgArXe_fits_fname)
    NeHgArXe_filename = os.path.basename(NeHgArXe_fits_fname).replace('.fits',
                                                                      '')
    NeHgArXe_tracing = np.genfromtxt(NeHgArXe_tracing_fname)
    NeHgArXe_lines = np.genfromtxt(NeHgArXe_lines_fname)

    ThAr_data = fits.getdata(ThAr_fits_fname)
    ThAr_filename = os.path.basename(ThAr_fits_fname).replace('.fits', '')
    ThAr_tracing = np.genfromtxt(ThAr_tracing_fname)
    ThAr_lines = np.genfromtxt(ThAr_lines_fname)

    coeffs, rms = coeffs_map(NeHgArXe_data, ThAr_data, NeHgArXe_lines, 
                             ThAr_lines, NeHgArXe_tracing, ThAr_tracing,
                             nthresh=nthresh, sigma=sigma, ptest=ptest)
    
    coeffs_file = open(os.path.join(output_dir,
                                    ThAr_filename+'_wave_coeffs.out'),
                       'w')
    rms_file = open(os.path.join(output_dir,
                                    ThAr_filename+'_wave_rms.out'),
                       'w')
    
    np.savetxt(coeffs_file, coeffs)
    np.savetxt(rms_file, rms)
    coeffs_file.close()
    rms_file.close()


def get_wavelength(wave_file, nfiber, cols):
    """
    Return wavelength coefficients calibration in one fiber
    """
    wave_coeff = wave_file[nfiber]

    if(all(np.isnan(wave_coeff))):
        return cols*np.nan
    return np.polyval(wave_coeff, cols)


def wave_values(wave_file, ncols):
    nfibers = len(wave_file)
    cols = np.arange(ncols)
    wave_vals = np.zeros((nfibers, ncols))
    for fiber in range(nfibers):
        wave_vals[fiber, :] = get_wavelength(wave_file, fiber, cols)
    
    return wave_vals