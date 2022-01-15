import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from scipy.optimize import minimize
from astropy.stats import biweight
from astropy.io import fits
from scipy.signal import correlate
from scipy.optimize import curve_fit
from m2fs_pipeline import extract
from m2fs_pipeline import fibermap
from m2fs_pipeline import tracer
from m2fs_pipeline import wavecalib

"""
This script makes the sky substraction.
The output is a new science image where the sky is substracted.
It needs the fibermap of the fibers to identify the sky fibers.
"""


def gaussian(x, a, b, c, d):
    return a*np.exp(-(x-b)**2/(2*c**2))+d


def skyline_correction(wave, flux, sky_waves, wave_aperture=5):
    correction = np.zeros(len(sky_waves))
    for i in range(len(sky_waves)):
        wave_window = np.where((sky_waves[i]-wave_aperture < wave) & 
                                (wave < sky_waves[i]+wave_aperture))[0]

        new_wave = wave[wave_window]
        new_flux = flux[wave_window]

        nan_count = np.count_nonzero(np.isnan(new_flux))
        L = len(new_flux)
        new_wave = new_wave[~np.isnan(new_flux)]
        new_flux = new_flux[~np.isnan(new_flux)]

        if nan_count >= 0.4*L:
            correction[i] = np.nan
            continue
        new_wave = new_wave - sky_waves[i]
        coef, pcov = curve_fit(gaussian, new_wave, new_flux,
                               p0=[1000, 0, 2, 0], maxfev=2000000)
        correction[i] = coef[1]
        if (abs(coef[1]) > 5):
            correction[i] = np.nan

    if all(np.isnan(correction)):
        return wave, np.nan
    else:
        return wave - np.nanmean(correction), np.nanmean(correction)


def verify(knots, data):
    """
    Remove consecutive knots with no datapoints in between
    """
    i = 0
    while i < (len(knots)-1):
        lower = knots[i]; upper = knots[i+1]
        aux = np.where((data > lower) & (data < upper))[0]
        if(len(aux) == 0):
            knots = np.delete(knots, i)
        else:
            i = i+1
    return knots


def outliers(wave, flux, width=2., times_sigma=2):
    """
    Reject outliers at more than times_sigma sigmas
    """
    auxarray = np.arange(wave[0], wave[-1], width)

    outers = np.array([])
    for i in range(len(auxarray)-1):
        indices = np.where((wave > auxarray[i]) & (wave < auxarray[i+1]))[0]
        median = np.nanmedian(flux[indices])
        sigma = np.nanstd(flux[indices])

        aux = np.where(abs(flux[indices]-median) < times_sigma*sigma)[0]
        outers = np.append(outers, indices[aux])

    return outers.astype(int)


def find(array, array1):
    """
    Associate 1D vector values to 2D vector
    """
    nrows, ncols = array.shape
    output = np.zeros((nrows, ncols))

    for i in range(nrows):
        for j in range(ncols):
            output[i, j] = array1[i]

    return output


def get_wavelength_dispersion(wave_data, ncols=2048):
    nfibers = len(wave_data)
    disp1d = np.zeros((nfibers, ncols))
    disp = wave_data[:, 1]
    for fiber in range(nfibers):
        disp1d[fiber, :] = np.tile(disp[fiber], ncols)

    return disp1d


def get_median_per_fiber(data_1D, data_err_1D):
    nfibers, ncols = data_1D.shape
    median_data = np.zeros(nfibers)
    median_err = np.zeros(nfibers)
    for fiber in range(nfibers):
        if (all(np.isnan(data_1D[fiber, :]))):
            median_data[fiber] = np.nan
            median_err[fiber] = np.nan
        else:            
            median_data[fiber] = np.nanmedian(data_1D[fiber, :])
            median_err[fiber] = np.nanmedian(data_err_1D[fiber, :])

    return median_data, median_err


def change_wave_by_skyline(wave_1D, data_1D, sky_wave):
    nfibers, ncols = data_1D.shape
    offset = np.zeros(nfibers)
    for fiber in range(nfibers):
        wave_1D[fiber, :], offset[fiber] = skyline_correction(wave_1D[fiber, :],
                                                              data_1D[fiber, :],
                                                              sky_wave)

    return wave_1D, offset


def common_wavelength(wave_1D):
    nfibers, ncols = wave_1D.shape
    wmin = np.nanmin(wave_1D[np.where(wave_1D != 0)])
    wmax = np.nanmax(wave_1D[np.where(wave_1D != 0)])
    w_arr = np.linspace(wmin, wmax, ncols)

    return w_arr, wmin, wmax


def mode(arr, size):
    clean_arr = arr[~np.isnan(arr)]
    N = len(clean_arr)
    rep = np.zeros(N)

    for i in range(N):
        rep[i] = len(np.where(abs(clean_arr-clean_arr[i]) <= size)[0])

    return arr[rep.argmax()]


def get_median_mode(median_data, median_err):
    flux_size = 0.01*np.nanmedian(median_data)
    err_size = 0.1*flux_size
    mode_flux = mode(median_data, flux_size)
    mode_err = mode(median_err, err_size)

    return mode_flux, mode_err


def collapse_sort_filter(wave, flux, err):
    sort_wave = np.argsort(wave, axis=None)
    wave = wave.flatten()[sort_wave]
    flux = flux.flatten()[sort_wave]
    err = err.flatten()[sort_wave]

    filter_nan = np.where((np.isfinite(flux)) &
                          (np.isfinite(err)))[0]
    wave = wave[filter_nan]
    flux = flux[filter_nan]
    err = err[filter_nan]

    filter_zero = np.where((flux != 0) & (err > 0))[0]
    wave = wave[filter_zero]
    flux = flux[filter_zero]
    err = err[filter_zero]

    return wave, flux, err


def define_knots(wave_1D, data_1D, wave, flux, fibers, edges=1, eps=1e-7,
                 spacing=1, power=0.1):
    """
    Define the knots for the spline
    """
    auxwave = np.arange(wave[0], wave[-1], edges)
    auxflux = np.zeros(len(auxwave))
    for i in range(len(fibers)):
        clean = np.where(np.isfinite(data_1D[fibers[i], :]))[0]
        wave_clean = wave_1D[fibers[i], :][clean]
        data_clean = data_1D[fibers[i], :][clean]
        f = inter.interp1d(wave_clean, data_clean, bounds_error=False)
        for j in range(len(auxflux)):
            if(np.isfinite(f(auxwave[j]))):
                auxflux[j] = auxflux[j] + f(auxwave[j])
    auxflux = auxflux/len(fibers)

    density = np.histogram(wave, bins=len(np.arange(wave[0], wave[-1], edges)))

    difference = np.zeros(len(auxflux)-1)
    for i in range(len(difference)):
        difference[i] = abs(auxflux[i+1] - auxflux[i])

    spacing_gradient = np.nanmedian(difference)
    gradient = np.gradient(auxflux, spacing_gradient)

    new_density = np.array([wave[0]])
    for i in range(len(density[1])-1):
        pre_aux = spacing/abs(gradient[i])**power
        aux = np.arange(density[1][i], density[1][i+1], pre_aux)[0:-1]
        new_density = np.append(new_density, aux)

    new_density = verify(new_density, wave)
    new_density = new_density[5:-5]

    return new_density


def select_sky_fibers(start_fiber, end_fiber, tracefile, median_data,
                      mode_flux, thresh=1.2):
    nfibers = len(median_data)
    pre_fibers = np.arange(start_fiber, end_fiber + 1, 1)
    pre_fibers = pre_fibers.astype(int)
#   Select real fibers
    fibers = []
    for i in pre_fibers:
        if (all(np.isnan(tracefile[i]))):
            continue
        fibers.append(i)
    fibers = np.array(fibers)
    if len(np.where((median_data[fibers] >= mode_flux/thresh) & 
                    (median_data[fibers] <= mode_flux*thresh))[0]) != 0:

        fib_sky = fibers[np.where((median_data[fibers] >= mode_flux/thresh) &
                                  (median_data[fibers] <= mode_flux*thresh))[0]]
        return fib_sky
    else:
        if (end_fiber == nfibers):
            s_fiber = start_fiber - len(fibers)
            return select_sky_fibers(s_fiber, end_fiber, tracefile,
                                     median_data, mode_flux, thresh=thresh)
        elif (start_fiber == 0):
            e_fiber = end_fiber + len(fibers)
            return select_sky_fibers(start_fiber, e_fiber, tracefile,
                                     median_data, mode_flux, thresh=thresh)
        else:
            s_fiber = start_fiber - len(fibers)
            e_fiber = end_fiber + len(fibers)
            return select_sky_fibers(s_fiber, e_fiber, tracefile, median_data,
                                     mode_flux, thresh=thresh)


def select_block_fibers(block, nblocks=8, total_fibers=128):
    """
    Return all the indexes in one block
    """

    fib_p_block = int(total_fibers/nblocks)
    start_fiber = block*fib_p_block
    end_fiber = (block*fib_p_block + fib_p_block)
    fibers = np.arange(start_fiber, end_fiber, 1)
    return fibers.astype(int)


def create_skymodel(sky_model, fibers, wave_1D, tracefile, sset,
                    cols_array, row_aper_eachside=2):
    nrows, ncols = sky_model.shape
    for i in range(len(fibers)):
        sys.stdout.write('\rCreating Sky Model: ' + str(fibers[i]))
        sys.stdout.flush()
        trace_peak = tracer.get_tracing_row(tracefile, fibers[i], cols_array)

        if (len(trace_peak[~np.isnan(trace_peak)]) == 0):
            continue
        fiber_wave = wave_1D[fibers[i], :]
        fiber_sky = sset(fiber_wave)
        for j in range(ncols):
            trace_min = int(np.round(trace_peak[j]) - row_aper_eachside)
            trace_max = int(np.round(trace_peak[j]) + row_aper_eachside)
            sky_model[trace_min:trace_max+1, j] = fiber_sky[j]
    print('')


def substract_sky(arr, arr_substracted, fibers, wave_1D, tracefile,
                  sset, cols_array, row_aper_eachside=2):
    nrows, ncols = arr.shape
    for i in range(len(fibers)):
        sys.stdout.write('\rSubstracting Sky: ' + str(fibers[i]))
        sys.stdout.flush()
        trace_peak = tracer.get_tracing_row(tracefile, fibers[i], cols_array)

        if (len(trace_peak[~np.isnan(trace_peak)]) == 0):
            continue
        fiber_wave = wave_1D[fibers[i], :]
        fiber_sky = sset(fiber_wave)
        for j in range(ncols):
            trace_min = int(np.round(trace_peak[j]) - row_aper_eachside)
            trace_max = int(np.round(trace_peak[j]) + row_aper_eachside)
            arr_substracted[trace_min:trace_max+1, j] = (arr[trace_min:trace_max+1,
                                                             j] -
                                                         fiber_sky[j])
    print('')


def sky_model(data, wave_1D, data_1D, data_err_1D, median_data, median_err,
              mode_flux, mode_err, tracefile, blocks=8, thresh=1.2, edges=1,
              eps=1e-7, spacing=1, power=0.1, sky_fibers='no'):
    nrows, ncols = data.shape
    cols_array = np.arange(ncols)
    nfibers = len(tracefile)
    sky_model = np.zeros((nrows, ncols))
    data_substracted = np.zeros((nrows, ncols))
    for b in range(blocks):
        fibers = select_block_fibers(b, nblocks=blocks, total_fibers=nfibers)
        start_fiber = fibers[0]
        end_fiber = fibers[-1]

#       Use all fibers for sky with a threshold
        if type(sky_fibers) == str:
            fib_sky = select_sky_fibers(start_fiber, end_fiber, tracefile,
                                        median_data, mode_flux, thresh=thresh)
        else:
            fib_sky = sky_fibers

        print('Using ' + str(len(fib_sky)) + ' sky fibers for block ' +
              str(b+1))

        wave_sky = wave_1D[fib_sky, :]
        flux_sky = data_1D[fib_sky, :]
        err_sky = data_err_1D[fib_sky, :]

#       collapse and filter nan and 0
        wave_sky, flux_sky, err_sky = collapse_sort_filter(wave_sky, flux_sky,
                                                           err_sky)

#       defining the knots for the spline
        density = define_knots(wave_1D, data_1D, wave_sky, flux_sky, fib_sky,
                               edges=edges, eps=eps, spacing=spacing,
                               power=power)

#       reject outliers
        out = outliers(wave_sky, flux_sky, width=3.)
        wave_sky = wave_sky[out]
        flux_sky = flux_sky[out]

#       spline fitting
        sset = inter.LSQUnivariateSpline(wave_sky, flux_sky, t=density, k=5)
#       perform sky substraction for the block
        create_skymodel(sky_model, fibers, wave_1D, tracefile, sset,
                        cols_array)
        substract_sky(data, data_substracted, fibers, wave_1D,
                      tracefile, sset, cols_array)
    return sky_model, data_substracted


def data_sky_substraction(data, sky_model):
    nrows, ncols = data.shape
    data_sub = data - sky_model
    return data_sub


def correction(data_sub, data_err, wave_1D, airmass, extinc_file, tracefile,
               cols_array, row_aper_eachside=2):
    """
    Correct for airmass and collapse to store residuals
    """
    nrows, ncols = data_sub.shape
    nfibers = len(tracefile)
    wave_atm = np.genfromtxt(extinc_file, comments='#')[:, 0]
    k_atm = np.genfromtxt(extinc_file, comments='#')[:, 1]
    extinc_func = inter.interp1d(wave_atm, k_atm)
    for fiber in range(nfibers):
        sys.stdout.write('\rCorrecting fiber: ' + str(fiber))
        sys.stdout.flush()
        trace_peak = tracer.get_tracing_row(tracefile, fiber, cols_array)

        if (len(trace_peak[~np.isnan(trace_peak)]) == 0):
            continue
        for j in range(ncols):
            k_atm_interp = extinc_func(wave_1D[fiber, :])
            aux = 10.0**(0.4*k_atm_interp*airmass)
            trace_min = int(np.round(trace_peak[j]) - row_aper_eachside)
            trace_max = int(np.round(trace_peak[j]) + row_aper_eachside)
            data_sub[trace_min:trace_max+1, j] = data_sub[trace_min:trace_max+1,
                                                          j]*aux[j]
            data_err[trace_min:trace_max+1, j] = data_err[trace_min:trace_max+1,
                                                          j]*aux[j]
    print('')
    return data_sub, data_err


def skysub(science_fname, trace_fname, wave_fname, extinction_fname,
           fibermap_fname, output_dir, spectro='b', type_sky=0,
           sky_waves=[5577.338, 6300.304, 6498.729]):
    """
    type_sky = 0 using all fibers for sky
    type_sky = 1 using sky fibers for sky
    """
    aux = fits.open(science_fname)
    data = aux[0].data
    hdr = aux[0].header
    data_err = aux[1].data
    hdr_err = aux[1].header
    aux.close()
    nrows, ncols = data.shape
    filename = os.path.basename(science_fname).replace('.fits', '')

    tracefile = np.genfromtxt(trace_fname)
    wavecalibfile = np.genfromtxt(wave_fname)
    nfibers = len(tracefile)

    skyfibers_name, skyfibers_number = fibermap.fibers_id('S', spectro, 
                                                          fibermap_fname)

    wave_1D = wavecalib.wave_values(wavecalibfile, ncols)

    data_1D, data_err_1D = extract.fibers_extraction(data, data_err, tracefile)
    wave_1D, offset = change_wave_by_skyline(wave_1D, data_1D, sky_waves)
    median_data, median_err = get_median_per_fiber(data_1D, data_err_1D)
    mode_flux, mode_err = get_median_mode(median_data, median_err)

    if type_sky == 0:
        skymodel, data_sub = sky_model(data, wave_1D, data_1D, data_err_1D,
                                       median_data, median_err, mode_flux,
                                       mode_err, tracefile)
    else:
        skymodel, data_sub = sky_model(data, wave_1D, data_1D, data_err_1D,
                                       median_data, median_err, mode_flux,
                                       mode_err, tracefile,
                                       sky_fibers=skyfibers_number)

#   data_sub = data_sky_substraction(data, skymodel)
    airmass = hdr['AIRMASS']
    output = correction(data_sub, data_err, wave_1D, airmass,
                        extinction_fname, tracefile, 
                        np.arange(ncols))
    data_sub, data_err = output

#   write sky substracted frame
    hdr['HISTORY'] = 'skysub: Substracted Sky'
    for i in range(nfibers):
        hdr['WDELT' + str(i)] = str(offset[i])
    hdr['HISTORY'] = 'exptime: Divided by exposure time'
    data1 = fits.PrimaryHDU(data_sub/hdr['EXPTIME'], header=hdr)
    data2 = fits.ImageHDU(data_err/hdr['EXPTIME'], header=hdr_err)
    data3 = fits.ImageHDU(skymodel)
    fits_file_data = fits.HDUList([data1, data2, data3])
    fits_file_data.writeto(os.path.join(output_dir, filename + 's.fits'),
                           overwrite=True)
