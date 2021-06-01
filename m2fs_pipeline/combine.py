import os
import sys
import warnings
import numpy as np
import scipy.interpolate as inter
from astropy.io import fits
from astropy.stats import biweight


# IMAGE COMBINATION ROUTINE (PIXEL BY PIXEL)
def combine(files, filename, output_dir, exptime=1, thresh=30000):
    """
    Image combination routine using mean (pixel by pixel).

    Parameters
    ----------
    files : list
        list of fits with the frames
    output_dir : str
        output directory
    filename : str
        file name, e.g. 'ThAr'
    exptime : int
        seconds to scale the exposure time of the sciences
        if == 0 it uses the median exposure time of the files
    thresh : int
        pixels above saturation threshold will be set as nans
    """

    #Read first file
    fits_file = fits.open(files[0])
    data1 = fits_file[0].data
    err1 = fits_file[1].data
    hdr1 = fits_file[0].header

    nrows, ncols = data1.shape

    info = data1[np.newaxis, :]
    info_err = err1[np.newaxis, :]
    exp = np.zeros(len(files))
    exp[0] = hdr1['EXPTIME']

    fits_file.close()

    #Rest of the files
    for i in range(len(files)-1):
        fits_file = fits.open(files[i+1])
        info = np.append(info, fits_file[0].data[np.newaxis, :], axis=0)
        info_err = np.append(info_err, fits_file[1].data[np.newaxis, :], axis=0)
        exp[i+1] = fits_file[0].header['EXPTIME']
        fits_file.close()
    
    #Divide each image by its exposure time and scale it to exptime
    if exptime==0:
        exptime = np.median(exp)
    for i in range(len(files)):
        info[i, :, :] = info[i, :, :]*exptime/exp[i]
        info_err[i, :, :] = info_err[i, :, :]*exptime/exp[i]
    
    #Pixels above saturation threshold are set as nans
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        info[np.where(info>thresh)] = np.nan
        info_err[np.where(info>thresh)] = np.nan

    #Combine data using mean
    #Do error propagation
    data = np.zeros((nrows, ncols))
    error = np.zeros((nrows, ncols))
    hdr = hdr1.copy()
    for i in range(nrows):
        sys.stdout.write('\rCombining frames: ' 
                         + str(int(100.0*i/(1.0*nrows)))+'%')
        sys.stdout.flush()
        for j in range(ncols):
            data_slice = info[:, i, j]
            err_slice = info_err[:, i, j]
            non_missing_data = np.count_nonzero(~np.isnan(data_slice))
            if non_missing_data == 0:
                data[i, j] = np.nan
            else:
                data[i, j] = np.nanmedian(data_slice)

            non_missing_error = np.count_nonzero(~np.isnan(info_err[:, i, j]))
            if non_missing_error == 0:
                error[i, j] = np.nan
            else:
                error[i, j] = np.nansum(err_slice**2)**.5/non_missing_error**.5
    
    print('')
    
    #Create output
    hdr['EXPTIME'] = exptime
    msg = 'combination: observations combined'
    hdr['HISTORY'] = msg
    hdr['FILENAME'] = filename
    data_fits = fits.PrimaryHDU(data, header=hdr)
    error_fits = fits.ImageHDU(error)
    combine_fits = fits.HDUList([data_fits, error_fits])
    combine_fits.writeto(os.path.join(output_dir, filename + '.fits'),
                         overwrite=True)


def area_value(lim_inf, lim_sup, wave1, wave2, flux1, err1, flux2, err2):
    f = inter.interp1d([wave1, wave2], [flux1, flux2])
    if lim_inf < wave1:
        if lim_sup > wave2:
            flux = ((flux1+flux2)*(wave2-wave1))/2
            error = (np.sqrt(err1**2+err2**2)*np.abs(wave2-wave1))/2
            return flux, error
        else:
            f2 = f(lim_sup)
            e2 = np.sqrt(((err1*(wave2-lim_sup))**2)+((err2*(lim_sup-wave1))**2))
            e2 = e2/(wave2-wave1)
            flux = ((flux1+f2)*(lim_sup-wave1))/2
            error = (np.sqrt(err1**2+e2**2)*np.abs(lim_sup-wave1))/2
            return flux, error
    else:
        if lim_sup > wave2:
            f1 = f(lim_inf)
            e1 = np.sqrt(((err1*(wave2-lim_inf))**2)+((err2*(lim_inf-wave1))**2))
            e1 = e1/(wave2-wave1)
            flux = ((f1+flux2)*(wave2-lim_inf))/2
            error = (np.sqrt(e1**2+err2**2)*np.abs(wave2-lim_inf))/2
            return flux, error
        else:
            f1 = f(lim_inf)
            e1 = np.sqrt(((err1*(wave2-lim_inf))**2)+((err2*(lim_inf-wave1))**2))
            e1 = e1/(wave2-wave1)
            f2 = f(lim_sup)
            e2 = np.sqrt(((err1*(wave2-lim_sup))**2)+((err2*(lim_sup-wave1))**2))
            e2 = e2/(wave2-wave1)
            flux = ((f1+f2)*(lim_sup-lim_inf))/2
            error = (np.sqrt(e1**2+e2**2)*np.abs(lim_sup-lim_inf))/2
            return flux, error


def standard_wavelength(wave_inter, wave, flux, error):
    flux_inter = np.zeros(len(wave_inter))
    err_inter = np.zeros(len(wave_inter))
    for i in range(len(wave_inter)):
        if i==0:
            w1 = wave_inter[i]
            w = wave_inter[i]
            w2 = wave_inter[i+1]
        elif i==len(wave_inter)-1:
            w1 = wave_inter[i-1]
            w = wave_inter[i]
            w2 = wave_inter[i]
        else:
            w1 = wave_inter[i-1]
            w = wave_inter[i]
            w2 = wave_inter[i+1]
        lim_sup = (w+w2)/2
        lim_inf = (w+w1)/2
        select_wave = np.where((wave<lim_sup) & (wave>lim_inf))[0]
        if len(select_wave)==0:
            if len(np.where(wave<lim_sup)[0])==0:
                f = np.nan
                e = np.nan
            elif len(np.where(wave>lim_inf)[0])==0:
                f = np.nan
                e = np.nan
            else:
                before_selection = np.where(wave<lim_inf)[0][-1]
                after_selection = np.where(wave>lim_sup)[0][0]
                f, e = area_value(lim_inf, lim_sup,
                                  wave[before_selection],
                                  wave[after_selection],
                                  flux[before_selection],
                                  error[before_selection],
                                  flux[after_selection],
                                  error[after_selection])
            flux_inter[i] = f
            err_inter[i] = e
        else:
            before_selection = select_wave[0]-1
            after_selection = select_wave[-1]+1
            if before_selection == -1:
                selection = []
                for j in range(len(select_wave)):
                    selection.append(select_wave[j])
                selection.append(after_selection)
            elif after_selection == len(wave):
                selection = [before_selection]
                for j in range(len(select_wave)):
                    selection.append(select_wave[j])
            else:
                selection = [before_selection]
                for j in range(len(select_wave)):
                    selection.append(select_wave[j])
                selection.append(after_selection)
            flux_array = np.zeros(len(selection)-1)
            err_array = np.zeros(len(selection)-1)
            for j in range(len(selection)-1):
                flux_array[j], err_array[j] = area_value(lim_inf, lim_sup,
                                                      wave[selection[j]],
                                                      wave[selection[j+1]],
                                                      flux[selection[j]],
                                                      error[selection[j]],
                                                      flux[selection[j+1]],
                                                      error[selection[j+1]])
            flux_inter[i] = np.sum(flux_array)
            err_inter[i] = np.sqrt(np.sum(err_array**2))
    return flux_inter, err_inter


def grid_fit(sciences, wave_inter, output_dir):
    for i in range(len(sciences)):
        print('Sampling science into grid: ' + str(i+1) + '/' +
                         str(len(sciences)))
        fits_file = fits.open(sciences[i])
        flux_array = fits_file[0].data
        err_array = fits_file[1].data
        wave_array = fits_file[2].data
        hdr = fits_file[0].header
        science_name = os.path.basename(sciences[i]).replace('.fits', '')
        fits_file.close()

        nfib = flux_array.shape[0]
        flux_inter = np.zeros((nfib, len(wave_inter)))
        err_inter = np.zeros((nfib, len(wave_inter)))
        for fiber in range(nfib):
            sys.stdout.write('\rFiber: ' + str(fiber + 1) + '/' + str(nfib))
            sys.stdout.flush()
            flux = flux_array[fiber]
            err = err_array[fiber]
            wave = wave_array[fiber]
            if (len(flux[~np.isnan(flux)])==0):
                flux_inter[fiber, :] = [np.nan]*len(wave_inter)
                err_inter[fiber, :] = [np.nan]*len(wave_inter)
            else:
                flux_inter[fiber, :], err_inter[fiber, :] = standard_wavelength(
                                                            wave_inter, wave,
                                                            flux, err)
        print('')
        msg = 'combine: Sampled science into defined grid'
        hdr['HISTORY'] = msg
        data_fits = fits.PrimaryHDU(flux_inter, header=hdr)
        err_fits = fits.ImageHDU(err_inter)
        wave_fits = fits.ImageHDU(wave_inter)
        fits_file = fits.HDUList([data_fits, err_fits, wave_fits])
        name_fits = science_name + 'w.fits'
        fits_file.writeto(os.path.join(output_dir, name_fits), overwrite=True)


def grid_merge(sciences, obj, spectro, output_dir, method='ivarmean'):
    print('Combining sciences')
#   Read firts file
    fits_file = fits.open(sciences[0])
    data1 = fits_file[0].data
    err1 = fits_file[1].data
    wave1 = fits_file[2].data
    hdr1 = fits_file[0].header

    info = data1[np.newaxis, :]
    info_err = err1[np.newaxis, :]
    fits_file.close()

#   Rest of the files
    for i in range(len(sciences)-1):
        fits_file = fits.open(sciences[i+1])
        info = np.append(info, fits_file[0].data[np.newaxis, :], axis=0)
        info_err = np.append(info_err, fits_file[1].data[np.newaxis, :], axis=0)
        fits_file.close()


    nfibers = len(data1)

    center_fiber = int(nfibers/2)
    wave = np.copy(wave1[center_fiber])
    for fiber in range(nfibers):
        sys.stdout.write('\rCombining sciences: ' 
                         + str(int(100.0*fiber/(1.0*nfibers)))+'%')
        sys.stdout.flush()
        fiber_data = np.zeros(len(wave))
        fiber_err = np.zeros(len(wave))
        for i in range(len(wave)):
            data_slice = info[:, fiber, i]
            err_slice = info_err[:, fiber, i]
            non_missing_data = np.count_nonzero(~np.isnan(data_slice))
            if non_missing_data == 0:
                fiber_data[i] = np.nan
            else:
                if method == 'mean':
                    fiber_data[i] = np.nanmean(data_slice)
                    fiber_err[i] = np.nanmean(err_slice**2)**.5
                elif method == 'biweight':
                    fiber_data[i] == biweight.biweight_location(data_slice)
                    fiber_err[i] == biweight.biweight_location(err_slice**2)**.5
                elif method == 'ivarmean':
                    fiber_data[i] = (np.nansum(data_slice/err_slice**2) /
                                      np.nansum(1./err_slice**2))
                    fiber_err[i] = (1./np.nansum(err_slice**-2))**.5
#   Create output
        specfile = obj + '_' + spectro + '_' + str(fiber) + '_spectrum.dat'
        specname = os.path.join(output_dir, specfile)
        lun = open(specname, 'w')
        lun.write('#Fiber ' + str(fiber) + '\n')
        lun.write('#Wavelength [\AA]' + '\t' + 'F_lda [ergs/s/cm^2/\AA]' +
                  '\t' + 'Error [ergs/s/cm^2/\AA]' + '\n')
        fiber_spectrum = np.vstack((wave, fiber_data, fiber_err)).T
        np.savetxt(lun, fiber_spectrum)
        lun.write('\n')
        lun.close()
    print('')
