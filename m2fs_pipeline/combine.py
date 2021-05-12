import os
import sys
import warnings
import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot as plt
from astropy.io import fits


# IMAGE COMBINATION ROUTINE (PIXEL BY PIXEL)
def combine(files, filename, output_dir, exptime=0, thresh=30000):
    """
    This function takes a list of fits files containing the science to combine
    and it combines all in just one fits using median. This is done pix by pix.

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
            index = np.where((np.isfinite(info[:, i, j])==True) &
                              (np.isfinite(info_err[:, i, j])==True))
            data[i, j] = np.nanmedian(info[:, i, j][index])
            clear = np.count_nonzero(~np.isnan(info_err[:, i, j]))
            if clear==0:
                error[i, j] = np.nan
            else:
                error[i, j] = np.nansum(info_err[:, i, j][index]**2)**.5/clear**.5
    
    print('')
    
    #Create output
    hdr['EXPTIME'] = exptime
    msg = 'Sciences combined'
    hdr['HISTORY'] = msg
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


def fit_into_wave_grid(sciences, wave_inter, output_dir):
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
            sys.stdout.write('\rFiber: ' + str(fiber) + '/' + str(nfib))
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
        msg = 'Sampled science into defined grid'
        hdr['HISTORY'] = msg
        data_fits = fits.PrimaryHDU(flux_inter, header=hdr)
        err_fits = fits.ImageHDU(err_inter)
        wave_fits = fits.ImageHDU(wave_inter)
        fits_file = fits.HDUList([data_fits, err_fits, wave_fits])
        name_fits = science_name + 'w.fits'
        fits_file.writeto(os.path.join(output_dir, name_fits), overwrite=True)


"""
def merge(sciences, output_dir, spectro='b', obj='C'):
    #First file
    fits_file = fits.open(sciences[0])
    data1 = fits_file[0].data
    err1 = fits_file[1].data
    wave1 = fits_file[2].data
    hdr1 = fits_file[0].header
    fits_file.close()


    wave_all = np.zeros((nfib, ncols*len(sciences)))
    data_all = np.zeros((nfib, ncols*len(sciences)))
    error_all = np.zeros((nfib, ncols*len(sciences)))
    frame_all = np.zeros((nfib, ncols*len(sciences)))
    wave_inter = np.copy(wave1)
    data_inter = np.zeros((nfib, ncols))
    error_inter = np.zeros((nfib, ncols))
    frame_inter = np.zeros((nfib, ncols))
    datapoints = np.zeros((nfib, ncols))

    for fiber in range(nfib):
        if (len(wave_inter[fiber, :][~np.isnan(wave_inter[fiber, :])])==0):
            for k in range(len(sciences)):
                auxwave = fits.open(sciences[k])[2].data[fiber, :]
                if (len(auxwave[~np.isnan(auxwave)]) > 0):
                    wave_inter[fiber, :] = auxwave
                    break
    
    for i in range(len(sciences)):
        sys.stdout.write('\rInterpolating for science: ' + str(i+1) + '/' +
                         str(len(sciences)))
        sys.stdout.flush()     
        fits_file = fits.open(sciences[i])
        sci1d = fits_file[0].data
        scierr1d = fits_file[1].data
        wave1d = fits_file[2].data

        for fiber in range(nfib):
            wave_all[fiber, ncols*i:ncols*(i+1)] = wave1d[fiber, :]
            data_all[fiber, ncols*i:ncols*(i+1)] = sci1d[fiber, :]
            error_all[fiber, ncols*i:ncols*(i+1)] = scierr1d[fiber, :]
            frame_all[fiber, ncols*i:ncols*(i+1)] = i+1

        for fiber in range(nfib):
            f = inter.interp1d(wave1d[fiber, :], sci1d[fiber, :],
                               bounds_error=False)
            for k in range(len(data_inter[fiber, :])):
                if (np.isfinite(f(wave_inter[fiber, k]))==True):
                    data_inter[fiber, k] = data_inter[fiber, k] + f(wave_inter[fiber, k])
                    datapoints[fiber, k] = datapoints[fiber, k] + 1
    print('')
    nonzero = np.where(datapoints!=0)
    zero = np.where(datapoints==0)
    data_inter[nonzero] = data_inter[nonzero]/datapoints[nonzero]
    data_inter[zero] = np.nan

    for fiber in range(nfib):
        sortedind = np.argsort(wave_all[fiber, :])
        wave_all[fiber, :] = wave_all[fiber, :][sortedind]
        data_all[fiber, :] = data_all[fiber, :][sortedind]
        error_all[fiber, :] = error_all[fiber, :][sortedind]
        frame_all[fiber, :] = frame_all[fiber, :][sortedind]
    
    msg = 'Merged science 1D'
    hdr1['HISTORY'] = msg
    data_fits = fits.PrimaryHDU(data_all, header=hdr1)
    error_fits = fits.ImageHDU(error_all)
    wave_fits = fits.ImageHDU(wave_all)
    frame_fits = fits.ImageHDU(frame_all)
    combine_fits = fits.HDUList([data_fits, error_fits, wave_fits, frame_fits])
    name_fits = 'GOODS_' + obj + '_' + spectro + '_merged.fits'
    combine_fits.writeto(os.path.join(output_dir, name_fits), overwrite=True)

    for fiber in range(nfib):
        sys.stdout.write('\rCreating text for fiber: ' + str(fiber))
        sys.stdout.flush()

        specfile = 'GOODS_' + obj + '_' + spectro + '_' + str(fiber) + '_spectrum.dat'
        specname = os.path.join(output_dir, specfile)
        lun = open(specname, 'w')
        lun.write('#Every frame spectrum' + '\n')
        lun.write('#Wavelength [\AA]' + '\t' + 'F_lda [ergs/s/cm^2/\AA]' +
                  '\t' + 'Error [ergs/s/cm^2/\AA]' + '\t' + 'Frame' + '\n')
        for k in range(len(wave_all[fiber, :])):
            lun.write(str(wave_all[fiber, k]) + '\t' +
                      str(data_all[fiber, k]) + '\t' +
                      str(error_all[fiber, k]) + '\t' +
                      str(frame_all[fiber, k]) + '\n')
        lun.write('\n')
        lun.close()

        interfile = 'GOODS_' + obj + '_' + spectro + '_' + str(fiber) + '_interpolated.dat'
        intername = os.path.join(output_dir, interfile)
        lun = open(intername, 'w')
        lun.write('# Interpolated spectrum' + '\n')
        lun.write('#Wavelength [\AA]' + '\t' + 'F_lda [ergs/s/cm^2/\AA]' +
                  '\t' + 'Error [ergs/s/cm^2/\AA]' + '\t' + 'Frame' + '\n')
        for k in range(len(wave_inter[fiber, :])):
            lun.write(str(wave_inter[fiber, k]) + '\t' +
                      str(data_inter[fiber, k]) + '\t' +
                      str(error_inter[fiber, k]) + '\t' +
                      str(frame_inter[fiber, k]) + '\n')
        lun.close()
    print('')

"""