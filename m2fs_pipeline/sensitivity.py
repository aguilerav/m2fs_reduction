import os
import sys
import numpy as np
import scipy.signal as sig
import scipy.interpolate as inter
import matplotlib.pyplot as plt
from astropy.io import fits

from m2fs_pipeline import fibermap
from m2fs_pipeline import standard

"""
This script calculates the sensitivity curves for the flux calibration for
each science observation.
"""

def gaussian(x, a0, a1, a2):
    z = (x-a1)/a2
    g = a0*np.exp((-z**2)/2)
    return g


def convolve(wave, flux, sigma):
    conv = gaussian(wave, 1, np.median(wave), sigma)
    conv /= np.sum(gaussian(wave, 1, np.median(wave), sigma))

    fluxconvaux = sig.convolve(flux[~np.isnan(flux)], conv, 'same')

    fluxconv = np.copy(flux)
    fluxconv[~np.isnan(flux)] = fluxconvaux

    return wave, fluxconv


def clean_skylines(wave, specinst, lmask=np.array([5577, 5890, 6298, 6360]),
                   dlmask=30, bound=200):

    for i in range(len(lmask)):
        specinst[np.where(abs(wave-lmask[i]) <= dlmask/2.)[0]] = np.nan
        specinst[np.where(abs(wave-wave[0]) <= bound)[0]] = np.nan
        specinst[np.where(abs(wave-wave[-1]) <= bound)[0]] = np.nan
    
    return wave, specinst


def interpolate_in_wavelength(wave_objective, wave_original, flux_original):
    g = inter.interp1d(wave_original, flux_original)
    flux_objective = g(wave_objective)
    return flux_objective


def fit_sens_factor(wave, factor_raw, nsmooth=1):
    norm = np.nanmedian(factor_raw)
    auxwave = wave[~np.isnan(factor_raw)]
    auxflux = factor_raw[~np.isnan(factor_raw)]/norm
    coef = np.polyfit(auxwave, sig.medfilt(auxflux, kernel_size=10*nsmooth+1),
                      deg=3)
    factor = np.polyval(coef, wave)
    factor = factor*norm

    return factor


def save_calibration(wave, factor, chi_square, filename, spectral, output_dir):
    standard_calibration = []
    for i in range(len(factor)):
        standard_calibration.append([wave[i], factor[i]])
    standard_calibration = np.array(standard_calibration)

    filename = os.path.join(output_dir, filename + '_calibration.out')

    standard = open(filename, 'w')
    standard.write('# ' + spectral + '\n')
    standard.write('# Chi square = ' + str(chi_square) + '\n')
    standard.write('# Flux Calibration Sensitivity Curve' + '\n')
    standard.write('# Angstroms (erg/s/cm2/A)/(e/s/pix)' + '\n')
    np.savetxt(standard, standard_calibration)
    standard.close()


def calibration(flux_array, err_array, wave_array, starfibers,
                standard_path, sigma1=2., sigma0=10.,
                lmask=np.array([5577, 5890, 6298, 6360]), dlmask=30, bound=100,
                nsmooth=1):
    sigma = abs(sigma1**2 - sigma0**2)**.5
    for i in range(len(starfibers)):
        sys.stdout.write('\rCalculating sensitivity curve: ' + str(i+1) + 
                         '/' + str(len(starfibers)))
        sys.stdout.flush()
        fiber = starfibers[i]
        stypes = np.genfromtxt(os.path.join(standard_path, 
                                            str(fiber) + '_chi_standard.out'),
                               dtype=str)[:, 0]
        chi_correc = np.zeros(len(stypes))
        wave = wave_array[fiber]
        flux = flux_array[fiber]
        err = err_array[fiber]
        wave, flux = clean_skylines(wave, flux, lmask=lmask, dlmask=dlmask,
                                    bound=bound)
        if sigma0>sigma1:
            wave, flux = convolve(wave, flux, sigma)
        for j in range(len(stypes)):
            temp_path = os.path.join(standard_path, str(fiber) + '_' + stypes[j] + '_standard.out')
            wave_temp = np.genfromtxt(temp_path, comments='#')[:, 0]
            flux_temp_c = np.genfromtxt(temp_path, comments='#')[:, 1]
            if sigma1>sigma0:
                wave_temp, flux_temp_c = convolve(wave_temp, flux_temp_c, sigma)
            
            flux_int = interpolate_in_wavelength(wave, wave_temp, flux_temp_c)

            factor_raw = flux_int/flux

            factor = fit_sens_factor(wave, factor_raw, nsmooth=nsmooth)

            chi_square = ((flux*factor - flux_int)**2)/(err*factor)**2
            chi_correc[j] = np.nansum(chi_square)
            save_calibration(wave, factor, chi_square,
                             str(starfibers[i]) + '_' + stypes[j],
                             stypes[j], standard_path)
        
        sort_factor = np.argsort(chi_correc)
        spectrals = stypes[sort_factor]
        chi = chi_correc[sort_factor]
        chi_file = open(os.path.join(standard_path, str(fiber) + '_chi_calibration.out'), 'w')
        chi_file.write('# Chi square between observed star (corrected by sensitivity curve) and template (corrected by input magnitude)' + '\n')
        chi_file.write('# Spectral type' + '\t' + 'Chi square' + '\n')
        for k in range(len(chi)):
            chi_file.write(spectrals[k] + '\t' + str(chi[k]) + '\n')
        chi_file.close()
    print('')


def visual_test(wave, flux, error, wave_temp, flux_temp, temp_name, fiber,
                sigma1=2., sigma0=10., lmask=np.array([5577, 6298, 6360]),
                dlmask=30, bound=100, nsmooth=1):
    wave, flux = clean_skylines(wave, flux, lmask=lmask, dlmask=dlmask,
                                bound=bound)
    sigma = abs(sigma1**2 - sigma0**2)**.5
    if sigma1>sigma0:
        wave, flux_temp = convolve(wave, flux_temp, sigma)
    else:
        wave, flux_conv = convolve(wave, flux, sigma)

    fldaint = interpolate_in_wavelength(wave, wave_temp, flux_temp)

    factor_raw = fldaint/flux_conv

    factor = fit_sens_factor(wave, factor_raw, nsmooth=nsmooth)

    chi = np.nansum((flux_conv*factor - fldaint)**2/(error*factor)**2)
    dif_conv = (flux_conv*factor - fldaint)/(error*factor)**2
    dif = (flux*factor - fldaint)/(error*factor)**2
    f, axarr = plt.subplots(4, 1, sharex=True)
    axarr[0].set_title('Fiber: ' + str(fiber) + ', stype: ' + temp_name + ', chi = ' + str(chi))
    axarr[0].plot(wave, flux/np.nanmedian(flux), linewidth=0.5, color='blue')
    axarr[0].plot(wave, fldaint/np.nanmedian(fldaint), color='red')
    axarr[0].plot(wave, flux_conv/np.nanmedian(flux_conv), color='black')
    axarr[0].plot(wave, factor/np.nanmedian(factor), color='orange')
    axarr[0].set_ylabel('Flux/median(flux)')
    axarr[1].plot(wave, factor_raw, color='black')
    axarr[1].plot(wave, factor, color='orange')
    axarr[1].set_ylabel('Sensitivity factor')
    axarr[2].plot(wave, flux*factor/np.nanmedian(flux*factor), linewidth=0.5, color='blue')
    axarr[2].plot(wave, flux_conv*factor/np.nanmedian(flux_conv*factor), color='black')
    axarr[2].plot(wave, fldaint/np.nanmedian(fldaint), color='red')
    axarr[2].set_ylabel('Flux/median(flux)')
    axarr[3].axhline(0, color='orange')
    axarr[3].plot(wave, dif, linewidth=0.5, color='blue')
    axarr[3].plot(wave, dif_conv, color='black')
    axarr[3].set_ylabel('Flux*sens_factor - Template flux')
    axarr[3].set_xlabel('Wavelength [Angstroms]', fontsize=26)
    f.show()
    input('Close plot and press ENTER')


def factor(science, fibermap_fname, magnitudes_fname, output_dir, spectro='b',
           sigma1=2., sigma0=10., lmask=np.array([5577, 5890, 6298, 6360]),
           dlmask=20, bound=100, nsmooth=1, plot=False):
    starnames, starfibers = fibermap.fibers_id('C', spectro, fibermap_fname)
    starnames, starfibers = standard.select_stars(starnames, starfibers,
                                                  magnitudes_fname)

    science_fits = fits.open(science)
    flux_array = science_fits[0].data
    err_array = science_fits[1].data
    wave_array = science_fits[2].data
    science_fits.close()

    science_name = os.path.basename(science)
    temporal_path = os.path.join(output_dir,
                                 science_name[0:5] + '_standard_templates')
    if not os.path.isdir(temporal_path):
        os.mkdir(temporal_path)

    calibration(flux_array, err_array, wave_array, starfibers,
                temporal_path, sigma1=sigma1, sigma0=sigma0, lmask=lmask,
                dlmask=dlmask, bound=bound, nsmooth=nsmooth)
    
    if plot==True:
        for i in range(len(starfibers)):
            fiber = starfibers[i]
            stype = np.genfromtxt(os.path.join(temporal_path, 
                                               str(fiber) + '_chi_calibration.out'),
                                  dtype=str)[:, 0][0]
            wave = wave_array[fiber]
            flux = flux_array[fiber]
            err = err_array[fiber]
            temp_path = os.path.join(temporal_path, str(fiber) + '_' + stype + '_standard.out')
            wave_t = np.genfromtxt(temp_path, comments='#')[:, 0]
            flux_t_c = np.genfromtxt(temp_path, comments='#')[:, 1]

            visual_test(wave, flux, err, wave_t, flux_t_c, stype, fiber,
                        sigma1=sigma1, sigma0=sigma0, lmask=lmask,
                        dlmask=dlmask, bound=bound)


def plot_curves(sciences, starfibers, wave_factor, factor, output_dir):
    
    fig1 = plt.figure()
    for i in range(len(sciences)):
        science_name = os.path.basename(sciences[i])[0:5]
        standard_path = os.path.join(output_dir,
                                     science_name + '_standard_templates')
        for j in range(len(starfibers)):
            fiber = starfibers[j]
            stype = np.genfromtxt(os.path.join(standard_path, 
                                               str(fiber) + '_chi_calibration.out'),
                                  dtype=str)[:, 0][0]
            calib_name = os.path.join(standard_path, str(fiber) + '_' + stype +
                                                    '_calibration.out')
            wave_calib = np.genfromtxt(calib_name, comments='#')[:, 0]
            sens_calib = np.genfromtxt(calib_name, comments='#')[:, 1]
            star_sens = np.vstack((wave_calib, sens_calib/np.nanmedian(sens_calib))).T
            star_sens_fname = os.path.join(standard_path, science_name + '_' +
                                                          str(starfibers[j]) +
                                                          '_sens.dat')
            np.savetxt(star_sens_fname, star_sens)
            plt.plot(wave_calib, sens_calib/np.nanmedian(sens_calib), '--',
                     label=science_name + '_' + str(fiber))
    plt.plot(wave_factor, factor, linewidth=4, color='black',
                                               label='Sensitivity curve')
    plt.legend()
    plt.title('Sensitivity')
    plt.xlabel(r'Wavelength [$\AA$]')
    plt.ylabel(r'$F_{\lambda}$')
    plt.show()


def plot_chi(chi_squares):
    
    fig2 = plt.figure()
    plt.hist(chi_squares)
    plt.show()


def curve(sciences, fibermap_fname, magnitudes_fname, output_dir,
          obj='COSMOS_C', spectro='b', plot=False):
    starnames, starfibers = fibermap.fibers_id('C', spectro,
                                               fibermap_fname)
    starnames, starfibers = standard.select_stars(starnames, starfibers,
                                                  magnitudes_fname)
    
    wave_factor = np.arange(4500, 7000, 1)
    factor_weight = np.zeros(len(wave_factor))
    factor_median = np.zeros((len(sciences)*len(starfibers),len(wave_factor)))
    curves_w = np.zeros(len(wave_factor))
    chi_squares = np.zeros(len(sciences)*len(starfibers))

    for i in range(len(sciences)):
        science_name = os.path.basename(sciences[i])[0:5]
        standard_path = os.path.join(output_dir,
                                     science_name + '_standard_templates')
        
        for j in range(len(starfibers)):
            fiber = starfibers[j]
            stype = np.genfromtxt(os.path.join(standard_path, 
                                               str(fiber) + '_chi_calibration.out'),
                                  dtype=str)[:, 0][0]
            chi = np.genfromtxt(os.path.join(standard_path, 
                                             str(fiber) + '_chi_calibration.out'),
                                dtype=str)[:, 1][1]
            chi = float(chi)
            chi_squares[i*len(starfibers)+j] = chi
            calib_name = os.path.join(standard_path, str(fiber) + '_' + stype +
                                                    '_calibration.out')
            wave_calib = np.genfromtxt(calib_name, comments='#')[:, 0]
            sens_calib = np.genfromtxt(calib_name, comments='#')[:, 1]
            sens_calib = sens_calib/np.nanmedian(sens_calib)
            f = inter.interp1d(wave_calib, sens_calib, bounds_error=False)
            for k in range(len(wave_factor)):
                factor_median[(i*len(starfibers))+j, k] = f(wave_factor[k])
                factor_weight[k] = factor_weight[k] + (f(wave_factor[k])/chi)
                if (np.isfinite(f(wave_factor[k]))==True):
                    curves_w[k] = curves_w[k]+(1/chi)
    
    factor_w = factor_weight/curves_w
    factor_dispersion = np.zeros(len(wave_factor))
    for k in range(len(wave_factor)):
        slice = factor_median[:, k]
        if len(slice[~np.isnan(slice)]) < len(slice):
            factor_dispersion[k] = np.nan
        else:
            factor_dispersion[k] = np.nanstd(slice)

    if plot==True:
        plot_curves(sciences, starfibers, wave_factor, factor_w, output_dir)
        plot_chi(chi_squares)

    sens_data = np.vstack((wave_factor, factor_w, factor_dispersion)).T
    sens_file = open(os.path.join(output_dir, obj + '_' + spectro +
                                              '_sensitivity.out'), 'w')
    sens_file.write('# Sensitivity curve' + '\n')
    sens_file.write('# Angstroms sensitivity[(erg/s/cm2/A)/(e/s/pix)] sensitivity_error[(erg/s/cm2/A)/(e/s/pix)]' + '\n')
    np.savetxt(sens_file, sens_data)
    sens_file.close()