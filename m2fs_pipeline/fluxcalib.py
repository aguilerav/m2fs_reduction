import os
import sys
import numpy as np
import scipy.interpolate as inter
import scipy.signal as sig
import matplotlib.pyplot as plt
from astropy.io import fits
from m2fs_pipeline import fibermap
from m2fs_pipeline import extinction


def gaussian(x, a0, a1, a2):
    z = (x-a1)/a2
    g = a0*np.exp((-z**2)/2)
    return g


def convolve(wave, flux, sigma):
    conv = gaussian(wave, 1, np.median(wave), sigma)
    conv /= np.sum(gaussian(wave, 1, np.median(wave), sigma))

    fluxconvaux = sig.convolve(flux[~np.isnan(flux)], conv, 'same')

    fluxconv = flux
    fluxconv[~np.isnan(flux)] = fluxconvaux

    return wave, fluxconv


def clean_skylines(wave, specinst, lmask=np.array([5577, 5890, 6298, 6360]),
                   dlmask=20, bound=200):

    for i in range(len(lmask)):
        specinst[np.where(abs(wave-lmask[i]) <= dlmask/2.)[0]] = np.nan
        specinst[np.where(abs(wave-wave[0]) <= bound)[0]] = np.nan
        specinst[np.where(abs(wave-wave[-1]) <= bound)[0]] = np.nan
    
    return wave, specinst


def fit(std1d, wave1d, stdfib, spec_name, sens_fname, sigma1=2.,
        sigma0=10., lmask= np.array([5577, 5890, 6298, 6360]), dlmask=20,
        bound=200):

    wave = wave1d[stdfib, :]
    specinst = std1d[stdfib, :]

    wave, specinst = clean_skylines(wave, specinst, lmask=lmask, dlmask=dlmask,
                                    bound=bound)
        
    # read standard intrinsic spectrum
    specfile = np.genfromtxt(spec_name)
    wave0 = specfile[:, 0]
    flam0 = specfile[:, 1]

    sigma = abs(sigma1**2-sigma0**2)**.5
    if (sigma1>sigma0):
        wave0, flam0 = convolve(wave0, flam0, sigma)
    else:
        wave, specinst = convolve(wave, specinst, sigma)
    
    g = inter.interp1d(wave0, flam0)
    flamint = g(wave)

    # take ratio of the two
    factor0 = flamint/specinst

    wave1 = np.genfromtxt(sens_fname, comments='#')[:, 0]
    factor1 = np.genfromtxt(sens_fname, comments='#')[:, 1]
    s = inter.interp1d(wave1, factor1, bounds_error=False)
    sensy = s(wave)

    auxfactor = factor0/sensy
    norm = np.nanmedian(auxfactor)
    coef = np.polyfit(wave[~np.isnan(auxfactor)], auxfactor[~np.isnan(auxfactor)]/norm, deg=0)
    corr = np.polyval(coef, 0.)*norm
    factor = corr*sensy

    return wave, factor, corr


def fluxcalib(std1d, stderr1d, wave1d, hdr, filename, calibfile):
    nfibers, ncols = std1d.shape

    wave0 = np.genfromtxt(calibfile, comments='#')[:, 0]
    factor0 = np.genfromtxt(calibfile, comments='#')[: ,1]
    factor0_err = np.genfromtxt(calibfile, comments='#')[:, 2]

    datacalib = np.zeros((nfibers, ncols))
    errorcalib = np.zeros((nfibers, ncols))

    f = inter.interp1d(wave0, factor0, bounds_error=False)
    f_err = inter.interp1d(wave0, factor0_err, bounds_error=False)
    for i in range(nfibers):
        sys.stdout.write('\rFlux Calibrating Fiber: ' + str(i))
        sys.stdout.flush()
        if all(np.isnan(wave1d[i, :])):
            datacalib[i, :] = std1d[i, :]
            errorcalib[i, :] = stderr1d[i, :]
        else:
            auxwave = wave1d[i, :]
            factor = f(auxwave)
            factor_err = f_err(auxwave)
            datacalib[i, :] = std1d[i, :]*factor
            for j in range(len(datacalib)):
                if np.isnan(datacalib[i][j]):
                    errorcalib[i][j] = np.nan
                else:
                    errorcalib[i][j] = (np.abs(datacalib[i][j])*
                                        np.sqrt((stderr1d[i][j]/std1d[i][j])**2 +
                                                (factor_err[j]/factor[j])**2))
    print('')

    hdr['HISTORY'] = 'fluxcalib: Flux calibrated'
    aux = fits.PrimaryHDU(datacalib, header=hdr)
    aux2 = fits.ImageHDU(errorcalib)
    aux3 = fits.ImageHDU(wave1d)
    fits_file = fits.HDUList([aux, aux2, aux3])
    fits_file.writeto(filename+'a.fits', overwrite=True)


def flux_calibration(science, fibermap_fname,
                     sens_fname, output_dir, sigma1=2.,
                     sigma0=10., starfibers=np.array([]), spectro='b',
                     field='COSMOS', interactive=False):
    aux = fits.open(science)
    std1d = aux[0].data
    stderr1d = aux[1].data
    wave1d = aux[2].data
    hdr = aux[0].header
    aux.close()

    correction = 0
    counter = 0

    if (len(starfibers)==0):
        aux, starfibers = fibermap.fibers_id('C', spectro, fibermap_fname)
    
    science_name = os.path.basename(science)[0:5]
    standard_path = os.path.join(output_dir, science_name + '_standard_templates')

    fig1, axarr = plt.subplots(2,1)
    for i in range(len(starfibers)):
        fiber = starfibers[i]
        stype = np.genfromtxt(os.path.join(standard_path,
                                           str(fiber) + '_chi_calibration.out'),
                              dtype=str)[:, 0][0]
        chi = np.genfromtxt(os.path.join(standard_path, 
                                         str(fiber) + '_chi_calibration.out'),
                            dtype=str)[:, 1][1]
        chi = float(chi)
        spec_name = os.path.join(standard_path, str(fiber) + '_' + stype +
                                                '_standard.out')
        
        wave, factor, auxcorr = fit(std1d, wave1d, fiber,
                                    spec_name, sens_fname, sigma1=sigma1,
                                    sigma0=sigma0)
        axarr[0].plot(wave, wave*0+auxcorr, '--', linewidth=2,
                      label=science_name + '_' + str(fiber))
        axarr[1].plot(wave, factor, '--', linewidth=2,
                      label=science_name + '_' + str(fiber))
        
        correction = correction + auxcorr/chi
        counter = counter + (1/chi)
    correction = correction/counter

    wave1 = np.genfromtxt(sens_fname, comments='#')[:, 0]
    factor1 = np.genfromtxt(sens_fname, comments='#')[:, 1]
    factor1_err = np.genfromtxt(sens_fname, comments='#')[:, 2]

    if (interactive==True):

        axarr[0].plot(wave1, wave1*0+correction, color='black',
                      label='Weighted correction', linewidth=3)
        axarr[0].set_ylabel('Correction', fontsize=25)
        axarr[1].plot(wave1, factor1*correction, color='black',
                      label='Corrected sensitivity', linewidth=3)
        axarr[0].set_title('Sensitivity', fontsize=30)
        axarr[1].set_xlabel(r'Wavelength [$\AA$]', fontsize=25)
        axarr[1].set_ylabel(r'$F_{\lambda}$ [(erg/s/cm2/A)/(e/s/pix)]',
                            fontsize=25)
        axarr[0].legend()
        axarr[1].legend()
        axarr[1].tick_params(axis='x', labelsize=20)
        axarr[1].tick_params(axis='y', labelsize=20)
        axarr[0].tick_params(axis='y', labelsize=20)
        plt.show()

        var = input('ENTER 0 TO ACCEPT SOLUTION, -1 TO REMOVE STAR: ')
    else:
        var = 0
    
    if (float(var) == float(-1)):
        var = input('ENTER STARS TO REMOVE SEPARATED BY COMAS (eg 7,13,123): ')
        fib = np.array(var.split(',')).astype(int)
        for k in range(len(fib)):
            index = np.where(starfibers == fib[k])[0]
            starfibers = np.delete(starfibers, index)
        flux_calibration(science, fibermap_fname,
                         sens_fname, output_dir, sigma1=sigma1, sigma0=sigma0,
                         starfibers=starfibers, spectro=spectro, field=field)

    else:
        wave = np.copy(wave1)
        sensy = np.copy(factor1)
        sensy_err = np.copy(factor1_err)
        sensy = sensy*correction
        sensy_err = sensy_err*correction

        fig2 = plt.figure()
        plt.plot(wave, sensy)
        plt.show()

        std_file = science[0:science.find('.fits')] + '_correction.out'
        lun = open(std_file, 'w')
        lun.write('#Flux calibration curve' + '\n')
        lun.write('#Correction: ' + str(correction) + '\n')
        lun.write('#Angstroms  (erg/s/cm2/A)/(e/s/pix)' + '\n')
        std_data = np.vstack((wave, sensy, sensy_err)).T
        np.savetxt(lun, std_data)
        lun.close()

        #GALACTIC EXTINCTION AND FLUX CALIBRATION
        stdfile_corrected = science[0:science.find('.fits')] + '_correction_extinction.out'

        extinction.extinction(field, std_file, stdfile_corrected)
        filename = os.path.basename(science)
        filename = filename[0:filename.find('.fits')]
        filename = os.path.join(output_dir, filename)
        fluxcalib(std1d, stderr1d, wave1d, hdr, filename, stdfile_corrected)



