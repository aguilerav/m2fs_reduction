import os
import sys
import numpy as np
import scipy.signal as sig
import scipy.interpolate as inter
import scipy.integrate as integ
from astropy.io import fits
from astropy import constants as const

from m2fs_pipeline import fibermap


def load_filters(curves_list, curves_path):
    band_B = np.where('B' == curves_list[:, 1])[0]
    band_V = np.where('V' == curves_list[:, 1])[0]
    curve_B = np.genfromtxt(os.path.join(curves_path,
                                         curves_list[band_B[0]][0]))
    curve_V = np.genfromtxt(os.path.join(curves_path,
                                         curves_list[band_V[0]][0]))
    wave_B = curve_B[:, 0]
    trans_B = curve_B[:, 1]
    wave_V = curve_V[:, 0]
    trans_V = curve_V[:, 1]

    return wave_B, trans_B, wave_V, trans_V


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
                   dlmask=20, bound=200):

    for i in range(len(lmask)):
        specinst[np.where(abs(wave-lmask[i]) <= dlmask/2.)[0]] = np.nan
        specinst[np.where(abs(wave-wave[0]) <= bound)[0]] = np.nan
        specinst[np.where(abs(wave-wave[-1]) <= bound)[0]] = np.nan
    
    return wave, specinst


def normalize(wave, array1, iterations=4):

    array1 = array1/np.nanmedian(array1)

    auxwave = np.copy(wave)
    auxarray = np.copy(array1)
    fit = np.zeros(len(auxwave))
    
    for i in range(iterations):
        auxwave = auxwave[np.where(auxarray>fit)[0]]
        auxarray = auxarray[np.where(auxarray>fit)[0]]
        coef = np.polyfit(auxwave, auxarray, deg=3)
        fit = np.polyval(coef, auxwave)
    
    continuum = np.polyval(coef, wave)

    return array1-continuum


def chi_template(wave, flux, error, template_file, sigma1=2., sigma0=10.,
                 lmask=np.array([5577, 5890, 6298, 6360]), dlmask=20,
                 bound=100):
    wave, flux = clean_skylines(wave, flux, lmask=lmask, dlmask=dlmask,
                                bound=bound)

    sigma = abs(sigma1**2 - sigma0**2)**.5

    wave = wave[~np.isnan(flux)]
    error = error[~np.isnan(flux)]
    flux = flux[~np.isnan(flux)]

    if (len(wave)%2 == 0):
        wave = wave[:-1]
    
    flux = flux[np.where(wave)[0]]
    error = error[np.where(wave)[0]]

    template = np.genfromtxt(template_file)
    wave_temp = template[:, 0]
    flux_temp = template[:, 1]

    if (sigma1 <= sigma0):
        wave, flux = convolve(wave, flux, sigma)
    else:
        wave0, flux0 = convolve(wave_temp, flux_temp, sigma)
    
    f = inter.interp1d(wave_temp, flux_temp, bounds_error=False)
    flux_temp = f(wave)

    flux_norm = normalize(wave, flux)
    flux_temp_norm = normalize(wave, flux_temp)

    chi_square = (((flux_norm - flux_temp_norm))**2)/error**2

    return np.nansum(chi_square)


def sort_chi_standard(flux_array, err_array, wave_array, starfibers, templates,
                      pickles_path, standard_path, sigma1=2., sigma0=10.,
                      lmask=np.array([5577, 5890, 6298, 6360]), dlmask=20,
                      bound=100, num_selections=50):
    print('Number of stars: ' + str(len(starfibers)) +
          ', number of selections: ' + str(num_selections))
    for i in range(len(starfibers)):
        sys.stdout.write('\rSorting templates by chi square: ' + str(i+1) +
                         '/' + str(len(starfibers)))
        sys.stdout.flush()
        chi_square = np.zeros(len(templates))
        fiber = starfibers[i]
        wave = wave_array[fiber]
        flux = flux_array[fiber]
        err = err_array[fiber]

        for j in range(len(templates)):
            temp_path = os.path.join(pickles_path, templates[j])
            chi_square[j] = chi_template(wave, flux, err, temp_path,
                                         sigma1=sigma1, sigma0=sigma0,
                                         lmask=lmask, dlmask=dlmask,
                                         bound=bound)
        sort_template = np.argsort(chi_square)
        stypes = []
        chi = []

        for j in range(num_selections):
            stype = templates[sort_template[j]]
            stypes.append(stype[0:stype.find('.dat')])
            chi.append(chi_square[sort_template[j]])
        
        stype = np.array(stypes)
        chi = np.array(chi)
        chi_data = np.vstack((stype, chi)).T
        chi_file = open(os.path.join(standard_path, str(fiber) +
                                                    '_chi_standard.out'), 'w')
        chi_file.write('#Chi square between template and observed star' + '\n')
        chi_file.write('#Spectral type' + '\t' + 'Chi square' + '\n')
        np.savetxt(chi_file, chi_data, fmt='%s')
        chi_file.close()
    print('')


def extract_mag(magnitudes_fname, starname):
    mags = np.genfromtxt(magnitudes_fname, dtype=str)
    select = np.where(starname == mags[:, 0])[0]
    mag_V = float(mags[select[0]][3])
    err_V = float(mags[select[0]][4])
    mag_B = float(mags[select[0]][1])
    err_B = float(mags[select[0]][2])

    return mag_B, err_B, mag_V, err_V


def photometry(wave_filter, T_filter, wave_template, f_template):
    """
    It returns the observed AB magnitude of the template when is passed through
    a filter.
    wave_filter: Wavelength of the transmission curve
    T_filter: Transmission curve
    wave_template: Wavelength of the star template
    f_template: Flux of the star
    """
    wave_template = wave_template[~np.isnan(f_template)]
    f_template = f_template[~np.isnan(f_template)]
    func_temp = inter.interp1d(wave_template, f_template, bounds_error=False)
    flux_temp = func_temp(wave_template)

    func_filter = inter.interp1d(wave_filter, T_filter, kind='slinear',
                                 bounds_error=False)
    T_curve = func_filter(wave_template)
    T_curve[np.isnan(T_curve)] = 0

    c = const.c.to('AA/s').value
    lda_pivot = (integ.trapz(T_filter*wave_filter, x=wave_filter) /
                 integ.trapz(T_filter/wave_filter, x=wave_filter))
    f_lda = (integ.trapz(wave_template*flux_temp*T_curve, x=wave_template) / 
             integ.trapz(wave_template*T_curve, x=wave_template))
    
    mag_AB = -2.5*np.log10(f_lda) - 2.5*np.log10(lda_pivot/c) - 48.594

    return mag_AB


def get_correction(wave_template, flux_template, wave_B, trans_B, mag_B, err_B,
                   wave_V, trans_V, mag_V, err_V):
    obs_B = photometry(wave_B, trans_B, wave_template, flux_template)
    correc_B = 10**(-0.4*(mag_B-obs_B))
    obs_V = photometry(wave_V, trans_V, wave_template, flux_template)
    correc_V = 10**(-0.4*(mag_V-obs_V))

    if mag_V==99:
        correction = correc_B
    else:
        correction = correc_V

    return correction


def save_template(filename, wave, flux, chi_square, spectral, output_dir):
    standard_template = np.vstack((wave, flux)).T

    filename = os.path.join(output_dir, filename + '_standard.out')

    standard = open(filename, 'w')
    standard.write('# ' + spectral + '\n')
    standard.write('# Chi square = ' + str(chi_square) + '\n')
    standard.write('# Standard Star' + '\n')
    standard.write('# Angstroms (erg/s/cm2/A)/(e/s/pix)' + '\n')
    np.savetxt(standard, standard_template)
    standard.close()


def correct_template(starnames, starfibers, pickles_path, magnitudes_fname,
                     wave_B, trans_B, wave_V, trans_V, standard_path):
    for i in range(len(starfibers)):
        sys.stdout.write('\rCorrecting best templates by input magnitude: ' +
                         str(i+1) + '/' + str(len(starfibers)))
        sys.stdout.flush()
        fiber = starfibers[i]
        stypes = np.genfromtxt(os.path.join(standard_path, str(fiber) + 
                                                           '_chi_standard.out'),
                               dtype=str)[:, 0]
        chi_squares = np.genfromtxt(os.path.join(standard_path, str(fiber) + 
                                                           '_chi_standard.out'),
                                    dtype=str)[:, 1]
        chi_squares = chi_squares.astype(float)

        mag_B, err_B, mag_V, err_V = extract_mag(magnitudes_fname,
                                                 starnames[i])
        for j in range(len(stypes)):
            stype = stypes[j]
            temp_path = os.path.join(pickles_path, stype + '.dat')
            temp = np.genfromtxt(temp_path, comments='#')
            wave_temp = temp[:, 0]
            flux_temp = temp[:, 1]
            correction = get_correction(wave_temp, flux_temp, wave_B, trans_B,
                                        mag_B, err_B, wave_V, trans_V, mag_V,
                                        err_V)
            flux_temp_c = flux_temp*correction
            save_template(str(fiber) + '_' + stype, wave_temp, flux_temp_c,
                          chi_squares[j], stype,
                          standard_path)
    print('')


def standard_template(science, fibermap_fname, assets_dir,
                      magnitudes_fname, output_dir, spectro='b', sigma1=2.,
                      sigma0=10., lmask=np.array([5577, 5890, 6298, 6360]),
                      dlmask=20, bound=100, num_selections=50):
    starnames, starfibers = fibermap.fibers_id('C', spectro,
                                                    fibermap_fname)
#   Load list and paths (templates and filters)
    templates = np.genfromtxt(os.path.join(assets_dir, 'pickles.dat'),
                              dtype=str)
    pickles_path = os.path.join(assets_dir, 'Pickles')
    curves = np.genfromtxt(os.path.join(assets_dir, 'curves.dat'), dtype=str)
    curves_path = os.path.join(assets_dir, 'Curves')

#   Load transmission curves
    wave_B, trans_B, wave_V, trans_V = load_filters(curves, curves_path)

#   Create temporal directory
    science_name = os.path.basename(science)
    temporal_path = os.path.join(output_dir,
                                 science_name[0:5] + '_standard_templates')
    if not os.path.isdir(temporal_path):
        os.mkdir(temporal_path)

#   Load science
    science_fits = fits.open(science)
    flux_array = science_fits[0].data
    err_array = science_fits[1].data
    wave_array = science_fits[2].data
    science_fits.close()
#   Normalizes fluxes and takes chi square, select the better templates
    sort_chi_standard(flux_array, err_array, wave_array, starfibers, templates,
                      pickles_path, temporal_path, sigma1=sigma1, sigma0=sigma0,
                      lmask=lmask, dlmask=dlmask, bound=bound,
                      num_selections=num_selections)
#   Correct the better templates by the input magnitudes of the stars
    correct_template(starnames, starfibers, pickles_path, magnitudes_fname,
                     wave_B, trans_B, wave_V, trans_V, temporal_path)
