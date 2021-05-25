import os
import sys
import numpy as np
import scipy.signal as sig
import scipy.interpolate as inter
from scipy.optimize import curve_fit
from astropy.stats import biweight
from astropy.io import fits
from m2fs_pipeline import extract
from m2fs_pipeline import tracer
from m2fs_pipeline import wavecalib


def normalize(arr, arr_err, kernel_size=35):
    nfibers, ncols = arr.shape
    twilight_1d_norm = np.zeros((nfibers,ncols))
    twilight_err_1d_norm = np.zeros((nfibers, ncols))
    for fiber in range(nfibers):
        twi1d = np.copy(arr[fiber, :])
        twi1derr = np.copy(arr_err[fiber, :])
        twi1dsmooth = sig.medfilt(twi1d, kernel_size=kernel_size)
        twi1d_norm = twi1d / twi1dsmooth
        twi1derr_norm = twi1derr / twi1dsmooth
        twilight_1d_norm[fiber, :] = twi1d_norm
        twilight_err_1d_norm[fiber, :] = twi1derr_norm
    
    return (twilight_1d_norm, twilight_err_1d_norm)


def smooth(twilight1d, kernel_size=35):
    nfibers, ncols = twilight1d.shape
    twi1d_smooth = np.zeros((nfibers, ncols))
    for fiber in range(nfibers):
        twi1d_smooth[fiber, :] = sig.medfilt(twilight1d[fiber, :],
                                             kernel_size=kernel_size)
    return twi1d_smooth


def smooth_normalize(twilight1d, wave_arr, kernel_size=35):
    """
    Obtain smooth normalized twilight

    """
    nfibers, ncols = twilight1d.shape
    wmin = np.nanmax(wave_arr[:, 0])
    wmax = np.nanmin(wave_arr[:, ncols-1])
    twilight_smooth_norm = np.zeros((nfibers, ncols))
    for fiber in range(nfibers):
        twi1d = np.copy(twilight1d[fiber,:])
        if (all(np.isnan(twi1d))):
            twilight_smooth_norm[fiber, :] = twi1d*np.nan
        else:
            twi1dsmooth = sig.medfilt(twi1d, kernel_size=kernel_size)
            aux = np.where((wave_arr[fiber, :] >= wmin) &
                           (wave_arr[fiber, :] <= wmax))[0]
            twilight_smooth_norm[fiber, :] = (twi1dsmooth / 
                                              np.nanmedian(twi1dsmooth[aux]))
        
    return twilight_smooth_norm


def collapse_fibers(arr, wave):
    """
    Collapse all fibers into one
    """
    auxwave = wave.flatten()[np.argsort(wave, axis=None)]
    auxflux = arr.flatten()[np.argsort(wave, axis=None)]
    auxflux = auxflux[~np.isnan(auxwave)]
    auxwave = auxwave[~np.isnan(auxwave)]

    return (auxwave, auxflux)


def continuum_fit(arr, wave, norm=True, k=4, s=80):
    """ Spline fit
    """
    auxwave, auxflux = collapse_fibers(arr, wave)
    auxflux = sig.medfilt(auxflux, kernel_size=3)
    if norm==True:
        sset = inter.UnivariateSpline(auxwave[np.where((auxflux!=0) & (auxflux!=1))[0]],
                                      auxflux[np.where((auxflux!=0) & (auxflux!=1))[0]],
                                      k=k, s=s)
    else:
        sset = inter.UnivariateSpline(auxwave[np.where(auxflux!=0)[0]],
                                      auxflux[np.where(auxflux!=0)[0]],
                                      k=k, s=s)
    
    return sset

def remove_solar_spec(twi, twi1d, wave_vals, tracing_twi, sset, sset2,
                      yaper_each_side=3):
    """ Remove the solar specturm in the twilight (twilight / sun model)
    """
    nrows, ncols = twi.shape
    nfibers = len(tracing_twi)
    twi1d_smooth = smooth(twi1d)
    cols = np.arange(ncols)
    twicorr = np.zeros((nrows, ncols))
    for fiber in range(nfibers):
        sys.stdout.write('\rRemoving solar spectrum from fiber: ' + str(fiber))
        sys.stdout.flush()
        ypeak = tracer.get_tracing_row(tracing_twi, fiber, cols)
        if (all(np.isnan(ypeak))):
            continue
        auxsun = sset(wave_vals[fiber, :]) * sset2(wave_vals[fiber, :]) * np.nanmedian(twi1d_smooth.flatten())
        for col in range(ncols):
            ymin = int(np.round(ypeak[col]) - yaper_each_side)
            ymax = int(np.round(ypeak[col]) + yaper_each_side)
            twicorr[ymin:ymax+1, col] = twi[ymin:ymax+1, col]/auxsun[col]
    print('')
    #normalize twicorr
    auxtwicorr = twicorr.flatten()
    auxtwicorr = auxtwicorr[np.where(auxtwicorr!=0)[0]]
    twicorr = twicorr/np.nanmedian(auxtwicorr)

    return twicorr


def twilight_pre_steps(twilight_fits, tracing_twi, wave_file, kernel_size=35):
    """ Make all the necessary steps in the twilight
    - Create sun model
    - Remove solar spectrum

    Parameters
    ----------
    kernel_size : Int
        solar spectrum median smoothing filter width
        (always odd to avoid artifacts)
    """
    twi = twilight_fits[0].data
    twierr = twilight_fits[1].data
    
    nrows, ncols = twi.shape
    #array with wavelength values for each fiber and pixel column
    wave_vals = wavecalib.wave_values(wave_file, ncols)
    
    #fibers collapsing
    twi1d, twierr1d = extract.fibers_extraction(twi, twierr, tracing_twi,
                                                method='mean')
    
    #obtain normalized twilight
    twi1d_norm, twi1derr_norm = normalize(twi1d, twierr1d,
                                          kernel_size=kernel_size)
    
    #collapse all fibers into one for solar spectrum derivation
    #compute best-fit continuum normalized solar spectrum
    sset = continuum_fit(twi1d_norm, wave_vals, s=80)
    print('Solar spectrum derivation complete')
    #there should be an acceptance test here with plot and input
    
    
    #obtain smooth normalized twilight
    twi1d_smooth_norm = smooth_normalize(twi1d, wave_vals)
    
    #collapse all fibers into one to average smoothed twilight spectrum
    #scale continuum normalized solar spectrum model to average smoothed
    #twilight spectrum
    sset2 = continuum_fit(twi1d_smooth_norm, wave_vals, norm=False, k=5, s=200)
    print('Average smoothed twilight spectrum complete')
    #there should be an acceptance test here with plot and input
    
    
    #remove solar spectrum for every fiber
    twicorr = remove_solar_spec(twi, twi1d, wave_vals, tracing_twi, sset, 
                                sset2)
    
    return twicorr


def smooth_component(twicorr, tracing_twi, yaper_each_side=3, kernel_size=35):
    """Obtain smooth flat for every fiber
    """
    nrows, ncols = twicorr.shape
    cols = np.arange(ncols)
    nfibers = len(tracing_twi)
    twismooth=np.zeros((nrows, ncols))
    for fiber in range(nfibers):
        sys.stdout.write('\rObtaining Smooth Flat for fiber: ' + str(fiber))
        sys.stdout.flush()
        ypeak = tracer.get_tracing_row(tracing_twi, fiber, cols)
        if (all(np.isnan(ypeak))):
            continue
        for k in range(int((yaper_each_side*2)+1)):
            auxarr = np.zeros(ncols)
            yaux = np.nan
            for j in range(ncols):
                ymin = int(np.round(ypeak[j] - yaper_each_side))
                if (ymin!=yaux):
                    auxarr = twicorr[ymin+k, :]
                    auxarrsmooth = sig.medfilt(auxarr,
                                               kernel_size = kernel_size)
                    yaux = ymin
                twismooth[ymin+k, j] = auxarrsmooth[j]
    print('')
    return twismooth


def move(array, index):
    return np.append(array[-index:], array[:-index])


def gaussian(x, a0, a1, a2, a3):
    z = (x-a1)/a2
    return abs(a0)*np.exp((-z**2)/2)+a3


def block_offset(data, twicorr, blocks, x_halfwidth=10, maxoffset=7):
    """
    blocks: number of fiber blocks for offset measurement
    x_halfwidth: width of x window to average and measure flat/science and offset
    max_offset: maximum offset in pixels between twilight and science
    
    This function obtain each blocks offset between twilight and science
    Offset = trace(twilight) - trace(science)

    Parameters
    ----------
    x_halfwidth : Int
        Width of columns window to average and measure flat/science and offset
    """
    nrows, ncols = data.shape
    offset = np.zeros(blocks)
    ny0 = nrows / blocks
    ny0 = int(ny0)
    for j in range(blocks):
        ycuts = np.zeros(ny0)
        ycutd = np.zeros(ny0)
        midx = np.round(ncols/2.)
        start_step = int(np.floor(midx-x_halfwidth))
        end_step = int(np.ceil(midx+x_halfwidth)+1)
        for i in range(ny0):
            ycuts[i] = np.nanmedian(data[j*ny0+i, start_step:end_step])
            ycutd[i] = np.nanmedian(twicorr[j*ny0+i, start_step:end_step])
        
        auxoff = np.arange(0, maxoffset, 1) - int(maxoffset/2.)
        ccorrarr = np.zeros(len(auxoff))
        auxyarr = np.copy(ycuts)
        auxyarr0 = np.copy(ycutd)
        auxyarr[np.where(np.isfinite(ycuts)==False)[0]] = 0
        auxyarr0[np.where(np.isfinite(ycutd)==False)[0]] = 0
        
        for l in range(len(auxoff)):
            ccorrarr[l] = np.correlate(move(auxyarr, auxoff[l]), auxyarr0)[0]
        
        ccmax = (auxoff[np.where(ccorrarr == np.amax(ccorrarr))[0]])[0]
        ccsel1 = np.where(auxoff - ccmax >= -maxoffset)[0]
        ccsel2 = np.where(auxoff - ccmax <= maxoffset)[0]
        ccsel = np.intersect1d(ccsel1, ccsel2)

        ccoef, pcov = curve_fit(gaussian, auxoff[ccsel],
                                ccorrarr[ccsel]/np.nanmedian(ccorrarr[ccsel]),
                                maxfev=20000)
        offset[j] = ccoef[1]
    
    return offset


def sint(x, s, u):
    """Sinc interpolation
    """
    T = s[1] - s[0]
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    y = np.dot(x, np.sinc(sincM/T))
    return y


def shift_smooth(twismooth, tracing_data, offset, yaper_each_side=3):
    """
    Shift smooth component using sinc interpolation given the obtained offsets
    """
    nrows, ncols = twismooth.shape
    nfibers = len(tracing_data)
    cols = np.arange(ncols)
    blocks = len(offset)
    twismoothoff=np.zeros((nrows, ncols))
    for fiber in range(nfibers):
        ypeak = tracer.get_tracing_row(tracing_data, fiber, cols)
        if (all(np.isnan(ypeak))):
            continue
        k = int(ypeak[0]*blocks/nrows)
        for j in range(ncols):
            ymin = int(np.round(ypeak[j]) - yaper_each_side)
            ymax = int(np.round(ypeak[j]) + yaper_each_side)
            auxarr = twismooth[ymin:ymax+1, j]
            L = len(auxarr)
            twismoothoff[ymin+1:ymax+1-1, j] = sint(auxarr,
                                                    np.arange(0, L, 1),
                                                    np.arange(1., L-1, 1) + offset[k])
    
    return twismoothoff


def apply_twilight(data, err, flatfield):
    nrows, ncols = flatfield.shape
    output = data.copy()
    output_err = err.copy()
    for row in range(nrows):
        for col in range(ncols):
            if(flatfield[row,col] == 0):
                output[row, col] = 0
                output_err[row, col] = 0
            else:
                output[row, col] = output[row, col]/flatfield[row, col]
                output_err[row, col] = output_err[row, col]/flatfield[row, col]
    return output, output_err


def flat(sciences_fname, twilight_fname, output_dir, tracing_science_file,
         tracing_twilight_file, wavecalib_file, blocks=8, x_halfwidth=10,
         maxoffset=7, yaper_each_side=3, kernel_size=35):
    twilight_fits = fits.open(twilight_fname)
    tracing_sci = np.genfromtxt(tracing_science_file)
    tracing_twi = np.genfromtxt(tracing_twilight_file)
    wavecalib_science = np.genfromtxt(wavecalib_file)
    
    twilight_corr = twilight_pre_steps(twilight_fits, tracing_twi,
                                       wavecalib_science)

    twilight_smooth = smooth_component(twilight_corr, tracing_twi,
                                       yaper_each_side=yaper_each_side,
                                       kernel_size=kernel_size)
    for i in range(len(sciences_fname)):
        print('\rTwilight flat science: ' + str(i+1) + '/' +
              str(len(sciences_fname)))
        aux = fits.open(sciences_fname[i])
        data = aux[0].data
        err = aux[1].data
        hdr = aux[0].header
        
        print('Calculating offset between science and flat field')
        offset_sci_twi = block_offset(data, twilight_corr, blocks,
                                      x_halfwidth=x_halfwidth, maxoffset=maxoffset)

        print('Shifting flat field')
        twilight_shifted = shift_smooth(twilight_smooth, tracing_sci,
                                        offset_sci_twi,
                                        yaper_each_side=yaper_each_side)

        print('Applying twilight')
        output, output_err = apply_twilight(data, err, twilight_shifted)
        filename = os.path.basename(sciences_fname[i]).replace('.fits', '')

        hdr['HISTORY'] = 'flat: Applied Ilumination Corrected Flat Field'
        hdr['HISTORY'] = 'flat: Twilight Flat'
        flat_data = fits.PrimaryHDU(output, header=hdr)
        flat_err = fits.ImageHDU(output_err)
        flat_corr = fits.ImageHDU(twilight_corr)
        flat_smooth = fits.ImageHDU(twilight_smooth)
        flat_field = fits.ImageHDU(twilight_shifted)
        fits_file = fits.HDUList([flat_data, flat_err, flat_corr, flat_smooth,
                                  flat_field])
        fits_file.writeto(os.path.join(output_dir, filename + 'f.fits'),
                          overwrite=True)
    print('')
