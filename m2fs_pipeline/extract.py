import sys
import os
import numpy as np
from astropy.stats import biweight
from astropy.io import fits
from m2fs_pipeline import tracer
from m2fs_pipeline import wavecalib

"""
This is an auxiliary script that makes the extractions.
- Extract fibers
- Collapse fibers
"""


def extract1D(data_array, tracefile, nfiber, error_array = 'error',
              yaper=4, method='sum', output=0):
    """
    Extract one fiber of the data frame using the trace file.

    Parameters
    ----------
    data_array : numpy.ndarray
        Data frame
    tracefile : str
        Name of the tracing coefficients file with Nans in no detection,
        e.g. <path>/b0148_trace_coeffs_full.out
    nfiber : Int
        Fiber to be extracted
    error_array : numpy.ndarray or str
        Error frame. If it not needed just use a str
    yaper_each_side : Int
        Extraction aperture to each side.
    method : str
        Possibles values are
        sum : Simple sum along fiber aperture
        mean : Mean value along fiber aperture
        ivarmean : Inverse variance weighted average (error must be given)
    output : Int
        If 0 not error frame is returned
    
    Returns
    -------
    Tuple of two numpy.ndarray or one numpy.ndarray
    """
    
    ncols = data_array.shape[1]
    ypeak = tracer.get_tracing_row(tracefile, nfiber, np.arange(ncols))
    spec1d = np.zeros(ncols)
    err1d = np.zeros(ncols)
    
    #Return nans if no fiber
    if (len(ypeak[~np.isnan(ypeak)]) == 0):
        if (output == 0):
            return spec1d*np.nan
        else:
            return spec1d*np.nan, err1d*np.nan


    for col in range(ncols):
        yaper_each_side = yaper/2
        start_aper = int(np.floor(ypeak[col]) - yaper_each_side + 1)
        end_aper = int(np.ceil(ypeak[col]) + yaper_each_side)
        flux = data_array[start_aper:end_aper, col]
        
        sel = np.where((np.isfinite(flux)==True))[0]
        ngood = len(sel)
        
        if(method == 'sum'):
            if(ngood>=1):
                spec1d[col] = np.sum(flux[sel])
            else:
                spec1d[col] = np.nan
        
        elif(method == 'mean'):
            if (ngood == len(flux)):
                spec1d[col] = np.nanmean(flux[sel])
            else:
                spec1d[col] = np.nan

        elif(method == 'biweight'):
            if (ngood == len(flux)):
                spec1d[col] = biweight.biweight_location(flux[sel])
            else:
                spec1d[col] = np.nan

        elif (method == 'ivarmean'):
            if(type(error_array) == str):
                print('Must provide error frame for this method')
            
            eflux=error_array[start_aper:end_aper, col]
            if (ngood == len(flux)):
                spec1d[col] = (np.nansum(flux[sel]/eflux[sel]**2) /
                               np.nansum(1./eflux[sel]**2))
                err1d[col]= (1.0/np.nansum(eflux[sel]**-2))**.5
            else:
                spec1d[col] = np.nan
                err1d[col]=np.nan
    if (output == 0):
        return spec1d
    else:
        return spec1d, err1d


def fibers_extraction(data_array, error_array, tracefile, method='ivarmean',
                      row_aper=4):
    """
    This functions extracts in 1D all the fibers in the science

    Parameters
    ----------
    data_array : FITS
    error_array : FITS
    tracefile : numpy.ndarray
        Array with tracing coefficients for each fiber
    method : str
        Possibles values are
        sum : Simple sum along fiber aperture
        mean : Mean value along fiber aperture
        ivarmean : Inverse variance weighted average (error must be given)
    row_aper : Int
        Pixels aperture for the extraction
    
    Returns
    -------
    numpy.ndarray
        Data array with shape nfibers x ncols
    numpy.ndarray
        Error array with shape nfibers x ncols
    """
    ncols = data_array.shape[1]
    nfibers = len(tracefile)
    data1D = np.zeros((nfibers, ncols))
    error1D = np.zeros((nfibers, ncols))
    for fiber in range(nfibers):
        sys.stdout.write('\rCollapsing fiber: ' + str(fiber))
        sys.stdout.flush()
        if method == 'ivarmean':
            data1D[fiber, :], error1D[fiber, :] = extract1D(data_array,
                                                            tracefile, fiber,
                                                            error_array=error_array,
                                                            yaper=row_aper,
                                                            method=method,
                                                            output=1)
        else:
            data1D[fiber, :] = extract1D(data_array, tracefile, fiber,
                                         yaper=row_aper, method=method)
            error1D[fiber, :] = extract1D(error_array**2, tracefile, fiber,
                                          method=method)**.5
    print('')

    return (data1D, error1D)


def residuals(data_1D, err_1D):
    diagnose = data_1D/err_1D
    return diagnose


def collapse_fibers(fits_fname, tracefile, wavefile, output_dir, 
                    method='ivarmean', apsize=4):
    filename = os.path.basename(fits_fname).replace('.fits', '')
    data = fits.getdata(fits_fname)
    err = fits.getdata(fits_fname, 1)
    hdr = fits.getheader(fits_fname)
    ncols = data.shape[1]

    trace = np.genfromtxt(tracefile)

    wave_1D = wavecalib.wave_values(np.genfromtxt(wavefile), ncols)

    data_1D, err_1D = fibers_extraction(data, err, trace, method=method,
                                        row_aper=apsize)

    residual = residuals(data_1D, err_1D)
    hdr['HISTORY'] = 'Fibers collapsed, method = ' + method

    fits1 = fits.PrimaryHDU(data_1D, header=hdr)
    fits2 = fits.ImageHDU(err_1D, header=hdr)
    fits3 = fits.ImageHDU(wave_1D)
    fits4 = fits.ImageHDU(residual)

    fits_file = fits.HDUList([fits1, fits2, fits3, fits4])
    fits_file.writeto(os.path.join(output_dir, filename + 'c.fits'),
                      overwrite=True)