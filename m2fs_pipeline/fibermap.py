import os
import numpy as np


def nan_filler(array, close_distance, long_distance, total_length):
    """
    It fills an array when it has missing values according a possible close
    and long distances for the values.

    Parameters
    ----------
    array : list
        The array to edit
    close_distance : float
        The first distance, the short one
    long_distance : float
        The long possible distance
    total_length : int
        The length that the array must have
    
    Returns
    -------
    numpy.ndarray
        The array list but now with nans where there is a missing value
    """
    nan_array = []
    for i in range(len(array)-1):
        if (array[i+1] - array[i] < 1.5*close_distance):
            nan_array.append(array[i])
        else:
            if (array[i+1] - array[i] < 2.5*close_distance):
                nan_array.append(array[i])
                nan_array.append(np.nan)
            else:
                if (array[i+1] - array[i] < 3.5*close_distance):
                    nan_array.append(array[i])
                    nan_array.append(np.nan)
                    nan_array.append(np.nan)
                else:
                    if (array[i+1] - array[i] < long_distance + 0.5*close_distance):
                        nan_array.append(array[i])
                    else:
                        nan_array.append(array[i])
                        nan_array.append(np.nan)
    nan_array.append(array[-1])
    if len(nan_array) <= total_length:
        for i in range(total_length - len(nan_array)):
            nan_array.append(np.nan)
    
    return np.array(nan_array)


def fill_fibers(tracename, total_fibers=128):
    """
    Fill not found fibers with Nans to keep track of them.
    This will edit the trace file. It fills with Nans the non-detected fibers
    The output is a tracefile with 128 rows (or total_fibers rows).
    IMPORTANT - This does not work if the first fiber is dead! - IMPORTANT
    
    Parameters
    ----------
    tracename : str
        Name of the trace file, e.g. <path>/b0145_trace_coeffs.out
    total_fibers : int
        Number of total fibers in the detector
    
    Returns
    -------
    None
    """
    #Read trace file
    trace_coeffs = np.genfromtxt(tracename)
    trace_central = trace_coeffs[:, -1]

    #Distances between every traced fiber
    dfibers = np.diff(trace_central)
    close_distance = np.median(dfibers)
    long_distance = np.median(dfibers[dfibers>8*close_distance])

    #Create new tracefiles with nans where fibers where not detected.
    full_peaks = nan_filler(trace_central, close_distance, long_distance,
                            total_fibers)
    new_trace = np.zeros((total_fibers, trace_coeffs.shape[1]))
    for i in range(len(new_trace)):
        if np.isnan(full_peaks[i]):
            new_trace[i, :] = np.nan
        else:
            idx = (np.abs(trace_central - full_peaks[i])).argmin()
            new_trace[i, :] = trace_coeffs[idx, :]

    #Verify new file have same length as template (128)
    if (len(new_trace) != total_fibers):
        print('ERROR IN FIBERS NUMBERS: must stop program')

    print('Total detected fibers: {}'.format(
                                len(new_trace[~np.isnan(new_trace[:, 0])])))

    #Write new files
    new_trace_name = os.path.basename(tracename).replace('.out', '_full.out')
    new_tracefile = os.path.join(os.path.dirname(tracename), new_trace_name)
    with open(new_tracefile, 'w') as new_tracing_file:
        np.savetxt(new_tracefile, new_trace)


def fiber_number(spectro, block, fiber):
    if (spectro == 'b'):
        fibernumber = (block - 1)*16 + (16 - fiber)
        return fibernumber
    elif (spectro == 'r'):
        fibernumber = (8 - block)*16 + (16 - fiber)
        return fibernumber
    else:
        print('Invalid spectrograph')


def fibermap_info(fiber, spectro, fibermap_fname):
    """
    It returns the corresponding fiber identification and name
    e.g. ('B1-12', 'COS-539573')
    
    Parameters
    ----------
    fiber : Int
        Fiber number in (0-127) basis
    spectro : str
        'b' or 'r'
    fibermap_fname : str
        fibermap file path
    
    Returns
    -------
    str
        fiber idx e.g. 'B1-12'
    str
        fiber name e.g. 'COS-539573'
    """
    fibermap = np.genfromtxt(fibermap_fname, dtype=str, comments='#')
    idx = fibermap[:, 0]
    names = fibermap[:, 1]
    if (spectro == 'b'):
        fiberblock = int(np.floor(fiber/16)) + 1
        fibernumber = 16-(fiber - 16*(fiberblock-1))
        if fibernumber >= 10:
            fibernumber = str(fibernumber)
        else:
            fibernumber = '0' + str(fibernumber)
        fiber_idx = 'B' + str(fiberblock) + '-' + fibernumber
        fiber_name = names[fiber_idx == idx][0]
        return  fiber_idx, fiber_name
    elif (spectro == 'r'):
        fiberblock = 8 - int(np.floor(fiber/16))
        fibernumber = 16 - (fiber - 16*(8 - fiberblock))
        if fibernumber >= 10:
            fibernumber = str(fibernumber)
        else:
            fibernumber = '0' + str(fibernumber)
        fiber_idx = 'R' + str(fiberblock) + '-' + fibernumber
        fiber_name = names[fiber_idx == idx][0]
        return fiber_idx, fiber_name
    else:
        print('Incorrect spectro')


def fibers_id(char, spectro, fibermap_fname):
    """
    This function return the names and the number of the selected fibers

    Parameters
    ----------
    char : str
        'S' for sky and 'C' for stars
    spectro : str
        M2FS frame ('b' or 'r')
    fibermap_fname : str

    Returns
    -------
    list:
        name of the selected fibers
    list
        number of the selected fibers in 0-127 base
    """
    fibermap = np.genfromtxt(fibermap_fname, dtype=str, comments='#')
    fibernames = []
    fibernumbers = []
    for i in range(len(fibermap)):
        if (fibermap[i][0][0:1] == spectro.upper()):
            if (char == 'S'):
                if (fibermap[i][5] == 'S'):
                    block = int(fibermap[i][0][1:2])
                    fiber = int(fibermap[i][0][3:5])
                    fibernumbers.append(fiber_number(spectro, block, fiber))
                    fibernames.append(fibermap[i][1])
            if (char == 'C'):
                if (fibermap[i][-1] != '-'):
                    block = int(fibermap[i][0][1:2])
                    fiber = int(fibermap[i][0][3:5])
                    fibernumbers.append(fiber_number(spectro, block, fiber))
                    fibernames.append(fibermap[i][1])

    fibernumbers = np.array(fibernumbers)
    fibernames = np.array(fibernames)
    sorting = np.argsort(fibernumbers)
    fibernumbers = fibernumbers[sorting]
    fibernames = fibernames[sorting]

    return fibernames, fibernumbers
