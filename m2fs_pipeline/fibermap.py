import os
import numpy as np
# from m2fs_pipeline import template


def closest_detection(array1, array2):
    """
    Obtain difference between every value in array1 and closest value in array2.
    """
    aux = np.zeros(len(array1))

    for i in range(len(aux)):
        value = abs(array1[i] - array2)
        aux[i] = np.amin(value)
    
    return aux


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

    #Simulate complete fibers
    sim_peaks = []
    sim_peaks.append(trace_central[0])
    for i in range(total_fibers-1):
        if len(sim_peaks)%16!=0:
            sim_peaks.append(sim_peaks[-1] + close_distance)
        else:
            sim_peaks.append(sim_peaks[-1] + long_distance)

    closest = closest_detection(sim_peaks, trace_central)

    #Select fibers with no nearby detections
    aux = []
    aux = np.where(closest >= 6)[0]

    #Create new tracefiles with nans where fibers where not detected.
    new_trace = np.zeros((total_fibers, trace_coeffs.shape[1]))
    jump = 0
    for i in range(len(new_trace)):
        if (any((i-aux) == 0)):
            new_trace[i, :] = np.nan
            if (np.amin(abs(trace_central - sim_peaks[i])) >= 6):
                jump = jump + 1
        else:
            new_trace[i, :] = trace_coeffs[i - jump, :]

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
