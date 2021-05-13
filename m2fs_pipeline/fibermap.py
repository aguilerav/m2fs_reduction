import os
import numpy as np
from m2fsredux import template


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
    FILL NOT FOUND FIBERS WITH NANS TO KEEP TRACK OF THEM
    This will edit the trace file. It fill with Nans the non-detected fibers
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


def create_converter(fibermap_fname, spectro):
    """
    spectro: b or r
    """
    fibermap = np.genfromtxt(fibermap_fname, dtype=str, skip_header=22,
                             skip_footer=15)
    names = []
    block = []
    numbers = []
    fiber = []
    if (spectro == 'b'):
        for i in range(len(fibermap)):
            if (fibermap[i][0][0:1] == 'B'):
                names.append(fibermap[i][0])
                block.append(fibermap[i][0][1:2])
                numbers.append(fibermap[i][0][3:])
        names = np.array(names)
        block = np.array(block).astype(int)
        numbers = np.array(numbers).astype(int)
        for i in range(len(numbers)):
            b = (block[i]-1)*16
            n = 16 - numbers[i]
            fiber.append(b + n)
    if (spectro == 'r'):
        for i in range(len(fibermap)):
            if (fibermap[i][0][0:1] == 'R'):
                names.append(fibermap[i][0])
                block.append(fibermap[i][0][1:2])
                numbers.append(fibermap[i][0][3:])
        names = np.array(names)
        block = np.array(block).astype(int)
        numbers = np.array(numbers).astype(int)
        for i in range(len(numbers)):
            b = (8-block[i])*16
            n = 16- numbers[i]
            fiber.append(b + n)
    fiber = np.array(fiber)
    return fiber
    

def fibers_routine(char, spectro, fibermap_fname, converter_fname,
                   magnitudes_fname):
    """
    char: S for sky, C for stars
    """
    fibermap = np.genfromtxt(fibermap_fname, dtype=str, skip_header=22,
                             skip_footer=15)
    converter = np.genfromtxt(converter_fname, dtype=str)
    target_fibers = []
    fibername = []
    if (spectro == 'b'):
        for i in range(len(fibermap)):
            if (fibermap[i][0][0:1] == 'B'):
                if (char == 'S'):
                    if (fibermap[i][5] == 'S'):
                        index = np.where(converter[:,0] == fibermap[i][0])[0][0]
                        fiber = int(converter[index, 1])
                        target_fibers.append(fiber)
                        fibername.append(fibermap[i][1])
                if (char == 'C'):
                    if (fibermap[i][-1] != '-'):
                        index = np.where(converter[:,0] == fibermap[i][0])[0][0]
                        fiber = int(converter[index, 1])
                        target_fibers.append(fiber)
                        fibername.append(fibermap[i][1])
    
    if (spectro == 'r'):
        for i in range(len(fibermap)):
            if (fibermap[i][0][0:1] == 'R'):
                if (char == 'S'):
                    if (fibermap[i][5] == 'S'):
                        index = np.where(converter[:,0] == fibermap[i][0])[0][0]
                        fiber = int(converter[index, 1])
                        target_fibers.append(fiber)
                        fibername.append(fibermap[i][1])
                if (char == 'C'):
                    if (fibermap[i][-1] != '-'):
                        index = np.where(converter[:,0] == fibermap[i][0])[0][0]
                        fiber = int(converter[index, 1])
                        target_fibers.append(fiber)
                        fibername.append(fibermap[i][1])
    target_fibers = np.array(target_fibers)
    fibername = np.array(fibername)
    sorting = np.argsort(target_fibers)
    target_fibers = target_fibers[sorting]
    fibername = fibername[sorting]

    if char == 'C':
        target = []
        name = []
        for star in range(len(fibername)):
            mag_B, err_B, mag_V, err_V = template.extract_mag(magnitudes_fname,
                                                              fibername[star])
            if (mag_B!=99) or (mag_V!=99):
                target.append(target_fibers[star])
                name.append(fibername[star])
        target_fibers = np.array(target)
        fibername = np.array(name)

    return fibername, target_fibers


def substraction(array1, array2):
    aux = np.zeros(len(array1))

    for i in range(len(aux)):
        value = abs(array1[i] - array2)
        aux[i] = np.amin(value)
    
    return aux




