import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from m2fs_pipeline import basic
from m2fs_pipeline import combine
from m2fs_pipeline import tracer
from m2fs_pipeline import fibermap
from m2fs_pipeline import wavecalib
from m2fs_pipeline import flat

_scripts_dir = os.path.dirname(os.path.realpath(__file__))
_assets_dir = os.path.dirname(_scripts_dir)
_raw_images_dir = os.path.join(_assets_dir, 'raw_images')
_output_dir = os.path.join(_assets_dir, 'temp_products')
assets_dir = os.path.join(os.path.dirname(os.path.dirname(_assets_dir)),
                          'assets')


#INPUTS
spectro = 'b'
obj = 'COSMOS_C'
sciences = ['148', '149', '153', '154']
thar_lamps = ['147', '156', '161']
nehg_lamps = ['146', '151', '157', '158', '159', '160']
led_lamps = ['144', '145', '152', '155', '162']
dark = spectro + 'dark'
twilights = ['163', '164', '165', '166', '167', '168', '169', '170', '171',
             '172', '173']

#FUNCTIONS
do_basic = False
do_combine_led = False
do_trace = False
do_combine_lamps = False
do_wavecalib = False
do_combine_twilight = False
do_flat = True

#-----------------------REDUCTION-----------------------------
raw_sciences = ['']*len(sciences)
raw_thar_lamp = ['']*len(thar_lamps)
raw_nehg_lamp = ['']*len(nehg_lamps)
raw_twilights = ['']*len(twilights)
raw_led_lamp = ['']*len(led_lamps)

for i in range(len(sciences)):
    if len(sciences[i])<4:
        sciences[i] = '0' + sciences[i]
    sciences[i] = spectro + sciences[i]
    raw_sciences[i] = os.path.join(_raw_images_dir, sciences[i])

for i in range(len(thar_lamps)):
    if len(thar_lamps[i])<4:
        thar_lamps[i] = '0' + thar_lamps[i]
    thar_lamps[i] = spectro + thar_lamps[i]
    raw_thar_lamp[i] = os.path.join(_raw_images_dir, thar_lamps[i])

for i in range(len(nehg_lamps)):
    if len(nehg_lamps[i])<4:
        nehg_lamps[i] = '0' + nehg_lamps[i]
    nehg_lamps[i] = spectro + nehg_lamps[i]
    raw_nehg_lamp[i] = os.path.join(_raw_images_dir, nehg_lamps[i])

for i in range(len(twilights)):
    if len(twilights[i])<4:
        twilights[i] = '0' + twilights[i]
    twilights[i] = spectro + twilights[i]
    raw_twilights[i] = os.path.join(_raw_images_dir, twilights[i])

for i in range(len(led_lamps)):
    if len(led_lamps[i])<4:
        led_lamps[i] = '0' + led_lamps[i]
    led_lamps[i] = spectro +  led_lamps[i]
    raw_led_lamp[i] = os.path.join(_raw_images_dir, led_lamps[i])


dark = os.path.join(_raw_images_dir, dark)
twilight = 'twilight_' + obj + '_' + spectro
nehg_lamp = 'NeHgArXe_' + obj + '_' + spectro
thar_lamp = 'ThAr_' + obj + '_' + spectro
led_lamp = 'LED_' + obj + '_'  + spectro

if do_basic:
    basic.basic(raw_sciences, raw_twilights,
                raw_led_lamp+raw_nehg_lamp+raw_thar_lamp, dark, _output_dir)
    print('--------------------FINISHED BIAS/TRIM/GAIN----------------------')
sciences_b = ['']*len(sciences)
for i in range(len(sciences)):
    sciences_b[i] = os.path.join(_output_dir, sciences[i] + 'b.fits')

led_lamps_b = ['']*len(led_lamps)
for i in range(len(led_lamps)):
    led_lamps_b[i] = os.path.join(_output_dir, led_lamps[i] + 'b.fits')

twilights_b = ['']*len(twilights)
for i in range(len(twilights)):
    twilights_b[i] = os.path.join(_output_dir, twilights[i] + 'b.fits')

nehg_lamps_b = ['']*len(nehg_lamps)
for i in range(len(nehg_lamps)):
    nehg_lamps_b[i] = os.path.join(_output_dir, nehg_lamps[i] + 'b.fits')

thar_lamps_b = ['']*len(thar_lamps)
for i in range(len(thar_lamps)):
    thar_lamps_b[i] = os.path.join(_output_dir, thar_lamps[i] + 'b.fits')


if do_combine_led:
    combine.combine(led_lamps_b, led_lamp, _output_dir)
    print('----------------FINISHED LED LAMPS COMBINATION------------------')

if do_trace:
    tracer.trace(os.path.join(_output_dir, led_lamp + '.fits'), _output_dir,
                 step_size=40, bin_width=31, degree=4)
    tracing_coeffs = os.path.join(_output_dir, led_lamp + '_trace_coeffs.out')
    fibermap.fill_fibers(tracing_coeffs)
    print('-----------------------FINISHED TRACING-------------------------')
tracing_fname = os.path.join(_output_dir, led_lamp + '_trace_coeffs_full.out')

if do_combine_lamps:
    combine.combine(nehg_lamps_b, nehg_lamp, _output_dir)

    combine.combine(thar_lamps_b, thar_lamp, _output_dir)
    print('------------------FINISHED LAMPS COMBINATION--------------------')

if do_wavecalib:
    NeHgArXe_lines_fname = os.path.join(assets_dir, 'wavecalib',
                                        'NeHgArXe.dat')
    ThAr_lines_fname = os.path.join(assets_dir, 'wavecalib',
                                    'ThAr.dat')
    NeHgArXe_fname = os.path.join(_output_dir, nehg_lamp + '.fits')
    ThAr_fname = os.path.join(_output_dir, thar_lamp + '.fits')
    wavecalib.calibration(NeHgArXe_fname, ThAr_fname, _output_dir,
                          NeHgArXe_lines_fname, ThAr_lines_fname,
                          tracing_fname, tracing_fname, nthresh=3)
    print('----------------FINISHED WAVELENGTH CALIBRATION------------------')
wave_fname = os.path.join(_output_dir, thar_lamp + '_wave_coeffs.out')

if do_combine_twilight:
    combine.combine(twilights_b, twilight, _output_dir)
    print('-----------------FINISHED TWILIGHT COMBINATION-------------------')

if do_flat:
    twilight_fname = os.path.join(_output_dir, twilight + '.fits')
    flat.flat(sciences_b, twilight_fname, _output_dir, tracing_fname,
              tracing_fname, wave_fname)
    print('---------------------FINISHED FLATFIELDING-----------------------')

