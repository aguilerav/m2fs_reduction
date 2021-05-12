import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from m2fs_pipeline import basic

_scripts_dir = os.path.dirname(os.path.realpath(__file__))
_assets_dir = os.path.dirname(_scripts_dir)
_raw_images_dir = os.path.join(_assets_dir, 'raw_images')
_output_dir = os.path.join(_assets_dir, 'temp_products')
assets_dir = os.path.join(os.path.dirname(os.path.dirname(_assets_dir)), 'assets')


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
do_basic = True


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
    print('------------------FINISHED BIAS/TRIM/GAIN--------------------')