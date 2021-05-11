# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='m2fs_pipeline',
    version='0.0.1',
    description='Pipeline to reduce M2FS data',
    long_description=readme,
    author='Valentino Gonzalez, Vicente Donaire, Grecco Oyarzun',
    author_email='vdonaire@das.uchile.cl',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'example')),
    scripts=['bin/m2fs-db', 'bin/m2fs-compile-scripts', 'bin/m2fs-runbasic'],
)


