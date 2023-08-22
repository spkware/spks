#!/usr/bin/env python
import os
from os.path import join as pjoin
from setuptools import setup
from setuptools.command.install import install

longdescription = ''' Tools to analyse large scale electrophysiology.
 The aim is to be as simple and transparent as possible.
 
'''

setup(
    name = 'spks',
    version = '0.0b',
    author = 'Joao Couto and Max Melin',
    author_email = 'jpcouto@gmail.com',
    description = ('Large-scale electrophysiology analysis.'),
    long_description = longdescription,
    license = 'GPL',
    packages = ['spks'],
    entry_points = {
      'console_scripts': [
      ]
    }
)
