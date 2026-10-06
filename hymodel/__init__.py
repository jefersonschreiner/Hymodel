# Hymodel - A Python package for modeling and simulating Water Resources systems

'''
Source:

    - Hydrology: 
    - Hydraulics:
    - Hydrosedimentology:
    - Analysis:
    - Common:

'''

__version__ = '0.1.0'
__author__ = 'Jeferson Schreiner Junior'

from . import hydrology
from . import hydraulics
from . import hydrosedimentology
from . import analysis
from . import common

__all__ = ['hydrology', 
           'hydraulics', 
           'hydrosedimentology', 
           'analysis', 
           'common']


