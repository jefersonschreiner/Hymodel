# Physical Constants and Unit Conversions

import numpy as np

# Physical Constants
G = 9.80665  # Acceleration due to gravity (m/s^2)
GAMMA_AGUA = 9810.0  # Specific weight of water (N/m^3) a 20°C
RHO_AGUA = 1000.0  # Density of water (kg/m^3) a 4°C
NU_AGUA = 1.003e-6  # Kinematic viscosity of water (m^2/s) a 20°C

# Unit Conversion Functions
def cv_to_kw(cv):
    # Convert horsepower (cv) to kilowatts (kW)
    return cv * 0.7355

def kw_to_cv(kw):
    # Convert kilowatts (kW) to horsepower (cv)
    return kw / 0.7355

def m_to_ft(m):
    # Convert meters to feet
    return m * 3.28084

def ft_to_m(ft):
    # Convert feet to meters
    return ft / 3.28084

def m3s_to_lps(Q):
    # Convert cubic meters per second (m^3/s) to liters per second (L/s)
    return Q * 1000.0

def lps_to_m3s(Q):
    # Convert liters per second (L/s) to cubic meters per second (m^3/s)
    return Q / 1000.0

def mm_to_m(mm):
    # Convert millimeters to meters
    return mm / 1000.0

def m_to_mm(m):
    # Convert meters to millimeters
    return m * 1000.0

def km2_to_m2(km2):
    # Convert square kilometers to square meters
    return km2 * 1e6

def hectare_to_m2(ha):
    # Convert hectares to square meters
    return ha * 1e4



