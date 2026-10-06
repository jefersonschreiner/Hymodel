# Validation and Input Verification Functions

import numpy as np

def check_positive(value, name):
    # Check if a value is positive
    if value <= 0:
        raise ValueError(f"{name} must be positive. Got {value}.")
    return value

def check_range(value, min_value, max_value, name):
    # Check if a value is within a specified range
    if not (min_value <= value <= max_value):
        raise ValueError(f"{name} must be between {min_value} and {max_value}. Got {value}.")
    return value

def check_array(arr, name):
    # Check if an array is a numpy array and not empty
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} must not contain NaN values.")
    return arr