# Performance metrics for hydrological models

import numpy as np

def rmse(obs, sim):
    # Root Mean Square Error (RMSE)
    return np.sqrt(np.mean((obs - sim)**2))


def nash_sutcliffe(obs, sim):
    # Nash-Sutcliffe Efficiency (NSE)
    return 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)


def kge(obs, sim):
    # Kling-Gupta Efficiency
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def pbias(obs, sim):
    # Percent Bias (PBIAS)
    return 100 * np.sum(sim - obs) / np.sum(obs)