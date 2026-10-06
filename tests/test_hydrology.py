import warnings

import numpy as np
import pytest

from hymodel.hydrology.stochastic import Stoch


def acf(x, lag):
    x = np.asarray(x) - np.mean(x)
    return np.sum(x[lag:] * x[:-lag]) / np.sum(x * x)


# ---------------- stationarity ----------------
def test_stationarity_check():
    assert Stoch._is_stationary([0.5])
    assert Stoch._is_stationary([0.5, 0.3])
    assert Stoch._is_stationary([])
    assert not Stoch._is_stationary([1.1])
    assert not Stoch._is_stationary([0.6, 0.6])


def test_stationary_params_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Stoch.ar_p(100, 10, 2, [0.5], seed=1)


def test_nonstationary_params_warn():
    with pytest.warns(UserWarning):
        Stoch.ar_p(50, 10, 2, [1.1], seed=1)


# ---------------- AR(p) ----------------
def test_ar1_mean_std_and_autocorrelation():
    x = Stoch.ar_p(20000, 50.0, 8.0, [0.7], seed=42)
    assert len(x) == 20000
    assert np.mean(x) == pytest.approx(50.0, abs=0.5)
    assert np.std(x) == pytest.approx(8.0)
    assert acf(x, 1) == pytest.approx(0.7, abs=0.03)


def test_ar2_autocorrelation():
    phi1, phi2 = 0.5, 0.3
    x = Stoch.ar_p(30000, 0.0, 1.0, [phi1, phi2], seed=3)
    rho1 = phi1 / (1 - phi2)
    assert acf(x, 1) == pytest.approx(rho1, abs=0.03)


def test_reproducible_with_seed():
    a = Stoch.ar_p(100, 0, 1, [0.5], seed=7)
    b = Stoch.ar_p(100, 0, 1, [0.5], seed=7)
    assert np.array_equal(a, b)


def test_init_value_is_used():
    x = Stoch.ar_p(5, 10.0, 1.0, [0.9], init_value=[100.0], seed=1)
    assert x[0] > 10.0  # starts near the (high) initial value, not the mean


# ---------------- skewed AR ----------------
def test_ar_skewed_is_positively_skewed():
    from scipy.stats import skew
    x = Stoch.ar_p_skewed(50000, 20.0, 5.0, [0.3], 1.5, seed=5)
    assert np.mean(x) == pytest.approx(20.0, abs=0.3)
    assert np.std(x) == pytest.approx(5.0)
    assert skew(x) > 0.8


# ---------------- ARMA ----------------
def test_arma_std_and_ma1_autocorrelation():
    theta = 0.6
    x = Stoch.arma(30000, 5.0, 2.0, [], [theta], seed=11)
    assert np.std(x) == pytest.approx(2.0)
    assert acf(x, 1) == pytest.approx(theta / (1 + theta ** 2), abs=0.03)
    assert abs(acf(x, 3)) < 0.03


# ---------------- SARIMA ----------------
def test_sarima_seasonal_autocorrelation():
    x = Stoch.sarima(40000, 100.0, 10.0, [], [], [0.6], [], 12, seed=2)
    assert np.std(x) == pytest.approx(10.0)
    assert acf(x, 12) == pytest.approx(0.6, abs=0.03)
    assert abs(acf(x, 5)) < 0.05


def test_sarima_combined_poly_expansion():
    # (1 - 0.5B)(1 - 0.4B^12) = 1 - 0.5B - 0.4B^12 + 0.2B^13
    full = Stoch._combined_poly([0.5], [0.4], 12, sign=-1)
    assert full[1] == pytest.approx(-0.5)
    assert full[12] == pytest.approx(-0.4)
    assert full[13] == pytest.approx(0.2)


def test_sarima_integration_shapes():
    x = Stoch.sarima(120, 0.0, 1.0, [0.3], [], [0.2], [], 12, d=1, D=1, seed=1)
    assert x.shape == (120,)


def test_validation_errors():
    with pytest.raises(ValueError):
        Stoch.ar_p(0, 0, 1, [0.5])
    with pytest.raises(ValueError):
        Stoch.ar_p(10, 0, -1, [0.5])
