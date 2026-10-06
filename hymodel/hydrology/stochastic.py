# Stochastic Hydrological Models (Synthetic Time Series)

import warnings
import numpy as np
from ..common.validation import check_positive

__all__ = ["Stoch"]

class Stoch:
    "Stochastic Hydrological Models (Synthetic Time Series)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_params(params):
        # Converts a list/array of coefficients to a 1-D float array.
        # An empty input is valid (e.g. a pure MA model has no AR terms).
        return np.asarray(params, dtype=float).ravel()

    @staticmethod
    def _is_stationary(ar_params):
        # Checks stationarity of an AR(p) process.
        #
        # The process is stationary when the roots of 1 - phi_1*z - ... - phi_p*z^p
        # lie OUTSIDE the unit circle. Equivalently, the roots of the
        # companion polynomial  z^p - phi_1*z^(p-1) - ... - phi_p
        # (which are the reciprocals of the former) must lie INSIDE the unit
        # circle. np.roots computes the roots of the latter.
        p = len(ar_params)
        if p == 0:
            return True
        coeffs = np.concatenate(([1.0], -np.asarray(ar_params, dtype=float)))
        roots = np.roots(coeffs)
        return bool(np.all(np.abs(roots) < 1))

    @staticmethod
    def _combined_poly(params, seasonal_params, s, sign):
        # Builds the coefficient array (lag 0..order) of the product of a
        # non-seasonal lag polynomial and a seasonal lag polynomial, e.g.
        # (1 + sign*p1*B + sign*p2*B^2 + ...) * (1 + sign*P1*B^s + sign*P2*B^2s + ...)
        # Used to combine regular and seasonal AR/MA operators (Box-Jenkins
        # multiplicative form).
        ns_poly = np.zeros(len(params) + 1)
        ns_poly[0] = 1.0
        for i, val in enumerate(params):
            ns_poly[i + 1] = sign * val

        s_poly = np.zeros(len(seasonal_params) * s + 1)
        s_poly[0] = 1.0
        for j, val in enumerate(seasonal_params):
            s_poly[(j + 1) * s] = sign * val

        return np.convolve(ns_poly, s_poly)

    @staticmethod
    def _seasonal_cumsum(x, s):
        # Inverts a seasonal (lag-s) differencing operation: out[t] = out[t-s] + x[t]
        out = np.array(x, dtype=float).copy()
        for t in range(s, len(out)):
            out[t] += out[t - s]
        return out

    @staticmethod
    def _init_anomalies(init_value, order, mean):
        # Returns the last `order` values of init_value as anomalies from the
        # mean, or None when no valid initial condition was provided.
        if init_value is None or order <= 0 or len(init_value) < order:
            return None
        return np.asarray(init_value[-order:], dtype=float) - mean

    @staticmethod
    def _burn_in(max_lag, y0):
        # Warm-up length discarded from the start of the simulation, so the
        # series is not contaminated by the zero initial state. No burn-in is
        # used when the user supplied explicit initial conditions.
        return 0 if y0 is not None else 100 + 10 * max_lag

    @staticmethod
    def _run_arma(phi, theta, eps, y0=None):
        # Core ARMA recursion on anomalies:
        #   y[t] = sum_i phi[i]*y[t-1-i] + eps[t] + sum_j theta[j]*eps[t-1-j]
        # phi / theta are the full (already expanded) lag coefficients.
        p, q = len(phi), len(theta)
        L = max(p, q, 1)
        n = len(eps)

        y = np.zeros(n + L)
        e = np.concatenate((np.zeros(L), eps))

        if y0 is not None and p > 0:
            y[L - p:L] = y0

        for t in range(L, n + L):
            ar_term = np.dot(phi, y[t - p:t][::-1]) if p > 0 else 0.0
            ma_term = np.dot(theta, e[t - q:t][::-1]) if q > 0 else 0.0
            y[t] = ar_term + e[t] + ma_term

        return y[L:]

    @staticmethod
    def _rescale(series, std_dev):
        # Empirical variance correction to match the requested std_dev exactly.
        sd = np.std(series)
        if sd > 0:
            series = series / sd * std_dev
        return series

    # ------------------------------------------------------------------
    # Generative Models
    # ------------------------------------------------------------------
    @staticmethod
    def ar_p(n_steps, mean, std_dev, ar_params, init_value=None, seed=None):
        # Generate a synthetic series using a generic Autoregressive Model AR(p).
        # Generalizes the classic AR(1) formulation found in Bras & Rodriguez-Iturbe.
        #
        # ar_params: list/array [phi_1, ..., phi_p], where phi_1 is the lag-1
        # coefficient, phi_2 the lag-2 coefficient, etc.
        # seed: int, numpy Generator or None. Use it for reproducible series.
        #
        # Note: unlike the closed-form AR(1) noise factor (std_dev*sqrt(1-rho^2)),
        # there is no simple closed-form variance correction for a general AR(p).
        # Instead, we simulate on a demeaned series driven by unit white noise and
        # rescale the result empirically to match the requested std_dev exactly
        # (same strategy used in the arma() method).

        check_positive(n_steps, "n_steps")
        check_positive(std_dev, "std_dev")
        phi = Stoch._as_params(ar_params)
        p = len(phi)

        if not Stoch._is_stationary(phi):
            warnings.warn(
                "ar_p: the given ar_params do not satisfy the stationarity "
                "condition (roots of the characteristic polynomial inside "
                "the unit circle). The series may diverge."
            )

        rng = np.random.default_rng(seed)
        y0 = Stoch._init_anomalies(init_value, p, mean)
        burn = Stoch._burn_in(max(p, 1), y0)

        eps = rng.normal(0.0, 1.0, burn + n_steps)
        centered = Stoch._run_arma(phi, np.zeros(0), eps, y0)[burn:]

        return Stoch._rescale(centered, std_dev) + mean

    @staticmethod
    def ar_p_skewed(n_steps, mean, std_dev, ar_params, skewnesse, init_value=None, seed=None):
        # Generate a synthetic series using a generic AR(p) model with skewed
        # (Wilson-Hilferty / Gamma-type) innovations.
        # Generalizes the classic AR(1)-skewed formulation found in
        # Bras & Rodriguez-Iturbe.
        #
        # IMPORTANT / APPROXIMATION:
        # The original Bras & Rodriguez-Iturbe formula for gamma_e (the skew of
        # the innovations needed to obtain a target series skew) is derived
        # analytically for AR(1) only, as a function of the single lag-1
        # correlation coefficient. There is no simple closed-form equivalent for
        # a general AR(p). Here we approximate the "effective persistence" of
        # the process as sum(ar_params) and plug it into the same AR(1) formula.
        # This is a reasonable approximation for weakly-to-moderately correlated
        # processes, but the resulting series skewness will drift from the
        # requested `skewnesse` as p grows or as the AR structure becomes more
        # complex. Validate empirically (e.g. scipy.stats.skew) for your use case.

        check_positive(n_steps, "n_steps")
        check_positive(std_dev, "std_dev")
        phi = Stoch._as_params(ar_params)
        p = len(phi)

        if not Stoch._is_stationary(phi):
            warnings.warn(
                "ar_p_skewed: the given ar_params do not satisfy the "
                "stationarity condition. The series may diverge."
            )

        rng = np.random.default_rng(seed)
        y0 = Stoch._init_anomalies(init_value, p, mean)
        burn = Stoch._burn_in(max(p, 1), y0)

        # 1. Standard Gaussian noise (mean=0, var=1)
        wn = rng.normal(0.0, 1.0, burn + n_steps)

        # 2. Effective persistence measure used as a stand-in for AR(1)'s rho
        rho_eff = float(np.sum(phi))

        # 3. Noise skewness (Gamma E), approximated for AR(p)
        denom = 1 - rho_eff ** 2
        if abs(denom) < 1e-9:
            gamma_e = skewnesse
        else:
            gamma_e = skewnesse * ((1 - rho_eff ** 3) / denom ** 1.5)

        # 4. Wilson-Hilferty transformation to obtain Gamma noise
        # (skipped if the target is perfectly symmetric, to avoid division by zero)
        if abs(gamma_e) < 10e-6:
            e = wn
        else:
            e = (2 / gamma_e) * ((1 + (gamma_e * wn / 6) - (gamma_e ** 2 / 36)) ** 3) - (2 / gamma_e)

        # 5. AR(p) recursion driven by the skewed noise
        centered = Stoch._run_arma(phi, np.zeros(0), e, y0)[burn:]

        return Stoch._rescale(centered, std_dev) + mean

    @staticmethod
    def arma(n_steps, mean, std_dev, ar_params, ma_params, init_value=None, seed=None):
        # Generate a synthetic series using the ARMA(p, q) Model
        # Based on the classic Box-Jenkins formulation.

        check_positive(n_steps, "n_steps")
        check_positive(std_dev, "std_dev")
        phi = Stoch._as_params(ar_params)
        theta = Stoch._as_params(ma_params)
        p, q = len(phi), len(theta)

        if not Stoch._is_stationary(phi):
            warnings.warn(
                "arma: the given ar_params do not satisfy the stationarity "
                "condition. The series may diverge."
            )

        rng = np.random.default_rng(seed)
        y0 = Stoch._init_anomalies(init_value, p, mean)
        burn = Stoch._burn_in(max(p, q, 1), y0)

        eps = rng.normal(0.0, 1.0, burn + n_steps)
        centered = Stoch._run_arma(phi, theta, eps, y0)[burn:]

        # The interaction between AR and MA parameters distorts the original
        # variance, so we rescale to the exact std_dev requested.
        return Stoch._rescale(centered, std_dev) + mean

    @staticmethod
    def sarima(n_steps, mean, std_dev, ar_params, ma_params,
               seasonal_ar_params, seasonal_ma_params, seasonal_period,
               d=0, D=0, init_value=None, seed=None):
        # Generate a synthetic series using a Seasonal ARIMA(p,d,q)(P,D,Q)_s model
        # (Box-Jenkins multiplicative formulation).
        #
        # ar_params / ma_params: non-seasonal AR(p) / MA(q) coefficients.
        # seasonal_ar_params / seasonal_ma_params: seasonal AR(P) / MA(Q) coefficients,
        # applied at lag multiples of `seasonal_period` (s).
        # seasonal_period: seasonal cycle length (e.g. 12 for monthly data with
        # annual seasonality).
        # d, D: non-seasonal and seasonal differencing orders. Use d=D=0 for a
        # stationary series (typical for hydrological synthetic generation,
        # e.g. streamflow around a fixed mean). d>0 or D>0 produce a
        # non-stationary (integrated) series with drift/trend, following the
        # standard ARIMA definition.
        # d/D > 0: when differencing is applied, the process is no longer
        # stationary and does not have a fixed statistical mean. In that case,
        # `mean` is used as a constant baseline level added to the whole series
        # (not as the exact long-run average), and `std_dev` refers to the
        # standard deviation of the underlying *stationary* ARMA component
        # before integration, not of the final integrated series.

        check_positive(n_steps, "n_steps")
        check_positive(std_dev, "std_dev")
        check_positive(seasonal_period, "seasonal_period")

        s = int(seasonal_period)
        ar_params = Stoch._as_params(ar_params)
        ma_params = Stoch._as_params(ma_params)
        seasonal_ar_params = Stoch._as_params(seasonal_ar_params)
        seasonal_ma_params = Stoch._as_params(seasonal_ma_params)

        # 1. Combine regular and seasonal AR / MA operators (Box-Jenkins
        # multiplicative expansion: phi(B)*Phi(B^s), theta(B)*Theta(B^s)).
        # ar_full / ma_full are lag polynomials with leading 1; the recursion
        # coefficients are phi_k = -ar_full[k] and theta_k = ma_full[k].
        ar_full = Stoch._combined_poly(ar_params, seasonal_ar_params, s, sign=-1)
        ma_full = Stoch._combined_poly(ma_params, seasonal_ma_params, s, sign=1)

        phi = -ar_full[1:]
        theta = ma_full[1:]
        ar_order, ma_order = len(phi), len(theta)

        if not Stoch._is_stationary(phi):
            warnings.warn(
                "sarima: the combined AR/seasonal-AR parameters do not satisfy "
                "the stationarity condition. The stationary component may diverge."
            )

        rng = np.random.default_rng(seed)
        y0 = Stoch._init_anomalies(init_value, ar_order, mean)
        burn = Stoch._burn_in(max(ar_order, ma_order, 1), y0)

        # 2. Multiplicative seasonal ARMA simulation (after burn-in)
        eps = rng.normal(0.0, 1.0, burn + n_steps)
        stationary_series = Stoch._run_arma(phi, theta, eps, y0)[burn:]

        # 3. Empirical variance correction on the stationary component
        stationary_series = Stoch._rescale(stationary_series, std_dev)

        # 4. Integrate: invert seasonal differencing D times, then regular differencing d times
        integrated = stationary_series
        for _ in range(D):
            integrated = Stoch._seasonal_cumsum(integrated, s)
        for _ in range(d):
            integrated = np.cumsum(integrated)

        # 5. Add the mean/baseline level back
        return integrated + mean
