from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline


# Quantile grid ported from CRAN kSamples::ad.pval(version = 1).
# The table contains simulated upper-tail quantiles for the standardized
# k-sample Anderson-Darling T_m statistic from Scholz & Stephens (1987),
# "K-Sample Anderson-Darling Tests", JASA 82:918-924. The kSamples source
# notes that these values were estimated with Nsim = 2e6 and sample size
# n = 500 to emulate the paper's Table 1, then used for p-value interpolation.
# Source: https://rdrr.io/cran/kSamples/src/R/ad.pval.R
_KSAMPLES_AD_TABLE_V1 = np.array(
    [
        1, 2, 3, 4, 6, 8, 10, np.inf, -1.1954, -1.5806, -1.8172, -2.0032, -2.2526, -2.4204, -2.5283, -4.2649,
        -1.1786, -1.5394, -1.7728, -1.9426, -2.1685, -2.3288, -2.4374, -3.8906, -1.166, -1.5193, -1.7462,
        -1.9067, -2.126, -2.2818, -2.3926, -3.719, -1.1407, -1.4659, -1.671, -1.8105, -2.0048, -2.1356,
        -2.2348, -3.2905, -1.1253, -1.4371, -1.6314, -1.7619, -1.9396, -2.0637, -2.1521, -3.0902, -1.0777,
        -1.3503, -1.5102, -1.6177, -1.761, -1.8537, -1.9178, -2.5758, -1.0489, -1.2984, -1.4415, -1.5355,
        -1.6625, -1.738, -1.7936, -2.3263, -0.9978, -1.2098, -1.3251, -1.4007, -1.4977, -1.5555, -1.5941,
        -1.96, -0.9417, -1.1187, -1.209, -1.2671, -1.3382, -1.379, -1.405, -1.6449, -0.8981, -1.0491,
        -1.1235, -1.1692, -1.2249, -1.2552, -1.2755, -1.4395, -0.8598, -0.9904, -1.0513, -1.0879, -1.1317,
        -1.155, -1.1694, -1.2816, -0.7258, -0.7938, -0.8188, -0.8312, -0.8435, -0.8471, -0.8496, -0.8416,
        -0.5966, -0.617, -0.6177, -0.6139, -0.6073, -0.5987, -0.5941, -0.5244, -0.4572, -0.4383, -0.419,
        -0.4033, -0.3834, -0.3676, -0.3587, -0.2533, -0.2966, -0.2428, -0.2078, -0.1844, -0.1548, -0.1346,
        -0.1224, 0, -0.1009, -0.0169, 0.0304, 0.0596, 0.0933, 0.1156, 0.1294, 0.2533, 0.1571, 0.2635,
        0.3169, 0.348, 0.3823, 0.4038, 0.4166, 0.5244, 0.5357, 0.6496, 0.6992, 0.7246, 0.7528, 0.7683,
        0.7771, 0.8416, 1.2255, 1.2989, 1.3202, 1.3254, 1.3305, 1.3286, 1.3257, 1.2816, 1.5262, 1.5677,
        1.5709, 1.5663, 1.5561, 1.5449, 1.5356, 1.4395, 1.9633, 1.943, 1.919, 1.8975, 1.8641, 1.8389,
        1.8212, 1.6449, 2.7314, 2.5899, 2.5, 2.4451, 2.3664, 2.3155, 2.2823, 1.96, 3.7825, 3.4425, 3.2582,
        3.1423, 3.0036, 2.9101, 2.8579, 2.3263, 4.1241, 3.716, 3.4984, 3.3651, 3.2003, 3.0928, 3.0311,
        2.4324, 4.6044, 4.0847, 3.8348, 3.6714, 3.4721, 3.3453, 3.2777, 2.5758, 5.409, 4.7223, 4.4022,
        4.1791, 3.9357, 3.7809, 3.6963, 2.807, 6.4954, 5.5823, 5.1456, 4.8657, 4.5506, 4.3275, 4.2228,
        3.0902, 6.8279, 5.8282, 5.3658, 5.0749, 4.7318, 4.4923, 4.3642, 3.1747, 7.2755, 6.197, 5.6715,
        5.3642, 4.9991, 4.7135, 4.5945, 3.2905, 8.1885, 6.8537, 6.2077, 5.8499, 5.4246, 5.1137, 4.9555,
        3.4808, 9.3061, 7.6592, 6.85, 6.4806, 5.9919, 5.6122, 5.5136, 3.719, 9.6132, 7.9234, 7.1025,
        6.6731, 6.1549, 5.8217, 5.7345, 3.7911, 10.0989, 8.2395, 7.4326, 6.9567, 6.3908, 6.011, 5.9566,
        3.8906, 10.8825, 8.8994, 7.8934, 7.4501, 6.9009, 6.4538, 6.2705, 4.0556, 11.8537, 9.5482, 8.5568,
        8.0283, 7.4418, 6.9524, 6.6195, 4.2649,
    ],
    dtype=float,
).reshape(8, 36, order="F")
_KSAMPLES_AD_M = _KSAMPLES_AD_TABLE_V1[:, 0]
_KSAMPLES_AD_QUANTILES_V1 = _KSAMPLES_AD_TABLE_V1[:, 1:]
_KSAMPLES_AD_SQRT_M = np.where(np.isfinite(_KSAMPLES_AD_M), 1.0 / np.sqrt(_KSAMPLES_AD_M), 0.0)
_KSAMPLES_AD_TAIL_PROBS = 1.0 - np.array(
    [
        0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99, 0.9925, 0.995, 0.9975, 0.999, 0.99925, 0.9995,
        0.99975, 0.9999, 0.999925, 0.99995, 0.999975, 0.99999,
    ],
    dtype=float,
)
_KSAMPLES_AD_LOGIT_PROBS = np.log(_KSAMPLES_AD_TAIL_PROBS / (1.0 - _KSAMPLES_AD_TAIL_PROBS))


def _expit(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _ksamples_ad_pvalue(tx: float, m: float, version: int = 1) -> float:
    if not np.isfinite(tx):
        return math.nan
    if m < 1:
        raise ValueError("m must be >= 1")
    if version != 1:
        raise ValueError("Only kSamples AD version 1 is implemented")

    if math.isinf(m):
        quantiles = _KSAMPLES_AD_QUANTILES_V1[-1]
    elif m == 1:
        quantiles = _KSAMPLES_AD_QUANTILES_V1[0]
    else:
        sqm0 = 1.0 / math.sqrt(m)
        quantiles = np.array(
            [float(CubicSpline(_KSAMPLES_AD_SQRT_M, _KSAMPLES_AD_QUANTILES_V1[:, idx], bc_type="natural")(sqm0))
             for idx in range(_KSAMPLES_AD_QUANTILES_V1.shape[1])],
            dtype=float,
        )

    spline = CubicSpline(quantiles, _KSAMPLES_AD_LOGIT_PROBS, bc_type="natural", extrapolate=True)
    logit_p = float(spline(tx))
    left_slope = (_KSAMPLES_AD_LOGIT_PROBS[1] - _KSAMPLES_AD_LOGIT_PROBS[0]) / (quantiles[1] - quantiles[0])
    right_slope = (_KSAMPLES_AD_LOGIT_PROBS[-1] - _KSAMPLES_AD_LOGIT_PROBS[-2]) / (quantiles[-1] - quantiles[-2])

    if tx < quantiles[0]:
        logit_p = float(_KSAMPLES_AD_LOGIT_PROBS[0] + left_slope * (tx - quantiles[0]))
    elif tx > quantiles[-1]:
        logit_p = float(_KSAMPLES_AD_LOGIT_PROBS[-1] + right_slope * (tx - quantiles[-1]))

    return _expit(logit_p)


def _ksamples_ad_statistic(samples: list[np.ndarray]) -> tuple[float, float]:
    if len(samples) < 2:
        raise ValueError("AD test requires at least two samples")

    samples_sorted = [np.sort(np.asarray(sample, dtype=float)) for sample in samples]
    ns = np.array([len(sample) for sample in samples_sorted], dtype=int)
    if np.any(ns == 0):
        raise ValueError("AD test encountered empty sample")

    pooled = np.concatenate(samples_sorted)
    zstar = np.unique(np.sort(pooled))
    if len(zstar) < 2:
        raise ValueError("AD test requires more than one distinct observation")

    k = len(samples_sorted)
    nsum = int(ns.sum())
    lvec = np.zeros(len(zstar), dtype=int)
    fij = np.zeros((k, len(zstar)), dtype=int)

    for idx, sample in enumerate(samples_sorted):
        left = np.searchsorted(sample, zstar, side="left")
        right = np.searchsorted(sample, zstar, side="right")
        fij[idx] = right - left
        lvec += fij[idx]

    bj = np.cumsum(lvec).astype(float)
    baj = bj - lvec / 2.0

    akn2 = 0.0
    aakn2 = 0.0
    for idx in range(k):
        nij = float(ns[idx])
        mij = np.cumsum(fij[idx]).astype(float)
        maij = mij - fij[idx] / 2.0

        if len(zstar) > 1:
            tmp = nsum * mij[:-1] - nij * bj[:-1]
            denom = bj[:-1] * (nsum - bj[:-1])
            akn2 += float(np.sum(lvec[:-1] * tmp * tmp / denom) / nij)

        tmp = nsum * maij - nij * baj
        denom = baj * (nsum - baj) - nsum * lvec / 4.0
        aakn2 += float(np.sum(lvec * tmp * tmp / denom) / nij)

    akn2 = akn2 / nsum
    aakn2 = (nsum - 1.0) * aakn2 / (nsum * nsum)
    return float(akn2), float(aakn2)


def safe_anderson_pvalue(values_x: np.ndarray, values_y: np.ndarray) -> tuple[float, float]:
    try:
        samples = [np.asarray(values_x, dtype=float), np.asarray(values_y, dtype=float)]
        akn2, _ = _ksamples_ad_statistic(samples)
        k = len(samples)
        n = sum(len(sample) for sample in samples)
        if n > 3:
            h = float(np.sum(1.0 / np.arange(1, n)))
            g = 0.0
            for idx in range(1, n - 1):
                g += (1.0 / (n - idx)) * float(np.sum(1.0 / np.arange(idx + 1, n)))
            H = float(np.sum(1.0 / np.array([len(sample) for sample in samples], dtype=float)))
            coef_a = (4.0 * g - 6.0) * (k - 1.0) + (10.0 - 6.0 * g) * H
            coef_b = (2.0 * g - 4.0) * k**2 + 8.0 * h * k + (2.0 * g - 14.0 * h - 4.0) * H - 8.0 * h + 4.0 * g - 6.0
            coef_c = (6.0 * h + 2.0 * g - 2.0) * k**2 + (4.0 * h - 4.0 * g + 6.0) * k + (2.0 * h - 6.0) * H + 4.0 * h
            coef_d = (2.0 * h + 6.0) * k**2 - 4.0 * h * k
            sig2 = (coef_a * n**3 + coef_b * n**2 + coef_c * n + coef_d) / ((n - 1.0) * (n - 2.0) * (n - 3.0))
            sig = math.sqrt(sig2)
            tk = (akn2 - (k - 1.0)) / sig
            return akn2, _ksamples_ad_pvalue(tk, k - 1.0, version=1)
        if n == 3 and k == 2:
            tk = (akn2 - 1.0) / 0.3535534
            return akn2, _ksamples_ad_pvalue(tk, 1.0, version=1)
        return akn2, 1.0
    except Exception:
        return 0.0, 1.0
