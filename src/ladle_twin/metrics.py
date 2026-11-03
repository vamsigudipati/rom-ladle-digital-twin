"""Error measures and plant KPIs used by the three notebooks."""
from __future__ import annotations

import numpy as np


def rel_error(ref, pred):
    """Relative L2 error of one field against a reference (notebook 01, ``ex_rel``)."""
    return float(np.linalg.norm(ref - pred) / np.linalg.norm(ref))


def knee(errs, ts, thr=0.05):
    """First sweep position `ts[i]` at which `errs[i]` exceeds `thr` (notebook 01, ``ex_knee``); None if never."""
    over = np.flatnonzero(np.asarray(errs) > thr)
    return float(ts[over[0]]) if len(over) else None


def rel_l2(Fhat, Ftrue, Tmean, wg):
    """Per-field relative L2 error of the *variation* (notebook 02 ``relL2``).

    Numerator: volume-weighted squared error. Denominator: weighted squared
    deviation of the true field from the training mean.
    """
    num = (((Fhat - Ftrue).reshape(len(Ftrue), -1)**2) * wg).sum(1)
    den = (((Ftrue.reshape(len(Ftrue), -1) - Tmean)**2) * wg).sum(1) + 1e-12
    return np.sqrt(num / den)


def rms_kelvin(Fhat, Ftrue, wg):
    """Per-field volume-weighted RMS error in kelvin (notebook 02 cell 28, notebook 03 ``rmsK``)."""
    return np.sqrt((((Fhat - Ftrue).reshape(len(Ftrue), -1)**2) * wg).sum(1) / wg.sum())


def kpi_masks(g):
    """Boolean flat masks of the wall band, the top fifth and the bottom fifth of the bath (notebook 02 cell 36)."""
    Rg, Zg, R, H = g["Rg"], g["Zg"], g["R"], g["H"]
    wall_band = (Rg > 0.85*R).ravel()
    top = (Zg > 0.8*H).ravel()
    bot = (Zg < 0.2*H).ravel()
    return wall_band, top, bot


def kpis(Fs, wg, wall_band, top, bot, T_bath=1873.0):
    """Bath superheat, near-wall temperature and bottom-minus-top stratification per field."""
    X = Fs.reshape(len(Fs), -1)
    superheat = (X * wg).sum(1)/wg.sum() - T_bath                 # bath superheat (K)
    wall_load = X[:, wall_band].mean(1)                       # near-wall temperature
    strat     = X[:, bot].mean(1) - X[:, top].mean(1)         # bottom-minus-top
    return dict(superheat=superheat, wall_load=wall_load, strat=strat)


def r2(t, p):
    """Coefficient of determination of predictions `p` against truth `t`."""
    return float(1 - ((t - p)**2).sum() / ((t - t.mean())**2).sum())


def kpi_r2(Fhat, Ftrue, wg, wall_band, top, bot):
    """R² of each KPI of the reconstructed fields against the true ones (notebook 02 cell 50)."""
    Kt_, Kp_ = kpis(Ftrue, wg, wall_band, top, bot), kpis(Fhat, wg, wall_band, top, bot)
    return {k_: r2(Kt_[k_], Kp_[k_]) for k_ in Kt_}
