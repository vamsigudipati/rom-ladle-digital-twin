"""The pieces of the ladle twin (notebooks 02 and 03), as pure functions.

Sensors, physics core, residual basis, the neural soft sensor and the
sensor-space guard. The volume-weighted POD lives in :mod:`ladle_twin.pod`.
"""
from __future__ import annotations

import numpy as np

from .ladle import temperature

Q_REF = 5e-4          # gas-flow scale: the notebooks feed Q / 5e-4 to the maps


def cell_index(g, rr, zz):
    """Flat index of the grid cell nearest to radius `rr` and height `zz` (metres)."""
    r, z, nr = g["r"], g["z"], g["nr"]
    return int(np.argmin(np.abs(z-zz)))*nr + int(np.argmin(np.abs(r-rr)))


def two_lance_probes(g, radii=(0.1, 0.9), n_per_lance=8):
    """The notebooks' 16 thermocouples: two vertical lances at 10 % and 90 % of the radius."""
    R, H = g["R"], g["H"]
    return [cell_index(g, rr*R, zz*H) for rr in radii for zz in np.linspace(0.06, 0.94, n_per_lance)]


def gas_flow_input(lat):
    """The scaled gas-flow reading appended to the thermocouples (notebook 02 cell 24)."""
    return lat[:, 0] / Q_REF


def read_sensors(F, idx, probes, Q, rng, noise_std=2.0):
    """Readings of the probes on heats `idx`, plus Gaussian noise, plus the known gas flow.

    Draws ``rng.normal(0, noise_std, (len(idx), len(probes)))`` exactly as the
    notebooks do, so a fresh ``default_rng(0)`` called first on the training
    heats and then on the test heats reproduces their sensor draws.
    """
    tc = F[idx].reshape(len(idx), -1)[:, probes] + rng.normal(0, noise_std, (len(idx), len(probes)))
    return np.column_stack([tc, Q[idx]])


def physics_core(g, lat, nom, idx):
    """Analytic field from the KNOWN inputs (gas flow, wall cooling); nominal for the hidden state, swirl 0."""
    out = []
    for i in idx:
        p = nom.copy()
        p[0] = lat[i, 0]
        p[1] = lat[i, 1]
        p[4] = 0.0   # know Q & wall, swirl unknown -> 0
        out.append(temperature(g, tuple(p)).ravel())
    return np.array(out)


def residual_basis(residuals, wg, K=8):
    """Mean and K volume-weighted POD modes of the residual ``T_true - T_core`` (notebook 02 cell 24)."""
    sw = np.sqrt(wg)
    Rmean = residuals.mean(0)
    _, _, VtR = np.linalg.svd((residuals - Rmean) * sw[None, :], full_matrices=False)
    basisR = (VtR / sw[None, :])[:K]
    return Rmean, basisR


def residual_coeffs(Rf, Rmean, basisR, wg):
    """Project residual fields onto the residual modes."""
    return ((Rf - Rmean) * wg[None, :]) @ basisR.T


def sensor_score(readings, PhiP, muP):
    """Sensor-space guard (notebook 03 cell 7): share of the readings' energy the sampled modes cannot fit.

    `PhiP` is ``basis[:, probes].T`` (m x K) and `muP` the mean field at the probes.
    """
    d = readings - muP; a = np.linalg.lstsq(PhiP, d.T, rcond=None)[0]; res = d - (PhiP @ a).T
    return (res**2).sum(1) / ((d**2).sum(1) + 1e-12)


def make_soft_sensor(n_in=17, hidden=64, n_out=8):
    """The neural soft sensor of notebook 02: ``Linear(n_in, 64) – GELU – Linear(64, 8)``."""
    import torch.nn as nn
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


def train_soft_sensor(net, Xt, Yt, steps=1500, lr=3e-3, weight_decay=1e-4, hook=None):
    """Full-batch Adam on the MSE of scaled coefficients, as in notebook 02 cell 33.

    `hook(net)` is called after every optimiser step when given (the notebook's
    animation hook); it must not touch the random state.
    """
    import torch
    import torch.nn as nn
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        nn.MSELoss()(net(Xt), Yt).backward()
        opt.step()
        if hook is not None:
            hook(net)
    return net
