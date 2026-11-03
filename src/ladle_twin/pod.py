"""Volume-weighted POD of the ladle fields (notebook 02 cell 16, notebook 03 cell 2).

The basis is built from the training block only; `coeffs` projects fields onto
it with the cell-volume weights and `lift` adds the mean back.
"""
from __future__ import annotations

import numpy as np


def build_basis(fields, wg, K=8):
    """Mean, K volume-weighted POD modes (in physical units) and cumulative energy.

    `fields` is (n, nz, nr) or (n, nz*nr); `wg` the flat cell volumes.
    Returns ``(Tmean, basis, energy)`` with ``basis`` of shape (K, nz*nr).
    """
    sw = np.sqrt(wg)
    Xtr = fields.reshape(len(fields), -1)
    Tmean = Xtr.mean(0)
    U, S, Vt = np.linalg.svd((Xtr - Tmean) * sw[None, :], full_matrices=False)
    basis = (Vt / sw[None, :])[:K]                     # K modes, back in physical units
    energy = np.cumsum(S**2) / (S**2).sum()
    return Tmean, basis, energy


def coeffs(fields, Tmean, basis, wg):
    """Project fields onto the modes (volume-weighted inner product)."""
    return ((fields.reshape(len(fields), -1) - Tmean) * wg[None, :]) @ basis.T


def lift(a, Tmean, basis, nz, nr):
    """Fields (n, nz, nr) from coefficients (n, K)."""
    return (Tmean + a @ basis).reshape(-1, nz, nr)


def leftover_energy(fields, Tmean, basis, wg):
    """Fraction of a field's (weighted) energy the basis cannot represent — the guard score.

    Notebook 02 cell 39 (``leftover_energy``) and notebook 03 cell 7 (``field_score``).
    """
    Xc = fields.reshape(len(fields), -1) - Tmean
    kept = ((Xc * wg[None, :]) @ basis.T) @ basis
    return ((Xc - kept)**2 * wg[None, :]).sum(1) / ((Xc**2 * wg[None, :]).sum(1) + 1e-12)
