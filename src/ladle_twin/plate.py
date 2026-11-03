"""Stage 1 — the parametric transient heat-conduction problem and its ROMs.

Every function here is copied from notebook 01 (cells 10, 11, 12, 24, 32, 33,
48 and 57) without change of behaviour; only the module-level dependencies
(the sweep positions `ts` of `knee`) became explicit arguments.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class Problem:
    """Container for one full-order transient-conduction solver."""
    pass


def build_problem(n_grid, dt=2e-4, n_steps=300):
    """Full-order model on an `n_grid` x `n_grid` grid of the unit square.

    Backward Euler, 5-point Laplacian, zero Dirichlet boundary, Gaussian source
    of width 0.09 at the parameter ``mu = (x0, y0)``. The step matrix is
    factorised once; ``p.solve_fom(mu)`` steps `n_steps` times from T = 0.
    """
    p = Problem()
    p.n_grid = n_grid
    p.h      = 1.0 / (n_grid - 1)        # uniform spacing, boundaries included
    p.n_int  = n_grid - 2                # interior points per side
    p.n_dof  = p.n_int * p.n_int         # full-order degrees of freedom

    xs = np.linspace(0, 1, n_grid)
    Xg, Yg = np.meshgrid(xs, xs, indexing="ij")
    p.x_int = Xg[1:-1, 1:-1].ravel()     # interior-node coordinates
    p.y_int = Yg[1:-1, 1:-1].ravel()

    # 2D Laplacian as a Kronecker sum of two 1D second-difference operators.
    eye = sp.identity(p.n_int)
    d2  = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(p.n_int, p.n_int)) / p.h**2
    p.laplacian = (sp.kron(d2, eye) + sp.kron(eye, d2)).tocsc()

    # Backward-Euler step matrix, factorized once and reused.
    p.dt, p.n_steps = dt, n_steps
    p.system    = (sp.identity(p.n_dof) - dt * p.laplacian).tocsc()
    p.system_lu = splu(p.system)

    # Gaussian source centered at mu = (x0, y0).
    def source_term(mu, width=0.09):
        x0, y0 = mu
        return 200.0 * np.exp(-((p.x_int - x0)**2 + (p.y_int - y0)**2) / (2 * width**2))
    p.source_term = source_term

    # Step the FOM to the final field, optionally keeping snapshots for POD.
    def solve_fom(mu, width=0.09, store_snapshots=False, every=10):
        T = np.zeros(p.n_dof)
        forcing = dt * source_term(mu, width)
        snaps = []
        for step in range(n_steps):
            T = p.system_lu.solve(T + forcing)      # reuse the LU factors
            if store_snapshots and step % every == 0:
                snaps.append(T.copy())
        return (T, snaps) if store_snapshots else T
    p.solve_fom = solve_fom

    return p


def as_grid(p, vec):
    """Reshape an interior-DOF vector to a plottable (y, x) image."""
    return vec.reshape(p.n_int, p.n_int).T


def build_rom(p, basis, rank):
    """POD-Galerkin ROM. Returns (Phi, rom, A_r)."""
    Phi   = basis[:, :rank]
    A_r   = Phi.T @ (p.system @ Phi)          # r x r reduced operator
    A_inv = np.linalg.inv(A_r)                # precompute once, reuse every step

    def rom(mu, width=0.09):
        a = np.zeros(rank)
        forcing = p.dt * (Phi.T @ p.source_term(mu, width))   # project source down
        for _ in range(p.n_steps):
            a = A_inv @ (a + forcing)                         # step in the subspace
        return Phi @ a                                        # lift back to full space
    return Phi, rom, A_r


def poly_features(M, degree=4, center=0.5):
    """Monomials of (x0, y0) up to `degree`, centered for conditioning."""
    x, y = M[:, 0] - center, M[:, 1] - center
    cols = [(x**(d - i)) * (y**i) for d in range(degree + 1) for i in range(d + 1)]
    return np.column_stack(cols)


def poly_features_3d(M, degree=3):
    """Monomials of (x0, y0, w) up to `degree` (Experiment 4's `ex_poly3`)."""
    x, y, w = M[:, 0] - 0.5, M[:, 1] - 0.5, (M[:, 2] - 0.12) / 0.06
    cols = [x**i * y**j * w**k for i in range(degree + 1) for j in range(degree + 1 - i) for k in range(degree + 1 - i - j)]
    return np.column_stack(cols)


def fit_ridge(F, A, lam=1e-4):
    """Ridge regression, returning the weight matrix (n_features x r)."""
    G = F.T @ F + lam * np.eye(F.shape[1])
    return np.linalg.solve(G, F.T @ A)


def snapshot_basis(p, mus, every=10):
    """POD basis and singular values of the transient snapshots of the FOM at `mus`.

    This is the data-building step of notebook 01, Section 2 (cell 21), as a
    function: stack the snapshots of every solve and take the thin SVD.
    """
    snaps = []
    for mu in mus:
        snaps += p.solve_fom(mu, store_snapshots=True, every=every)[1]
    S = np.column_stack(snaps)
    basis, sv, _ = np.linalg.svd(S, full_matrices=False)
    return basis, sv
