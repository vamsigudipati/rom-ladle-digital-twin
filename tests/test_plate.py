"""Stage 1 solver and reduced-order models."""
import numpy as np
import pytest

from ladle_twin.plate import (as_grid, build_problem, build_rom, fit_ridge, poly_features,
                              poly_features_3d, snapshot_basis)


def test_problem_sizes(tiny_problem):
    p = tiny_problem
    assert (p.n_int, p.n_dof) == (6, 36)
    assert p.system.shape == (36, 36) and p.laplacian.shape == (36, 36)
    assert p.x_int.shape == (36,) and 0 < p.x_int.min() and p.x_int.max() < 1


def test_laplacian_is_symmetric_negative(tiny_problem):
    L = tiny_problem.laplacian.toarray()
    assert np.allclose(L, L.T)
    assert np.all(np.linalg.eigvalsh(L) < 0)


def test_source_peaks_at_mu(tiny_problem):
    p = tiny_problem
    f = p.source_term((0.5, 0.5))
    assert f.shape == (p.n_dof,) and f.max() <= 200.0
    k = np.argmax(f)
    assert abs(p.x_int[k] - 0.5) < p.h and abs(p.y_int[k] - 0.5) < p.h


def test_solve_fom_shapes_and_snapshots(tiny_problem):
    p = tiny_problem
    T = p.solve_fom((0.5, 0.5))
    assert T.shape == (p.n_dof,)
    T2, snaps = p.solve_fom((0.5, 0.5), store_snapshots=True, every=5)
    assert np.array_equal(T, T2) and len(snaps) == p.n_steps // 5


def test_fom_is_positive_and_symmetric_for_central_source(tiny_problem):
    """A positive source with zero boundary gives a positive field, symmetric under x <-> y."""
    p = tiny_problem
    T = p.solve_fom((0.5, 0.5))
    assert T.min() > 0
    G = as_grid(p, T)
    assert np.allclose(G, G.T, atol=1e-10)


def test_as_grid_roundtrip(tiny_problem):
    p = tiny_problem
    v = np.arange(p.n_dof, dtype=float)
    assert as_grid(p, v).shape == (p.n_int, p.n_int)
    assert as_grid(p, v)[2, 1] == v.reshape(p.n_int, p.n_int)[1, 2]


def test_full_basis_rom_reproduces_fom(tiny_problem):
    """Galerkin projection onto the whole space is the full-order model itself."""
    p = tiny_problem
    Phi, rom, A_r = build_rom(p, np.eye(p.n_dof), p.n_dof)
    assert A_r.shape == (p.n_dof, p.n_dof)
    assert np.allclose(A_r, p.system.toarray())
    mu = (0.4, 0.6)
    assert np.allclose(rom(mu), p.solve_fom(mu), atol=1e-9)


def test_pod_rom_error_falls_with_rank_to_zero_at_full_rank(tiny_problem):
    p = tiny_problem
    mus = np.array([[0.4, 0.4], [0.5, 0.6], [0.6, 0.45], [0.45, 0.55]])
    basis, sv = snapshot_basis(p, mus, every=2)
    assert basis.shape == (p.n_dof, p.n_dof) and np.all(np.diff(sv) <= 1e-12)
    assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)
    ref = p.solve_fom((0.5, 0.5))
    errs = []
    for r in (3, 6, 12, p.n_dof):
        _, rom, A_r = build_rom(p, basis, r)
        assert A_r.shape == (r, r)
        errs.append(np.linalg.norm(ref - rom((0.5, 0.5))) / np.linalg.norm(ref))
    assert np.all(np.diff(errs) <= 1e-12) and errs[-1] < 1e-8


def test_poly_features_shapes_and_values():
    M = np.array([[0.5, 0.5], [0.7, 0.3]])
    assert poly_features(M, degree=4).shape == (2, 15)
    F1 = poly_features(M, degree=1)
    assert np.allclose(F1[0], [1.0, 0.0, 0.0])          # centred at 0.5: only the constant survives
    assert np.allclose(F1[1], [1.0, 0.2, -0.2])
    assert poly_features_3d(np.array([[0.5, 0.5, 0.12]]), degree=3).shape == (1, 20)
    assert np.allclose(poly_features_3d(np.array([[0.5, 0.5, 0.12]]), degree=3)[0, 1:], 0.0)


def test_fit_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(50, 4)); W_true = rng.normal(size=(4, 3))
    W = fit_ridge(F, F @ W_true, lam=1e-12)
    assert W.shape == (4, 3) and np.allclose(W, W_true, atol=1e-6)
