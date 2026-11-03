"""Volume-weighted POD, error measures and KPIs."""
import numpy as np

from ladle_twin.metrics import kpi_masks, kpi_r2, kpis, knee, r2, rel_error, rel_l2, rms_kelvin
from ladle_twin.pod import build_basis, coeffs, leftover_energy, lift


def test_basis_is_orthonormal_in_weighted_inner_product(small_ladle):
    F, wg = small_ladle["F"][:40], small_ladle["wg"]
    Tmean, basis, energy = build_basis(F, wg, K=5)
    assert basis.shape == (5, F[0].size) and Tmean.shape == (F[0].size,)
    assert np.allclose((basis * wg[None, :]) @ basis.T, np.eye(5), atol=1e-8)
    assert np.all(np.diff(energy) >= -1e-12) and abs(energy[-1] - 1.0) < 1e-9


def test_coeffs_and_lift_roundtrip(small_ladle):
    F, wg, g = small_ladle["F"][:40], small_ladle["wg"], small_ladle["g"]
    Tmean, basis, _ = build_basis(F, wg, K=5)
    a = np.array([[1.0, -2.0, 0.5, 0.0, 3.0], [0.0, 1.0, 1.0, 1.0, 1.0]])
    fields = lift(a, Tmean, basis, g["nz"], g["nr"])
    assert fields.shape == (2, g["nz"], g["nr"])
    assert np.allclose(coeffs(fields, Tmean, basis, wg), a, atol=1e-8)
    assert np.allclose(leftover_energy(fields, Tmean, basis, wg), 0.0, atol=1e-8)


def test_leftover_energy_is_one_for_orthogonal_structure(small_ladle):
    F, wg, g = small_ladle["F"][:40], small_ladle["wg"], small_ladle["g"]
    Tmean, basis, _ = build_basis(F, wg, K=3)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, F[0].size))
    X -= (((X * wg[None, :]) @ basis.T) @ basis)          # remove the part the basis explains
    assert np.allclose(leftover_energy(Tmean + X, Tmean, basis, wg), 1.0, atol=1e-8)


def test_rel_l2_and_rms_known_cases():
    wg = np.ones(4)
    Tmean = np.zeros(4)
    Ftrue = np.array([[2.0, 0.0, 0.0, 0.0]])
    Fhat = np.array([[2.0, 1.0, 0.0, 0.0]])
    assert np.allclose(rel_l2(Fhat, Ftrue, Tmean, wg), 0.5)        # |err| 1 over |variation| 2
    assert np.allclose(rel_l2(Ftrue, Ftrue, Tmean, wg), 0.0)
    assert np.allclose(rms_kelvin(Ftrue + 3.0, Ftrue, wg), 3.0)


def test_rel_error_and_knee():
    assert rel_error(np.array([3.0, 4.0]), np.array([3.0, 4.0])) == 0.0
    assert np.isclose(rel_error(np.array([3.0, 4.0]), np.array([3.0, 0.0])), 0.8)
    ts = np.linspace(0, 1, 5)
    assert knee([0.01, 0.02, 0.06, 0.5, 2.0], ts) == 0.5
    assert knee([0.01, 0.02, 0.03, 0.04, 0.05], ts) is None


def test_kpis_known_case(small_ladle):
    g, wg = small_ladle["g"], small_ladle["wg"]
    wall_band, top, bot = kpi_masks(g)
    assert wall_band.sum() > 0 and top.sum() > 0 and bot.sum() > 0 and not np.any(top & bot)
    Fs = np.full((1, g["nz"], g["nr"]), 1873.0 + 5.0)
    k = kpis(Fs, wg, wall_band, top, bot)
    assert np.allclose(k["superheat"], 5.0) and np.allclose(k["wall_load"], 1878.0) and np.allclose(k["strat"], 0.0)


def test_r2_and_kpi_r2(small_ladle):
    t = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(t, t) == 1.0 and np.isclose(r2(t, np.full(4, t.mean())), 0.0)
    g, wg, F = small_ladle["g"], small_ladle["wg"], small_ladle["F"][:5]
    out = kpi_r2(F, F, wg, *kpi_masks(g))
    assert set(out) == {"superheat", "wall_load", "strat"} and all(np.isclose(v, 1.0) for v in out.values())
