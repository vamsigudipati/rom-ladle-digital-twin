"""The analytic ladle generator, sensor placement and gappy-POD inverse."""
import numpy as np

from ladle_twin.ladle import (design_numbers, draw_operating_point, gappy_reconstruct, generate,
                              geom, place_sensors, temperature, volume_weights)


def test_geom_shapes():
    g = geom(nr=8, nz=12)
    assert g["Rg"].shape == (12, 8) and g["r"][-1] == g["R"] and g["z"][0] == 0.0


def test_temperature_is_bath_plus_patterns():
    g = geom(nr=8, nz=12)
    lat = (5e-4, -100.0, 0.5, 18.0, 0.0, 0.93)
    T = temperature(g, lat)
    assert T.shape == (12, 8) and np.all(np.isfinite(T))
    assert np.all(T <= 1873.0 + 40.0) and T.max() > 1873.0      # stratification heats the bottom
    assert np.abs(temperature(g, lat, "double") - T).max() > 1.0  # the abnormal regime adds structure


def test_draw_operating_point_respects_clips():
    rng = np.random.default_rng(1)
    pts = np.array([draw_operating_point(rng, t) for t in range(200)])
    lo = [2e-4, -160, 0.20, 2, -18, 0.86]; hi = [1.2e-3, -40, 0.90, 32, 18, 0.98]
    assert np.all(pts >= np.array(lo) - 1e-12) and np.all(pts <= np.array(hi) + 1e-12)


def test_generate_shapes_labels_and_determinism(small_ladle):
    F, reg, lat = small_ladle["F"], small_ladle["reg"], small_ladle["lat"]
    assert F.shape == (46, 96, 64) and lat.shape == (46, 6)
    assert list(reg[:40]) == ["single"] * 40 and list(reg[40:]) == ["double"] * 6
    _, F2, _, _ = generate(n_single=40, n_double=6, seed=0)
    assert np.array_equal(F, F2)


def test_volume_weights_positive_and_sum_to_volume():
    g = geom()
    wg = volume_weights(g)
    assert wg.shape == (g["nz"] * g["nr"],) and np.all(wg > 0)
    assert abs(wg.sum() / (np.pi * g["R"]**2 * g["H"]) - 1.0) < 0.05


def test_place_sensors_unique_and_inside_border(small_ladle):
    g, wg = small_ladle["g"], small_ladle["wg"]
    X = small_ladle["F"][:40].reshape(40, -1); sw = np.sqrt(wg)
    _, _, Vt = np.linalg.svd((X - X.mean(0)) * sw[None, :], full_matrices=False)
    basis = (Vt / sw[None, :])[:8]
    sel = place_sensors(basis, g, m=10, exclude_border=2)
    assert sel.shape == (10,) and len(set(sel.tolist())) == 10
    rr, zz = sel % g["nr"], sel // g["nr"]
    assert rr.min() >= 2 and rr.max() <= g["nr"] - 3 and zz.min() >= 2 and zz.max() <= g["nz"] - 3


def test_gappy_reconstruct_exact_for_in_span_fields():
    rng = np.random.default_rng(0)
    n, K = 30, 4
    basis = np.linalg.qr(rng.normal(size=(n, K)))[0].T      # K orthonormal rows
    Tmean = rng.normal(size=n)
    a = rng.normal(size=(5, K))
    fields = Tmean + a @ basis
    sel = np.array([0, 3, 7, 11, 19, 23])
    rec = gappy_reconstruct(basis, Tmean, sel, fields[:, sel])
    assert rec.shape == fields.shape and np.allclose(rec, fields, atol=1e-10)


def test_design_numbers_small():
    d = design_numbers(n_single=60, n_double=8, seed=0, K=6, m=8)
    cum = d["cum_energy"]
    assert np.all(np.diff(cum) >= -1e-12) and abs(cum[-1] - 1.0) < 1e-9
    assert 1 <= d["modes_for_99"] <= len(cum) and d["cond"] > 1.0
    assert 0.0 < d["in_dist_relL2"] < 1.0 and len(d["sel"]) == 8
