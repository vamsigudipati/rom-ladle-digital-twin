"""The package functions behave exactly like the notebooks' inline definitions.

The definitions are pulled out of the .ipynb JSON with `ast` and executed on
their own (no notebook cell is run as a whole; nothing is written or plotted),
then compared with the package on tiny inputs.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from conftest import notebook_functions
from ladle_twin import metrics, plate, pod, twin

NB1 = "01_pod_rom_intrusive_and_data_driven.ipynb"
NB2 = "02_ladle_hybrid_digital_twin.ipynb"
NB3 = "03_sensors_and_guard.ipynb"


def test_stage1_solver_and_roms_match_notebook():
    ns = dict(np=np, sp=sp, splu=splu)
    f = notebook_functions(NB1, ["Problem", "build_problem", "as_grid", "build_rom", "poly_features", "fit_ridge"], ns)
    p_nb, p_pk = f["build_problem"](8, n_steps=10), plate.build_problem(8, n_steps=10)
    mu = (0.45, 0.6)
    assert np.array_equal(p_nb.solve_fom(mu), p_pk.solve_fom(mu))
    assert np.array_equal(f["as_grid"](p_nb, p_nb.solve_fom(mu)), plate.as_grid(p_pk, p_pk.solve_fom(mu)))
    basis, _ = plate.snapshot_basis(p_pk, np.array([[0.4, 0.4], [0.6, 0.5]]), every=2)
    _, rom_nb, Ar_nb = f["build_rom"](p_nb, basis, 4)
    _, rom_pk, Ar_pk = plate.build_rom(p_pk, basis, 4)
    assert np.array_equal(Ar_nb, Ar_pk) and np.array_equal(rom_nb(mu), rom_pk(mu))
    M = np.array([[0.3, 0.7], [0.55, 0.45]])
    assert np.array_equal(f["poly_features"](M, degree=4), plate.poly_features(M, degree=4))
    A = np.arange(30, dtype=float).reshape(2, 15)
    assert np.array_equal(f["fit_ridge"](plate.poly_features(M), A), plate.fit_ridge(plate.poly_features(M), A))


def test_stage1_experiment_helpers_match_notebook():
    ts = np.linspace(0, 1, 15)
    ns = dict(np=np, ts=ts)
    f = notebook_functions(NB1, ["ex_rel", "ex_knee", "ex_poly3"], ns)
    a, b = np.array([1.0, 2.0, 2.0]), np.array([1.0, 1.0, 2.0])
    assert f["ex_rel"](a, b) == metrics.rel_error(a, b)
    errs = [0.01] * 6 + [0.2] * 9
    assert f["ex_knee"](errs) == metrics.knee(errs, ts) and f["ex_knee"]([0.0]) is None is metrics.knee([0.0], ts)
    M = np.array([[0.4, 0.6, 0.09], [0.5, 0.5, 0.18]])
    assert np.array_equal(f["ex_poly3"](M), plate.poly_features_3d(M))


def test_stage2_pod_metrics_and_guard_match_notebook(small_ladle):
    g, F, wg, lat = small_ladle["g"], small_ladle["F"], small_ladle["wg"], small_ladle["lat"]
    nz, nr = g["nz"], g["nr"]
    Tmean, basis, _ = pod.build_basis(F[:40], wg, K=8)
    Rg, Zg, R, H = g["Rg"], g["Zg"], g["R"], g["H"]
    ns = dict(np=np, Tmean=Tmean, basis=basis, wg=wg, nz=nz, nr=nr, r=g["r"], z=g["z"],
              wall_band=(Rg > 0.85*R).ravel(), top=(Zg > 0.8*H).ravel(), bot=(Zg < 0.2*H).ravel())
    f = notebook_functions(NB2, ["coeffs", "relL2", "leftover_energy", "kpis", "cell"], ns)
    Fh, Ft = F[40:43], F[:3]
    assert np.array_equal(f["coeffs"](Ft), pod.coeffs(Ft, Tmean, basis, wg))
    assert np.array_equal(f["relL2"](Fh, Ft), metrics.rel_l2(Fh, Ft, Tmean, wg))
    assert np.array_equal(f["leftover_energy"](Fh), pod.leftover_energy(Fh, Tmean, basis, wg))
    k_nb, k_pk = f["kpis"](Ft), metrics.kpis(Ft, wg, *metrics.kpi_masks(g))
    assert all(np.array_equal(k_nb[k], k_pk[k]) for k in k_nb)
    assert f["cell"](0.1 * R, 0.5 * H) == twin.cell_index(g, 0.1 * R, 0.5 * H)
    probes = twin.two_lance_probes(g)
    assert probes == [f["cell"](rr*R, zz*H) for rr in (0.1, 0.9) for zz in np.linspace(0.06, 0.94, 8)]


def test_stage3_scores_match_notebook(small_ladle):
    g, F, wg = small_ladle["g"], small_ladle["F"], small_ladle["wg"]
    Tmean, basis, _ = pod.build_basis(F[:40], wg, K=8)
    probes16 = twin.two_lance_probes(g)
    PhiP = basis[:, probes16].T; muP = Tmean[probes16]
    ns = dict(np=np, Tmean=Tmean, basis=basis, wg=wg, PhiP=PhiP, muP=muP, probes16=probes16)
    f = notebook_functions(NB3, ["field_score", "sensor_score", "rmsK", "probe"], ns)
    Fh, Ft = F[40:43], F[:3]
    assert np.array_equal(f["field_score"](Fh), pod.leftover_energy(Fh, Tmean, basis, wg))
    assert np.array_equal(f["sensor_score"](f["probe"](Fh)), twin.sensor_score(Fh.reshape(3, -1)[:, probes16], PhiP, muP))
    assert np.array_equal(f["rmsK"](Fh, Ft), metrics.rms_kelvin(Fh, Ft, wg))
