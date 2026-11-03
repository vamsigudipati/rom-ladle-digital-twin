"""Probe layout, sensor readings, physics core, residual basis, guards and the neural soft sensor."""
import numpy as np
import pytest

from ladle_twin.ladle import temperature
from ladle_twin.pod import build_basis
from ladle_twin.twin import (cell_index, gas_flow_input, make_soft_sensor, physics_core, read_sensors,
                             residual_basis, residual_coeffs, sensor_score, train_soft_sensor,
                             two_lance_probes)


def test_cell_index_and_probes(small_ladle):
    g = small_ladle["g"]
    assert cell_index(g, 0.0, 0.0) == 0
    assert cell_index(g, g["R"], g["H"]) == g["nz"] * g["nr"] - 1
    probes = two_lance_probes(g)
    assert len(probes) == 16 and len(set(probes)) == 16
    assert all(0 <= k < g["nz"] * g["nr"] for k in probes)


def test_read_sensors_shape_and_reproducibility(small_ladle):
    F, lat, g = small_ladle["F"], small_ladle["lat"], small_ladle["g"]
    probes = two_lance_probes(g); Q = gas_flow_input(lat); idx = np.arange(10)
    S1 = read_sensors(F, idx, probes, Q, np.random.default_rng(0))
    S2 = read_sensors(F, idx, probes, Q, np.random.default_rng(0))
    assert S1.shape == (10, 17) and np.array_equal(S1, S2)
    assert np.allclose(S1[:, -1], lat[idx, 0] / 5e-4)
    clean = F[idx].reshape(10, -1)[:, probes]
    assert np.abs(S1[:, :16] - clean).max() < 12.0           # 2 K noise, 6 sigma


def test_physics_core_uses_known_inputs_only(small_ladle):
    F, lat, g = small_ladle["F"], small_ladle["lat"], small_ladle["g"]
    nom = lat[:40].mean(0)
    core = physics_core(g, lat, nom, np.array([3, 7]))
    assert core.shape == (2, g["nz"] * g["nr"])
    p = nom.copy(); p[0], p[1], p[4] = lat[3, 0], lat[3, 1], 0.0
    assert np.allclose(core[0], temperature(g, tuple(p)).ravel())


def test_residual_basis_and_coefficients(small_ladle):
    F, lat, g, wg = small_ladle["F"], small_ladle["lat"], small_ladle["g"], small_ladle["wg"]
    nom = lat[:40].mean(0); idx = np.arange(40)
    Rtr = F[idx].reshape(40, -1) - physics_core(g, lat, nom, idx)
    Rmean, basisR = residual_basis(Rtr, wg, K=4)
    assert basisR.shape == (4, Rtr.shape[1]) and np.allclose((basisR * wg[None, :]) @ basisR.T, np.eye(4), atol=1e-8)
    assert residual_coeffs(Rtr, Rmean, basisR, wg).shape == (40, 4)


def test_sensor_score_zero_for_in_span_readings_and_bounded(small_ladle):
    F, wg, g = small_ladle["F"][:40], small_ladle["wg"], small_ladle["g"]
    Tmean, basis, _ = build_basis(F, wg, K=4)
    probes = two_lance_probes(g); PhiP = basis[:, probes].T; muP = Tmean[probes]
    a = np.array([[1.0, 2.0, -1.0, 0.5]])
    readings = (Tmean + a @ basis)[:, probes]
    assert np.allclose(sensor_score(readings, PhiP, muP), 0.0, atol=1e-10)
    noisy = readings + np.random.default_rng(0).normal(0, 50.0, readings.shape)
    s = sensor_score(noisy, PhiP, muP)
    assert s.shape == (1,) and 0.0 < s[0] <= 1.0


def test_soft_sensor_forward_pass_on_two_samples():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = make_soft_sensor(17, 64, 8)
    assert sum(p.numel() for p in net.parameters()) == 17 * 64 + 64 + 64 * 8 + 8
    x = torch.zeros(2, 17)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 8) and y.device.type == "cpu" and torch.isfinite(y).all()
    assert torch.allclose(y[0], y[1])                       # identical inputs, identical outputs


def test_train_soft_sensor_reduces_loss_and_calls_hook():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    X = torch.randn(32, 17); W = torch.randn(17, 8); Y = X @ W * 0.1
    net = make_soft_sensor(17, 16, 8)
    before = torch.nn.functional.mse_loss(net(X), Y).item()
    calls = []
    train_soft_sensor(net, X, Y, steps=30, lr=1e-2, hook=lambda m: calls.append(1))
    after = torch.nn.functional.mse_loss(net(X), Y).item()
    assert len(calls) == 30 and after < before
