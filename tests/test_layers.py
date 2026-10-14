"""Stacks of several phase layers with moving interfaces.

Exact reference: a reaction layer beta between alpha and gamma, all three with
constant D and fixed interface compositions, grows with both interfaces at
S_i = k_i sqrt(t). Derived here by Zener's similarity argument (J. Appl. Phys. 20
(1949) 950) applied to each interface:

    alpha: x = xa_inf + (a1 - xa_inf) erfc(-z / 2 sqrt(Da t)) / erfc(-k1 / 2 sqrt(Da))
    beta:  x = A + B erf(z / 2 sqrt(Db t)),  fixed by x(S1) = b1 and x(S2) = b2
    gamma: x = xg_inf + (g2 - xg_inf) erfc(z / 2 sqrt(Dg t)) / erfc(k2 / 2 sqrt(Dg))

with the flux balance w (x_right - x_left) = J_right - J_left (Zener eq. 8) at both
interfaces giving two equations for k1 and k2.
"""

import re

import numpy as np
import pytest
from scipy.optimize import fsolve
from scipy.special import erf, erfc, erfcx

from janus_kinetics import FixedInterface, PhaseDissolvedError, solve_planar, uniform_faces
from janus_kinetics.moving_boundary import Layer, solve_layers

DA, DB, DG = 1e-15, 5e-15, 2e-15
XA_INF, A1, B1, B2, G2, XG_INF = 0.95, 0.80, 0.62, 0.45, 0.25, 0.05


def growth_constants():
    def flux_terms(k):
        k1, k2 = k
        u1, u2 = k1 / (2 * np.sqrt(DB)), k2 / (2 * np.sqrt(DB))
        b = (B2 - B1) / (erf(u2) - erf(u1))
        j_alpha = -(A1 - XA_INF) * np.sqrt(DA / np.pi) / erfcx(-k1 / (2 * np.sqrt(DA)))  # all fluxes times sqrt(t)
        j_gamma = (G2 - XG_INF) * np.sqrt(DG / np.pi) / erfcx(k2 / (2 * np.sqrt(DG)))
        j_beta1 = -b * np.sqrt(DB / np.pi) * np.exp(-u1 * u1)
        j_beta2 = -b * np.sqrt(DB / np.pi) * np.exp(-u2 * u2)
        return [k1 / 2 * (B1 - A1) - (j_beta1 - j_alpha), k2 / 2 * (G2 - B2) - (j_gamma - j_beta2)]

    k = fsolve(flux_terms, [-1e-8, 1e-8], xtol=1e-14)
    assert np.max(np.abs(flux_terms(k))) < 1e-20
    return k


def exact_profile(z, t, k):
    k1, k2 = k
    s1, s2 = k1 * np.sqrt(t), k2 * np.sqrt(t)
    u1, u2 = k1 / (2 * np.sqrt(DB)), k2 / (2 * np.sqrt(DB))
    b = (B2 - B1) / (erf(u2) - erf(u1))
    a = B1 - b * erf(u1)
    x = np.where(z < s1, XA_INF + (A1 - XA_INF) * erfc(-z / (2 * np.sqrt(DA * t))) / erfc(-k1 / (2 * np.sqrt(DA))),
                 np.where(z < s2, a + b * erf(z / (2 * np.sqrt(DB * t))),
                          XG_INF + (G2 - XG_INF) * erfc(z / (2 * np.sqrt(DG * t))) / erfc(k2 / (2 * np.sqrt(DG)))))
    return x


@pytest.fixture(scope="module")
def three_layer_run():
    k = growth_constants()
    t0, t1 = 2e3, 5e4
    half = 12 * np.sqrt(max(DA, DB, DG) * t1)
    s1, s2 = half + k[0] * np.sqrt(t0), half + k[1] * np.sqrt(t0)
    n = (300, 40, 300)
    bounds = [0.0, s1, s2, 2 * half]
    layers = []
    for j, d in enumerate((DA, DB, DG)):
        zc = bounds[j] + (bounds[j + 1] - bounds[j]) * (np.arange(n[j]) + 0.5) / n[j]
        layers.append(Layer(exact_profile(zc - half, t0, k), d, name=("alpha", "beta", "gamma")[j]))
    conds = [FixedInterface([A1], [B1]), FixedInterface([B2], [G2])]
    res = solve_layers(bounds, layers, conds, t1 - t0, (t1 - t0) / 400, save_times=[(t1 - t0) / 2])
    return res, k, half, t0, t1


def test_both_interfaces_follow_the_similarity_solution(three_layer_run):
    res, k, half, t0, t1 = three_layer_run
    for i in range(2):
        assert res.interfaces[-1, i] - half == pytest.approx(k[i] * np.sqrt(t1), rel=1e-2)
    beta = res.thickness()[:, 1]
    assert beta[-1] / beta[0] == pytest.approx(np.sqrt(t1 / t0), rel=1e-2)  # the layer grew five-fold


def test_profile_matches_through_the_stack(three_layer_run):
    res, k, half, _, t1 = three_layer_run
    z, x = res.profile()
    assert np.max(np.abs(x[:, 0] - exact_profile(z - half, t1, k))) < 1e-2


def test_stack_conserves_solute(three_layer_run):
    inv = three_layer_run[0].inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-12 * inv[0, 0]


def _film_on_substrate():
    """A 0.5 um alpha film on a no-flux back face, eaten by a beta layer that grows into it while giving B
    to the gamma substrate. With 5, 1 and 0.2 s steps the film is consumed near 85, 85 and 84.2 s, and with
    adaptive steps (tolerance 1e-4 and 1e-5) near 84.2 and 84.0 s: a real end of the film, not a step artifact."""
    layers = [Layer(np.full(20, 0.62), 1e-15, "alpha"), Layer(np.linspace(0.60, 0.50, 20), 1e-15, "beta"),
              Layer(np.full(100, 0.05), 1e-14, "gamma")]
    conds = [FixedInterface([0.62], [0.60]), FixedInterface([0.50], [0.30])]
    return [0.0, 0.5e-6, 1e-6, 51e-6], layers, conds


def test_consumed_layer_returns_partial_results():
    """A consumed film stops the run with an error carrying the history."""
    bounds, layers, conds = _film_on_substrate()
    with pytest.raises(PhaseDissolvedError, match="alpha") as err:
        solve_layers(bounds, layers, conds, 2000.0, 5.0, save_times=[10.0, 100.0])
    assert list(err.value.result.times) == [0.0, 10.0]
    assert err.value.result.thickness().shape == (2, 3)


def test_adaptive_steps_consume_the_film_when_it_is_used_up():
    bounds, layers, conds = _film_on_substrate()
    with pytest.raises(PhaseDissolvedError, match="alpha") as err:
        solve_layers(bounds, layers, conds, 2000.0, 100.0, save_times=[10.0, 100.0], tolerance=1e-4)
    t = float(re.search(r"t = ([0-9.eE+-]+) s", str(err.value)).group(1))
    assert 83.0 < t < 86.0
    assert err.value.result.thickness()[1, 0] == pytest.approx(438.8e-9, rel=2e-3)


def _thinning_beta(**kw):
    """A 2 nm-cell beta layer (200 nm, D = 1e-17) between fast alpha and gamma with fixed interface
    compositions. The early fluxes thin it to 1.565 nm at 15 s, after which it grows again (2.252 nm at
    300 s); 0.1 s and 0.02 s steps give the same numbers."""
    layers = [Layer(np.full(50, 0.95), 1e-15, "alpha"), Layer(np.full(5, 0.55), 1e-17, "beta"),
              Layer(np.full(50, 0.05), 1e-14, "gamma")]
    conds = [FixedInterface([0.62], [0.60]), FixedInterface([0.50], [0.30])]
    return solve_layers([0.0, 1e-4, 1.002e-4, 2e-4], layers, conds, 300.0, save_times=[15.0], **kw)


@pytest.mark.slow
def test_adaptive_steps_follow_a_layer_through_its_thinnest_point():
    """Steps of 0.5 s or more carry the beta interface past its neighbour near 14 s and report the layer
    consumed; it is not. Adaptive steps, measuring each interface move against the thinner neighbouring
    layer, follow it down to 1.565 nm and back (tolerance 1e-3 and 1e-4 agree within 1e-4)."""
    res = _thinning_beta(dt=50.0, tolerance=1e-3)
    np.testing.assert_allclose(res.thickness()[1:, 1], [1.565e-9, 2.252e-9], rtol=2e-3)
    with pytest.raises(PhaseDissolvedError, match="beta"):
        _thinning_beta(dt=1.0)



@pytest.mark.slow
def test_adaptive_steps_retry_an_overlong_first_step():
    """The first adaptive step has no error estimate; from a largest step of 2e4 s it would carry the thin
    beta interface through its neighbour, and is retried shorter instead of reporting the layer consumed."""
    res = _thinning_beta(dt=2e4, tolerance=1e-3)
    np.testing.assert_allclose(res.thickness()[1:, 1], [1.565e-9, 2.252e-9], rtol=2e-3)


def test_adaptive_steps_follow_compositions_inside_a_layer():
    """A composition step buried in alpha, 0.5 mm from an interface at rest: the interface does not move, so
    the step must be set by the compositions alone. Within 1e-3 of the planar solver (0.36e-3 measured); four
    equal steps of the same largest size miss by 6e-3."""
    n, saves = 200, [10.0, 100.0, 1e3]
    z = (np.arange(n) + 0.5) / n * 1e-3
    xa = np.where(z < 0.5e-3, 0.95, 0.80)
    res = solve_layers([0.0, 1e-3, 2e-3], [Layer(xa, 1e-14), Layer(np.full(50, 0.60), 1e-14)],
                       [FixedInterface([0.80], [0.60])], 1e4, 2.5e3, save_times=saves, tolerance=1e-4)
    ref = solve_planar(uniform_faces(1e-3, n), xa[:, None], 1e-14, 1e4, 0.5, save_times=saves)
    assert np.max(np.abs(res.interfaces - 1e-3)) < 1e-15
    assert np.max(np.abs(res.x[0][1:] - ref.x[1:])) < 1e-3

def test_adaptive_steps_follow_the_similarity_solution():
    """The three-layer case of three_layer_run (from t0 = 2000 s) with tolerance 1e-4: both interfaces within
    0.6 % of the beta thickness of Zener's similarity positions (0.42 % measured; 800 equal steps give 0.41 %,
    200 give 0.65 %), with the inventory kept."""
    k = growth_constants()
    t0, t1 = 2e3, 5e4
    half = 12 * np.sqrt(max(DA, DB, DG) * t1)
    s1, s2 = half + k[0] * np.sqrt(t0), half + k[1] * np.sqrt(t0)
    n = (300, 40, 300)
    bounds = [0.0, s1, s2, 2 * half]
    layers = []
    for j, d in enumerate((DA, DB, DG)):
        zc = bounds[j] + (bounds[j + 1] - bounds[j]) * (np.arange(n[j]) + 0.5) / n[j]
        layers.append(Layer(exact_profile(zc - half, t0, k), d))
    conds = [FixedInterface([A1], [B1]), FixedInterface([B2], [G2])]
    saves = [(t1 - t0) / 10, 3 * t0]
    res = solve_layers(bounds, layers, conds, t1 - t0, (t1 - t0) / 4, save_times=saves, tolerance=1e-4)
    np.testing.assert_array_equal(res.times, [0.0, *saves, t1 - t0])
    for i, t in enumerate(res.times[1:], 1):
        beta = (k[1] - k[0]) * np.sqrt(t + t0)
        assert np.max(np.abs(res.interfaces[i] - half - k * np.sqrt(t + t0))) < 6e-3 * beta
    inv = res.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-12 * inv[0, 0]


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="positions"):
        solve_layers([0.0, 1.0], [Layer(np.zeros(3), 1.0), Layer(np.zeros(3), 1.0)], [FixedInterface([0.1], [0.2])], 1.0, 0.1)
