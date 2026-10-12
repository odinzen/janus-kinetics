"""Two compound layers with interfacial reaction barriers, against Gosele and Tu, J. Appl. Phys. 53 (1982) 3252:
eqs. (6)-(13) for one layer, (15)-(29) for two, and the critical thickness of eq. (34). The paper gives no
numerical example, so the checks are its own limits and criteria."""

import numpy as np
import pytest
from scipy.optimize import fsolve

from janus_kinetics.layer_competition import (
    CompoundLayer,
    composition_factors,
    critical_thickness,
    growth_window,
    solve_two_layers,
)
from janus_kinetics.oxidation import oxide_thickness

OMEGA = 1.5e-29
ALPHA, DELTA = np.inf, 0.0  # pure A and pure B ends (section IV)
# A2B (beta) with a high reaction barrier and AB (gamma), the first to form
A2B = CompoundLayer(2.0, 5e28, 1e-17, (2e-10, 2e-10))
AB = CompoundLayer(1.0, 5e28, 1e-17, (1e-6, 1e-6))


def test_growth_window_is_the_ratio_of_composition_factors():
    """r_1 = G_beta,gamma / G_beta and r_2 = G_gamma / G_gamma,beta (eqs. 25-28), r_2 > r_1 (eq. 29); for AB and A2B
    between pure A and B they are 2/3 and 4/3."""
    for a, b, g, d in ((np.inf, 2.0, 1.0, 0.0), (5.0, 3.0, 1.5, 0.5), (1.2, 1.0, 0.4, 0.1)):
        g_b, g_g, g_bg, g_gb = composition_factors(a, b, g, d, OMEGA)
        r1, r2 = growth_window(a, b, g, d)
        assert r1 == pytest.approx(g_bg / g_b, rel=1e-14)
        assert r2 == pytest.approx(g_g / g_gb, rel=1e-14)
        assert r2 > r1
    assert growth_window(ALPHA, 2.0, 1.0, DELTA) == pytest.approx((2 / 3, 4 / 3), rel=1e-14)
    assert A2B.barrier == pytest.approx(1e-10, rel=1e-14)  # eq. 7
    assert AB.changeover == pytest.approx(1e-17 / 5e-7, rel=1e-14)  # eq. 11


def test_one_layer_is_linear_then_parabolic_as_deal_and_grove():
    """With the second compound unable to form, the first grows by eq. (9), dx/dt = G Delta C kappa / (1 + x kappa / D),
    which is Deal and Grove's law with A = 2 D / kappa and B = 2 G Delta C D: linear below x* (eq. 13a) and parabolic
    above it (eq. 13b)."""
    blocked = CompoundLayer(2.0, 5e28, 1e-17, (1e-40, 1e-40))
    slow = CompoundLayer(1.0, 5e28, 1e-17, (1e-9, 1e-9))
    g_g = composition_factors(ALPHA, 2.0, 1.0, DELTA, OMEGA)[1]
    t = np.logspace(-1, 6, 30)
    res = solve_two_layers(blocked, slow, ALPHA, DELTA, OMEGA, t[-1], save_times=t)
    assert np.all(res.beta == 0)
    a, b = 2 * slow.diffusivity / slow.barrier, 2 * g_g * slow.delta_c * slow.diffusivity
    np.testing.assert_allclose(res.gamma, oxide_thickness(res.times, a, b), rtol=1e-6)
    assert slow.changeover == pytest.approx(20e-9, rel=1e-12)


def test_second_layer_appears_at_the_critical_thickness():
    """The interface-controlled A2B layer cannot grow beside AB until AB is thick enough that r = j_beta / j_gamma
    reaches r_1. Eq. (34) gives 66.67 nm, taking the AB layer as diffusion controlled; with A2B at zero thickness the
    exact onset is that less x*_AB = 0.02 nm. A2B first appears between AB thicknesses bracketing 66.647 nm."""
    xc = critical_thickness(A2B, AB, ALPHA, DELTA)
    assert xc == pytest.approx(2 / 3 * 1e-17 / 1e-10, rel=1e-12)
    t = np.linspace(50.0, 70.0, 801)
    res = solve_two_layers(A2B, AB, ALPHA, DELTA, OMEGA, t[-1], save_times=t)
    first = np.flatnonzero(res.beta > 0)[0]
    assert res.gamma[first - 1] <= xc - AB.changeover <= res.gamma[first]
    assert res.gamma[first] - res.gamma[first - 1] < 0.1e-9


def test_both_layers_settle_on_the_diffusion_controlled_ratio():
    """Once both layers are far thicker than their changeover thicknesses they grow as sqrt(t) at the ratio of
    multilayer diffusion-controlled growth (section II B, Fig. 4), found here from eqs. (15)-(16) with j = Delta C D / x
    and x = k sqrt(t): 0.9428 for this couple. At 1e7 s (about 20 um) the ratio is within 0.6 %, of order x*/x."""
    g_b, g_g, g_bg, g_gb = composition_factors(ALPHA, 2.0, 1.0, DELTA, OMEGA)
    ab_, ag_ = A2B.delta_c * A2B.diffusivity, AB.delta_c * AB.diffusivity
    k = fsolve(lambda k: [k[0] / 2 - (g_b * ab_ / k[0] - g_bg * ag_ / k[1]),
                          k[1] / 2 - (g_g * ag_ / k[1] - g_gb * ab_ / k[0])], [1e-8, 1e-8], xtol=1e-14)
    res = solve_two_layers(A2B, AB, ALPHA, DELTA, OMEGA, 1e7, save_times=[1e6])
    late, later = res.gamma / res.beta
    assert abs(later / (k[1] / k[0]) - 1) < 0.006
    assert abs(later / (k[1] / k[0]) - 1) < abs(late / (k[1] / k[0]) - 1)


@pytest.mark.parametrize("kappa_beta,both_grow", [(5e-7, True), (2e-7, False)])
def test_interface_controlled_layers_follow_the_flux_ratio(kappa_beta, both_grow):
    """Both layers far below their changeover thicknesses: r = Delta C_beta kappa_beta / Delta C_gamma kappa_gamma is
    constant (eq. 32). Inside (2/3, 4/3) both grow linearly (Fig. 5 case b); below 2/3 the A2B layer shrinks away and
    AB grows (case c)."""
    fast = 1e-6  # m^2/s: changeover thicknesses of metres against micrometre layers, so the reactions control
    beta = CompoundLayer(2.0, 5e28, fast, (2 * kappa_beta, 2 * kappa_beta))
    gamma = CompoundLayer(1.0, 5e28, fast, (1e-6, 1e-6))
    g_b, g_g, g_bg, g_gb = composition_factors(ALPHA, 2.0, 1.0, DELTA, OMEGA)
    jb, jg = 5e28 * kappa_beta, 5e28 * 5e-7
    t = np.array([1.0, 2.0, 4.0])
    res = solve_two_layers(beta, gamma, ALPHA, DELTA, OMEGA, 4.0, initial=(5e-9, 5e-9), save_times=t)
    if both_grow:
        np.testing.assert_allclose(res.beta, 5e-9 + (g_b * jb - g_bg * jg) * res.times, rtol=1e-5)
        np.testing.assert_allclose(res.gamma, 5e-9 + (g_g * jg - g_gb * jb) * res.times, rtol=1e-5)
    else:
        assert res.beta[-1] == 0.0 and res.gamma[-1] > res.gamma[1]


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="alpha > beta"):
        growth_window(2.0, 1.0, 1.5, 0.0)
    with pytest.raises(ValueError, match="negative"):
        solve_two_layers(A2B, AB, ALPHA, DELTA, OMEGA, 1.0, initial=(-1e-9, 0.0))
