"""Electromigration stress in confined lines, against Korhonen, Borgesen, Tu and Li, J. Appl. Phys. 73 (1993)
3790 (eqs. 8-16 and Figs. 8-9), and the threshold of Blech, J. Appl. Phys. 47 (1976) 1203 (eqs. 1-6).

Korhonen et al.'s solutions are exact for constant kappa; the numerical solver is checked against each, and
eq. (9) against their own fit to Hemmert and Costa's measurements, digitized from their Fig. 9."""

import numpy as np
import pytest

from janus_kinetics.electromigration import (
    ELEMENTARY_CHARGE,
    K_B,
    LineEnd,
    blech_product,
    drift_velocity,
    finite_line_stress,
    semi_infinite_stress,
    solve_line_stress,
    steady_stress,
    stress_diffusivity,
    void_line_stress,
    wind_stress_gradient,
)

G = 16.5e12  # Pa/m, 16.5 MPa/um (Korhonen et al. p. 3793, Fig. 9)
KAPPA = 2.6e-12 / 3600  # m^2/s, 2.6 um^2/h (Fig. 9)
L = 50e-6


@pytest.mark.parametrize("steps,bound", [(1600, 2e-4), (6400, 5e-5)])
def test_line_blocked_at_both_ends_follows_eq_13(steps, bound):
    """At kappa t / L^2 = 0.01, 0.1 and 0.2 the numerical stress converges on eq. (13) at first order in the
    step: 1.8e-4, 9.4e-5 and 7.0e-5 of G L with 1600 steps, a quarter of that with 6400."""
    t_end = 0.2 * L**2 / KAPPA
    saves = [0.01 * L**2 / KAPPA, 0.1 * L**2 / KAPPA]
    res = solve_line_stress(L, 200, G, KAPPA, t_end, t_end / steps, save_times=saves)
    for i, t in enumerate(res.times[1:], 1):
        assert np.abs(res.stress[i] - finite_line_stress(res.x, t, L, KAPPA, G)).max() < bound * G * L


def test_eq_13_and_eq_15_start_from_zero_stress():
    """The series must cancel the leading terms at t = 0, which also fixes eq. (15)'s printed -G xi as -G L xi."""
    x = np.linspace(0.0, L, 21)
    assert np.abs(finite_line_stress(x, 0.0, L, KAPPA, G, terms=4000)).max() < 3e-5 * G * L
    assert np.abs(void_line_stress(x, 0.0, L, KAPPA, G, L / 5, terms=4000)).max() < 6e-5 * G * L


def test_semi_infinite_line_follows_eqs_9_and_10():
    t = 0.05 * L**2 / KAPPA
    res = solve_line_stress(20 * L, 2000, G, KAPPA, t, t / 400)
    near = res.x < 3 * L
    assert np.abs(res.stress[-1][near] - semi_infinite_stress(res.x[near], t, KAPPA, G)).max() < 5e-4 * G * np.sqrt(KAPPA * t)
    assert semi_infinite_stress(0.0, t, KAPPA, G) == pytest.approx(2 * G * np.sqrt(KAPPA * t / np.pi), rel=1e-14)


def test_void_end_follows_eq_15():
    """A void at x = 0 (alpha = L/w = 5, their Fig. 6) and a block at x = L: the solver follows eqs. (15)-(16)
    within 3e-4 of G L at kappa t/L^2 = 0.1 and 1 (2.8e-4 and 9.7e-5 measured) and 3, and the series relaxes to
    -G x, compressive except at the void. The slowest mode, beta_0 = 1.31 for alpha = 5, decays as
    exp(-1.72 kappa t / L^2), so at 3 the line is still 0.4 % of G L from it."""
    w = L / 5
    t_end = 3.0 * L**2 / KAPPA
    saves = [0.1 * L**2 / KAPPA, 1.0 * L**2 / KAPPA]
    res = solve_line_stress(L, 400, G, KAPPA, t_end, t_end / 6000, left=LineEnd("void", width=w), save_times=saves)
    for i, t in enumerate(res.times[1:], 1):
        assert np.abs(res.stress[i] - void_line_stress(res.x, t, L, KAPPA, G, w)).max() < 3e-4 * G * L
    late = void_line_stress(res.x, 12 * L**2 / KAPPA, L, KAPPA, G, w)
    np.testing.assert_allclose(late, -G * res.x, rtol=0, atol=1e-6 * G * L)


def test_void_at_the_right_end_mirrors_the_left():
    w = L / 5
    t_end = 0.1 * L**2 / KAPPA
    left = solve_line_stress(L, 200, G, KAPPA, t_end, t_end / 400, left=LineEnd("void", width=w))
    right = solve_line_stress(L, 200, -G, KAPPA, t_end, t_end / 400, right=LineEnd("void", width=w))
    np.testing.assert_allclose(right.stress[:, ::-1], left.stress, rtol=0, atol=1e-10 * G * L)


def test_steady_state_is_blechs_and_its_end_difference_is_his_threshold():
    """A blocked line settles on eq. (12), whose end-to-end difference G L is Blech's Delta sigma: at the
    threshold of his eq. (6) a line of length L carries exactly the current that gives G."""
    res = solve_line_stress(L, 200, G, KAPPA, 3 * L**2 / KAPPA, L**2 / KAPPA / 100)
    np.testing.assert_allclose(res.stress[-1], steady_stress(res.x, L, G), rtol=0, atol=1e-9 * G * L)
    rho, z, omega = 4.6e-8, 3.5, 1.66e-29
    j = G * omega / (rho * z * ELEMENTARY_CHARGE)
    assert wind_stress_gradient(rho, j, -z, omega) == pytest.approx(G, rel=1e-14)
    assert blech_product(G * L, -z, rho, omega) / L == pytest.approx(j, rel=1e-14)


def test_reservoir_end_holds_its_stress():
    """A line fed from a stress-free pad at x = 0 and blocked at L settles on sigma = -G x."""
    res = solve_line_stress(L, 200, G, KAPPA, 4 * L**2 / KAPPA, L**2 / KAPPA / 100, left=LineEnd("reservoir"))
    np.testing.assert_allclose(res.stress[-1], -G * res.x, rtol=0, atol=2e-3 * G * L)


def test_fit_of_fig_9_is_eq_9():
    """Their Fig. 9 fit (G = 16.5 MPa/um, kappa = 2.6 um^2/h, 270 h), digitized at 400 dpi with the axes taken
    from the tick marks, against eq. (9): within 0.006 in log10 stress from 1 to 48 um (the trace sits a nearly
    constant 0.003 low, the width of the printed line)."""
    digitized = {1: 2.6749, 3: 2.6454, 5: 2.6166, 8: 2.5694, 15: 2.4512, 25: 2.2672, 35: 2.0632, 45: 1.8356, 48: 1.7657}
    for x_um, log_stress in digitized.items():
        assert np.log10(semi_infinite_stress(x_um, 270.0, 2.6, 16.5)) == pytest.approx(log_stress, abs=0.006)


def test_diffusivity_inferred_from_the_fit_is_inconsistent_in_the_paper():
    """Korhonen et al. (p. 3796) take B = 50 GPa, for which B Omega / kT is about 120 at 200 C (p. 3792), and infer
    D_a = 3e-16 m^2/s from kappa = 2.6 um^2/h. That kappa gives D_a = 6e-18 m^2/s; 3e-16 follows from dividing by
    60 rather than 3600 seconds. The fit itself (Figs. 8-9, about 700 MPa after some 300 h) is consistent with
    2.6 um^2/h, while their 'about 200 MPa within an hour' (p. 3793) follows from the 3e-16. Recorded, not used."""
    omega = 1.66e-29  # Al
    t = 473.15
    factor = 50e9 * omega / (K_B * t)
    assert 115 < factor < 135
    d_a = KAPPA / factor
    assert 4e-18 < d_a < 8e-18
    assert 2.5e-16 < (2.6e-12 / 60) / factor < 4.5e-16
    assert stress_diffusivity(d_a, 50e9, omega, t) == pytest.approx(KAPPA, rel=1e-14)
    assert 450e6 < semi_infinite_stress(0.0, 300 * 3600, KAPPA, G) < 650e6
    assert semi_infinite_stress(0.0, 3600, 3e-16 * factor, G) == pytest.approx(200e6, rel=0.15)


def test_blech_threshold_from_his_own_numbers():
    """Blech (p. 1204): no drift below 1.1e5 A/cm^2 in a 115 um stripe, and j_c l = 1260 A/cm. His eq. (4) then
    gives no drift below the threshold and a drift linear in j above it with slope (D/kT) e Z* rho."""
    assert 1.1e5 * 115e-4 == pytest.approx(1260, rel=0.01)
    rho, z, omega, d, temp = 6.5e-8, 1.0, 1.66e-29, 1e-14, 623.15
    length = 115e-6
    stress = 1260 * 1e2 * abs(z) * ELEMENTARY_CHARGE * rho / omega  # Delta sigma that makes (jL)_c = 1260 A/cm
    assert blech_product(stress, z, rho, omega) == pytest.approx(1260e2, rel=1e-12)
    j_c = 1260e2 / length
    assert drift_velocity(0.99 * j_c, length, d, temp, z, rho, omega, stress) == 0.0
    slope = d / (K_B * temp) * ELEMENTARY_CHARGE * z * rho
    v1, v2 = (drift_velocity(f * j_c, length, d, temp, z, rho, omega, stress) for f in (2.0, 3.0))
    assert (v2 - v1) / j_c == pytest.approx(slope, rel=1e-12)
    assert v1 == pytest.approx(slope * j_c, rel=1e-12)


def test_yielding_front_follows_eq_11():
    """With ideal plasticity at 200 MPa the yielded zone at a blocked end spreads at v = G kappa / sigma_y and the
    stress ahead of it is sigma_y exp(-G x'/sigma_y): measured front speed within 0.3 % and tail within 1 % of
    sigma_y (the front moves a cell at a time, 0.5 um here)."""
    sy = 200e6
    times = np.array([400.0, 800.0, 1200.0]) * 3600
    res = solve_line_stress(600e-6, 1200, G, KAPPA, times[-1], times[-1] / 3000, save_times=times, yield_stress=sy)
    fronts = []
    for s in res.stress[1:]:
        yielded = np.flatnonzero(s >= sy * (1 - 1e-9))
        front = res.x[yielded[-1]] + 0.25e-6
        fronts.append(front)
        ahead = (res.x > front) & (res.x < front + 3 * sy / G)
        assert np.abs(s[ahead] - sy * np.exp(-(res.x[ahead] - front) * G / sy)).max() < 0.01 * sy
    np.testing.assert_allclose(np.diff(fronts) / np.diff(times), G * KAPPA / sy, rtol=3e-3)
    assert np.all(res.stress <= sy * (1 + 1e-12))


def test_stress_dependent_kappa_builds_tension_faster_than_compression():
    """With eq. (18), kappa = kappa_0 exp(Omega sigma / kT) at 200 C, tension speeds diffusion and compression slows
    it: in a line heading for +-600 MPa the tensile end runs 4.5 % ahead of the constant-kappa solution at kappa_0
    t / L^2 = 0.02 and the compressive end 4.8 % behind (within the 15 % their Fig. 7 shows), and both reach the same
    steady state, eq. (12), since it does not depend on the diffusivity (p. 3794). With a flat kappa(sigma) the
    iteration returns the constant-kappa result."""
    omega_kt = 1.66e-29 / (K_B * 473.15)
    g = 2 * 600e6 / L
    saves = [0.02 * L**2 / KAPPA]
    t_end = 4 * L**2 / KAPPA
    variable = solve_line_stress(L, 200, g, lambda s: KAPPA * np.exp(omega_kt * s), t_end, t_end / 2000, save_times=saves)
    constant = solve_line_stress(L, 200, g, KAPPA, t_end, t_end / 2000, save_times=saves)
    early_v, early_c = variable.stress[1], constant.stress[1]
    assert 1.03 < early_v[0] / early_c[0] < 1.15
    assert 0.85 < early_v[-1] / early_c[-1] < 0.97
    np.testing.assert_allclose(variable.stress[-1], steady_stress(variable.x, L, g), rtol=0, atol=1e-9 * g * L)
    flat = solve_line_stress(L, 200, g, lambda s: np.full_like(s, KAPPA), t_end, t_end / 2000, save_times=saves)
    np.testing.assert_allclose(flat.stress, constant.stress, rtol=0, atol=1e-9 * g * L)


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="width"):
        LineEnd("void")
    with pytest.raises(ValueError, match="blocked"):
        LineEnd("open")
    with pytest.raises(ValueError, match="yield"):
        solve_line_stress(L, 10, G, KAPPA, 1.0, 0.1, yield_stress=-1.0)
