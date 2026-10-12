"""Multicomponent coarsening, against Philippe and Voorhees, Acta Mater. 61 (2013) 4237: the closed forms of eqs.
(29), (42)-(45) and (47), their binary (LSW) limit, and a coarsening run of the precipitation solver in an ideal
ternary whose two solutes diffuse at different rates."""

import functools

import numpy as np
import pytest
from pycalphad import Database
from scipy.integrate import quad
from scipy.optimize import brentq

from janus_kinetics import NucleationSettings, PrecipitateThermo, SolutionPhase, solve_multicomponent_precipitation
from janus_kinetics.coarsening import (
    NUMBER_DENSITY_FACTOR,
    coarsening_rate_constant,
    critical_radius,
    matrix_supersaturation,
    number_density,
    precipitate_composition_shift,
)
from janus_kinetics.multicomponent_precipitation import K_B, N_A
from janus_kinetics.thermo import R


def test_binary_limit_is_lsw():
    """A pure-B precipitate in a dilute ideal binary with D_A = D_B = D: dC^T M^-1 dC = RT (1 - x)/(x D), so eq. (42)
    is Perez et al.'s LSW constant (4/9) x/(1 - x) R0 D, R0 = 2 gamma v_at / kT, the one test_multicomponent_precipitation
    checks the solver against."""
    x, d, gamma, v_at, T = 0.01, 5e-20, 0.13, 1.6e-29, 433.15
    m = np.array([[x * (1 - x) * d / (K_B * N_A * T)]])  # M = D / G, G = RT / (x (1 - x)); R = k N_A as Perez use k
    k = coarsening_rate_constant([1 - x], m, v_at * N_A, gamma)
    assert k == pytest.approx(4 / 9 * x / (1 - x) * 2 * gamma * v_at / (K_B * T) * d, rel=1e-12)


def test_diagonal_mobility_is_eq_47():
    dc, mob, vm, gamma = np.array([0.2, -0.05, 0.1]), np.array([3e-25, 1e-25, 7e-26]), 7e-6, 0.05
    k = coarsening_rate_constant(dc, np.diag(mob), vm, gamma)
    assert k == pytest.approx(8 * vm * gamma / (9 * np.sum(dc**2 / mob)), rel=1e-12)


def test_composition_shift_along_the_tie_line_only_for_equal_hessians():
    """p. 4242: with equal Hessians and equal partial volumes the precipitate's shift equals the matrix
    supersaturation of eq. (43), on the tie-line; with different Hessians G_beta^-1 G_alpha rotates it off. In
    general it is eq. (35) applied at the interface of the mean particle."""
    dc = np.array([0.15, 0.08])
    m = np.array([[2e-25, -3e-26], [-3e-26, 8e-26]])
    g = 8.314 * 1000 * np.array([[110.0, 12.0], [12.0, 60.0]])
    t = np.array([1e5, 1e7])
    sup = matrix_supersaturation(t, dc, m, g, 7e-6, 0.03)
    np.testing.assert_allclose(precipitate_composition_shift(t, dc, m, g, g, 7e-6, 0.03), sup, rtol=1e-12)
    np.testing.assert_allclose(sup[1] / sup[0], 100 ** (-1 / 3), rtol=1e-12)
    assert abs(sup[0][0] * dc[1] - sup[0][1] * dc[0]) < 1e-12 * np.linalg.norm(sup[0]) * np.linalg.norm(dc)
    shift = precipitate_composition_shift(t, dc, m, g, np.diag([400.0, 900.0]) * 8.314, 7e-6, 0.03)[0]
    assert abs(shift[0] * dc[1] - shift[1] * dc[0]) / (np.linalg.norm(shift) * np.linalg.norm(dc)) > 0.1
    # eq. (45) by another route: the mean particle has the critical radius, <R> = (K t)^(1/3), so its matrix
    # interface sits at the far-field composition of eq. (43), and eq. (35) gives the precipitate side
    g_b, dv = 8.314 * np.array([[400.0, -50.0], [-50.0, 900.0]]), np.array([1e-7, -2e-7])
    mean_r = (coarsening_rate_constant(dc, m, 7e-6, 0.03) * t) ** (1 / 3)
    via_35 = np.linalg.solve(g_b, (g @ sup.T - 2 * 0.03 * np.outer(dv, 1 / mean_r))).T
    np.testing.assert_allclose(precipitate_composition_shift(t, dc, m, g, g_b, 7e-6, 0.03, dv), via_35, rtol=1e-10)
    # the partial-volume term only adds -G_beta^-1 dV / V_m times the same amplitude
    extra = precipitate_composition_shift(t, dc, m, g, g, 7e-6, 0.03, dv) - sup
    amp = (3 * 0.03 * 7e-6) ** (2 / 3) * float(dc @ np.linalg.solve(m, dc)) ** (1 / 3)
    np.testing.assert_allclose(extra[0], -amp * 1e5 ** (-1 / 3) * np.linalg.solve(g, dv) / 7e-6, rtol=1e-10)


def test_number_density_factor_is_the_lsw_moment():
    """Eq. (44)'s 0.21 is 3 / (4 pi <u^3>) for the LSW distribution, the same one Appendix A integrates."""
    def g(u):
        return 0.0 if u >= 1.5 else u**2 * (3 / (3 + u)) ** (7 / 3) * (1.5 / (1.5 - u)) ** (11 / 3) * np.exp(-u / (1.5 - u))

    m0, m1, m3 = (quad(lambda u, k=k: u**k * g(u), 0, 1.5, limit=400)[0] for k in (0, 1, 3))
    assert m1 / m0 == pytest.approx(1.0, rel=1e-9)
    assert NUMBER_DENSITY_FACTOR == pytest.approx(3 / (4 * np.pi * m3 / m0), rel=1e-6)
    assert round(NUMBER_DENSITY_FACTOR, 2) == 0.21
    assert number_density(2e4, 1e-28, 0.1) == pytest.approx(NUMBER_DENSITY_FACTOR * 0.1 / (1e-28 * 2e4), rel=1e-12)


T, GAMMA, D_A, D_B, D_C = 433.15, 0.13, 5e-20, 5e-20, 1e-20
XP = np.array([0.6, 0.4, 0.0])
VM = 1.6e-29 * N_A


@functools.lru_cache(maxsize=None)
def _ternary_run():
    """Ideal fcc A-B-C (A the solvent) with a B0.6C0.4 line compound, solubility product x_B^0.6 x_C^0.4 = 0.01,
    D_C five times slower than D_A = D_B, the Perez et al. molar volume and interfacial energy, 2 % of
    precipitate seeded around 5 nm at its equilibrium composition (part I's frozen composition is then
    Philippe and Voorhees' assumption exactly). SFFK's dissipation sum_i dx_i^2 RT / (x_i D_i) over all
    components is dC^T M^-1 dC with M^-1 = RT [diag(1/(x_k D_k)) + 1/(x_A D_A)] over k = B, C."""
    tdb = f"""
    ELEMENT VA VACUUM 0 0 0 !
    ELEMENT A BLANK 1 0 0 !
    ELEMENT B BLANK 1 0 0 !
    ELEMENT C BLANK 1 0 0 !
    TYPE_DEFINITION % SEQ * !
    PHASE FCC % 1 1 !
    CONSTITUENT FCC :A,B,C: !
    PARAMETER G(FCC,A;0) 298.15 0; 6000 N !
    PARAMETER G(FCC,B;0) 298.15 0; 6000 N !
    PARAMETER G(FCC,C;0) 298.15 0; 6000 N !
    PARAMETER MQ(FCC&A,*;0) 298.15 R*T*LN({D_A}); 6000 N !
    PARAMETER MQ(FCC&B,*;0) 298.15 R*T*LN({D_B}); 6000 N !
    PARAMETER MQ(FCC&C,*;0) 298.15 R*T*LN({D_C}); 6000 N !
    PHASE PPT % 2 0.6 0.4 !
    CONSTITUENT PPT :B:C: !
    PARAMETER G(PPT,B:C;0) 298.15 R*T*LN(0.01); 6000 N !
    """
    db = Database(tdb)
    comps = ["B", "C", "A"]
    x0 = 0.98 * np.array([0.012, 0.008, 0.98]) + 0.02 * XP

    def off_solubility(phi):
        xm = (x0 - phi * XP) / (1 - phi)
        return 0.6 * np.log(xm[0]) + 0.4 * np.log(xm[1]) - np.log(0.01)

    phi_eq = brentq(off_solubility, 0.0, 0.0395)
    x_eq = (x0 - phi_eq * XP) / (1 - phi_eq)
    radii = np.linspace(4.5e-9, 5.5e-9, 41)
    num = np.exp(-((radii - 5e-9) / 0.3e-9) ** 2)
    num *= phi_eq / (num @ (4 * np.pi / 3 * radii**3))
    times = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9])
    res = solve_multicomponent_precipitation(
        SolutionPhase(db, "FCC", comps, mobility_db=db), PrecipitateThermo(db, "PPT", comps, composition=list(XP)),
        list(x0), T, times[-1], NucleationSettings(gamma=GAMMA, molar_volume=VM, lattice_parameter=4.04e-10), times,
        t_start=1.0, max_radius_change=0.005, min_classes=200, max_classes=400,
        initial={"number": num, "radius": radii, "composition": np.tile(XP, (radii.size, 1))})
    return res, times, x_eq


def _closed_form(x_eq, d_c=D_C):
    dc = (XP - x_eq)[:2]
    hessian = R * T * (np.diag(1 / x_eq[:2]) + 1 / x_eq[2])
    m_inv = R * T * (np.diag([1 / (x_eq[0] * D_B), 1 / (x_eq[1] * d_c)]) + 1 / (x_eq[2] * D_A))
    return dc, hessian, np.linalg.inv(m_inv)


def test_solver_coarsens_at_the_multicomponent_rate():
    """From 1e7 s on, d<R>^3/dt averages K of eq. (42) within 2 % (single windows within 10 %, as in the binary
    LSW test), and d(1/N_v)/dt averages K / (0.2113 phi) of eq. (44) within 2 %. The test discriminates: with D_C
    equal to D_B, K would be 2.6 times larger, and keeping only the slow species C (eq. 48) would overstate it by
    31 %, the paper's point that a slow species alone does not set the rate."""
    res, times, x_eq = _ternary_run()
    dc, hessian, mobility = _closed_form(x_eq)
    k = coarsening_rate_constant(dc, mobility, VM, GAMMA)
    rates = (np.diff(res.mean_radius**3) / np.diff(times) / k)[2:]
    assert np.mean(rates) == pytest.approx(1.0, abs=0.02)
    assert np.all(np.abs(rates - 1) < 0.1)
    decay = (np.diff(1 / res.number_density) / np.diff(times) * NUMBER_DENSITY_FACTOR * res.volume_fraction[1:] / k)[2:]
    assert np.mean(decay) == pytest.approx(1.0, abs=0.02)
    assert np.all(np.abs(decay - 1) < 0.12)
    assert coarsening_rate_constant(dc, _closed_form(x_eq, D_B)[2], VM, GAMMA) / k == pytest.approx(2.57, abs=0.01)
    slow_only = 8 * GAMMA * VM / (9 * R * T * dc[1] ** 2 / (x_eq[1] * D_C))
    assert slow_only / k == pytest.approx(1.31, abs=0.01)


def test_supersaturation_decays_along_the_tie_line():
    """The far-field supersaturation lies on the tie-line at every save, its critical radius (eq. 29) closes on
    <R> (0.935 at 1e6 s, 0.997 at 3e9 s), and from 3e8 s its size is eq. (43)'s within 2 %."""
    res, times, x_eq = _ternary_run()
    dc, hessian, mobility = _closed_form(x_eq)
    sup = np.array([xm[:2] - x_eq[:2] for xm in res.matrix_composition])
    cos = sup @ dc / (np.linalg.norm(sup, axis=1) * np.linalg.norm(dc))
    assert np.all(cos > 1 - 1e-9)
    ratio = np.array([critical_radius(dc, hessian, s, VM, GAMMA) for s in sup]) / res.mean_radius
    assert np.all(np.diff(ratio) > 0) and ratio[-1] > 0.99
    theory = matrix_supersaturation(times, dc, mobility, hessian, VM, GAMMA)
    np.testing.assert_allclose((sup @ dc / (theory @ dc))[-3:], 1.0, atol=0.02)


def test_input_checks():
    with pytest.raises(ValueError, match="square"):
        coarsening_rate_constant([0.1, 0.2], np.eye(3), 7e-6, 0.1)
    with pytest.raises(ValueError, match="positive definite"):
        coarsening_rate_constant([0.1, 0.2], -np.eye(2), 7e-6, 0.1)
    with pytest.raises(ValueError, match="positive"):
        coarsening_rate_constant([0.1], [[1e-25]], 0.0, 0.1)
    with pytest.raises(ValueError, match="Hessian"):
        matrix_supersaturation(1.0, [0.1, 0.2], np.eye(2), -np.eye(2), 7e-6, 0.1)
    assert critical_radius([0.1], [[1e5]], [-1e-3], 7e-6, 0.1) == np.inf
