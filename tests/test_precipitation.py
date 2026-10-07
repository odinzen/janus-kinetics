"""Precipitation kinetics against Perez, Dumont and Acevedo-Reyes, Acta Mater. 56 (2008) 2119,
with eqs. (4), (7) and note 3 as corrected in Acta Mater. 57 (2009) 1318.

The parameters are those of their Table 1 (p. 2125): X^P = 1, X^0 = 0.06,
X^i = 0.01, D = 5e-20 m^2/s, gamma = 0.13 J/m^2, v_at = 1.6e-29 m^3,
a = 4.04e-10 m, T = 160 C. The paper does not state the number of nucleation
sites; here every lattice site counts, N0 = 1/v_at.
"""

import numpy as np
import pytest

from calphad_kinetics.precipitation import (
    K_B,
    DiluteSolution,
    Kinetics,
    growth_rate,
    nucleation_rate,
    solubility_from_pycalphad,
    solve_kwn,
)

V_AT = 1.6e-29
T = 160 + 273.15
THERMO = DiluteSolution(x_eq=0.01, x_p=1.0, v_at_p=V_AT, v_at_m=V_AT, gamma=0.13, T=T)
KIN = Kinetics(diffusivity=5e-20, lattice_parameter=4.04e-10, sites=1 / V_AT)
EDGES = np.arange(0.2e-9, 60e-9, 0.05e-9)


def test_gibbs_thomson_is_eq9():
    """For a pure precipitate (x = 0, y = 1) eq. (9) is X^i(R) = X^i(inf) exp(2 gamma v_at / R kT)."""
    r = np.array([0.5e-9, 2e-9, 20e-9])
    expected = 0.01 * np.exp(2 * 0.13 * V_AT / (r * K_B * T))
    np.testing.assert_allclose(THERMO.interface_composition(r), expected, rtol=1e-14)


@pytest.mark.parametrize("x_p", [1.0, 0.5])
def test_critical_nucleus_is_in_equilibrium_with_the_matrix(x_p):
    """R* = -2 gamma/dg (eq. 1) and eq. (9) agree: X^i(R*) equals the matrix fraction."""
    thermo = DiluteSolution(0.01, x_p, V_AT, V_AT, 0.13, T)
    for x in (0.012, 0.03, 0.06):
        assert float(thermo.interface_composition(thermo.critical_radius(x))) == pytest.approx(x, rel=1e-10)
    assert np.isinf(thermo.critical_radius(0.009))


def test_nucleation_rate_by_hand():
    """Eq. (6) with corrected eq. (4) for Z, eq. (5) for beta*, tau = 4/(2 pi beta* Z^2), corrected eq. (7)."""
    x, t = 0.05, 300.0
    kt = K_B * T
    dg = -(kt / V_AT) * np.log(x / 0.01)
    r_star = -2 * 0.13 / dg
    z = V_AT / (2 * np.pi * r_star**2) * np.sqrt(0.13 / kt)
    beta = 4 * np.pi * r_star**2 * 5e-20 * x / 4.04e-10**4
    tau = 4 / (2 * np.pi * beta * z**2)
    rate = (1 / V_AT) * z * beta * np.exp(-16 * np.pi * 0.13**3 / (3 * dg**2) / kt) * np.exp(-tau / t)
    got, r_nuc = nucleation_rate(THERMO, KIN, x, t)
    assert got == pytest.approx(rate, rel=1e-12)
    assert r_nuc == pytest.approx(r_star + 0.5 * np.sqrt(kt / (np.pi * 0.13)), rel=1e-12)


@pytest.mark.parametrize("alpha", [1.0, 1.4])
def test_isolated_particles_follow_zener_parabolic_growth(alpha):
    """Without capillarity and at fixed X, eq. (8) integrates to R^2 = R0^2 + 2 D (X - X^i)/(alpha X^P - X^i) t."""
    thermo = DiluteSolution(0.01, 1.0, V_AT, alpha * V_AT, 1e-12, T)  # negligible gamma: X^i(R) = X^i(inf)
    kin = Kinetics(5e-20, 4.04e-10, 0.0)
    edges = np.arange(1e-9, 40e-9, 0.02e-9)
    c = 0.5 * (edges[1:] + edges[:-1])
    n0 = np.where(np.abs(c - 5e-9) < 0.011e-9, 1e10, 0.0)  # so few particles that X stays at 0.04
    res = solve_kwn(thermo, kin, 0.04, 2e5, edges, number0=n0, save_times=[5e4])
    lam = 5e-20 * (0.04 - 0.01) / (alpha - 0.01)
    for i, t in enumerate(res.times):
        assert res.mean_radius[i] == pytest.approx(np.sqrt(res.mean_radius[0] ** 2 + 2 * lam * t), rel=2e-3)


@pytest.fixture(scope="module")
def table1():
    return solve_kwn(THERMO, KIN, 0.06, 1e7, EDGES, save_times=[1e3, 3e3, 1e4, 3e4, 1e5, 1e6])


def test_undersaturated_alloy_dissolves_every_particle():
    """Below the solubility every particle shrinks (eq. 10) and leaves through the
    lowest class edge, so the population vanishes and all solute is back in the matrix."""
    edges = np.arange(0.2e-9, 20e-9, 0.05e-9)
    c = 0.5 * (edges[1:] + edges[:-1])
    n0 = np.exp(-0.5 * ((c - 3e-9) / 0.4e-9) ** 2)
    n0 *= 0.002 / (4 / 3 * np.pi * (n0 @ c**3))
    res = solve_kwn(THERMO, KIN, 0.006, 1e8, edges, number0=n0, save_times=[1e3, 1e5])
    f = res.volume_fraction
    assert np.all(np.diff(f) <= 0) and f[-1] < 1e-9 * f[0]
    assert res.number_density[-1] < 1e-9 * res.number_density[0]
    assert res.matrix_fraction[-1] == pytest.approx(0.006, rel=1e-6)


def test_nucleation_growth_coarsening_sequence(table1):
    """Section 3.5.1: nucleation and growth deplete the matrix, then the number density falls by coarsening."""
    n, r, x = table1.number_density, table1.mean_radius, table1.matrix_fraction
    assert np.all(np.diff(x) < 0) and x[-1] == pytest.approx(0.01, abs=1e-3)
    assert np.all(np.diff(r[1:]) > 0)
    peak = np.argmax(n)
    assert 0 < peak < len(n) - 1 and n[-1] < 0.05 * n[peak]


def test_mean_radius_meets_critical_radius_after_growth(table1):
    """Once the supersaturation is consumed, R_mean and R* coincide (the LSW regime)."""
    t = table1.times
    late = t >= 3e4  # growth has consumed the supersaturation by then (X stops falling)
    np.testing.assert_allclose(table1.mean_radius[late], table1.critical_radius[late], rtol=2e-2)


def test_results_insensitive_to_size_grid_and_step(table1):
    fine = solve_kwn(THERMO, KIN, 0.06, 1e7, np.arange(0.2e-9, 60e-9, 0.025e-9), save_times=[3e4], max_change=0.005)
    i = list(table1.times).index(3e4)
    assert fine.number_density[1] == pytest.approx(table1.number_density[i], rel=3e-2)
    assert fine.mean_radius[-1] == pytest.approx(table1.mean_radius[-1], rel=2e-2)


def test_coarsening_approaches_lsw_rate():
    """Eq. (11) gives d(R_mean^3)/dt = (4/9) X^i / (alpha X^P - X^i) R0 D in the asymptotic limit.

    From a narrow initial distribution the rate falls toward this value from above
    as the distribution relaxes; refining the size classes lowers it further.
    """
    edges = np.arange(0.2e-9, 120e-9, 0.05e-9)
    c = 0.5 * (edges[1:] + edges[:-1])
    n0 = np.exp(-0.5 * ((c - 5e-9) / 0.5e-9) ** 2)
    n0 *= 0.04 / (4 / 3 * np.pi * (n0 @ c**3))
    x0 = 0.04 + 0.96 * float(THERMO.interface_composition(5e-9))
    res = solve_kwn(THERMO, Kinetics(5e-20, 4.04e-10, 0.0), x0, 3e8, edges, number0=n0, save_times=[1e7, 1e8])
    lsw = 4 / 9 * 0.01 / (1.0 - 0.01) * THERMO.capillary_length * 5e-20
    r3, t = res.mean_radius**3, res.times
    early = (r3[2] - r3[1]) / (t[2] - t[1]) / lsw
    late = (r3[3] - r3[2]) / (t[3] - t[2]) / lsw
    assert 1.0 < late < early
    assert late < 1.2
    assert res.critical_radius[-1] == pytest.approx(res.mean_radius[-1], rel=1e-3)


@pytest.mark.skip(reason="Perez et al. (2008) Fig. 7 cannot be reproduced from their Table 1: in the plateau "
                         "R_mean = R* = 4.8 nm and X = 0.0142, which by their eq. (9) needs 2 gamma v_at/kT = 1.7 nm, "
                         "while Table 1 gives 0.70 nm")
def test_table1_against_fig7():
    pass


def test_cutoff_and_overflow_are_reported():
    with pytest.raises(ValueError, match="lowest class edge"):
        solve_kwn(THERMO, KIN, 0.06, 1e3, np.arange(0.1e-9, 10e-9, 0.05e-9))
    with pytest.raises(ValueError, match="largest class"):
        solve_kwn(THERMO, KIN, 0.06, 1e7, np.arange(0.2e-9, 3e-9, 0.05e-9))


def test_growth_rate_sign_follows_supersaturation():
    r = np.array([0.5e-9, 5e-9])
    v = growth_rate(THERMO, KIN, 0.012, r)
    assert v[0] < 0 < v[1]  # small particles dissolve, large ones grow (Gibbs-Thomson, eq. 10)


def test_solubility_comes_from_pycalphad_equilibrium():
    """The A-B tie-line of the two-phase test database: alpha x_A = 0.5606, beta x_A = 0.2662 at 1000 K."""
    from pycalphad import Database
    from test_moving_boundary import TWO_PHASE

    db = Database(TWO_PHASE)
    x_beta, x_alpha = solubility_from_pycalphad(db, ["A", "B"], "BETA", "ALPHA", "A", 1000.0, 0.45)
    assert x_beta == pytest.approx(0.2662, abs=1e-4)
    assert x_alpha == pytest.approx(0.5606, abs=1e-4)
    with pytest.raises(ValueError, match="not in the"):
        solubility_from_pycalphad(db, ["A", "B"], "BETA", "ALPHA", "A", 1000.0, 0.9)
