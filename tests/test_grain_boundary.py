"""Grain-boundary diffusion from a surface source, Fisher, J. Appl. Phys. 22 (1951) 74, and Whipple's solution as
Le Claire, Brit. J. Appl. Phys. 14 (1963) 351, gives and analyses it."""

import numpy as np
import pytest
from scipy.special import erfc

from janus_kinetics.grain_boundary import (
    FisherBoundary,
    boundary_product_from_slope,
    harrison_regime,
    le_claire_boundary_product,
    solve_fisher,
    suzuoka_coefficient,
    suzuoka_layer_integral,
    whipple_beta,
    whipple_concentration,
    whipple_layer_integral,
)

# Fig. 3 (p. 75) at t1 = 100: (y1, x1) -> eq. (8) value in units of C_surface / 1000 (lower numbers)
FIG3_EQ8 = {(0, 16): 258, (0, 32): 24, (0, 48): 1, (4, 0): 261, (4, 16): 67, (4, 32): 6, (8, 0): 68,
            (8, 16): 18, (8, 32): 2, (12, 0): 18, (12, 16): 5, (16, 0): 5, (16, 16): 1, (20, 0): 1}


def talbot(transform, t, terms=32):
    """Inverse Laplace transform on a fixed Talbot contour, for the exact reference below.
    Checked against a transform with a known inverse in the first test."""
    r = 2 * terms / (5 * t)
    theta = np.arange(1, terms) * np.pi / terms
    cot = 1 / np.tan(theta)
    s = r * theta * (cot + 1j)
    sigma = theta + (theta * cot - 1) * cot
    return r / terms * (0.5 * np.real(transform(r + 0j)) * np.exp(r * t)
                        + np.sum(np.real(np.exp(t * s) * transform(s) * (1 + 1j * sigma))))


def exact(x1, y1, t1):
    """Eqs. (3a)-(4a) by Laplace transform in t1: grains give C~ = phi~ exp(-x1 s^1/2) and the slab
    phi~'' = (s + 2 s^1/2) phi~ with phi~(0) = 1/s, so C~ = exp(-y1 (s + 2 s^1/2)^1/2 - x1 s^1/2) / s."""
    return talbot(lambda s: np.exp(-y1 * np.sqrt(s + 2 * np.sqrt(s)) - x1 * np.sqrt(s)) / s, t1)


def test_laplace_inverter_against_erfc():
    for x in (5.0, 16.0, 32.0):
        assert talbot(lambda s: np.exp(-x * np.sqrt(s)) / s, 100.0) == pytest.approx(erfc(x / 20), rel=1e-9)


def test_eq8_reproduces_the_values_fisher_tabulated():
    b = FisherBoundary(d_lattice=1.0, d_boundary=1e6, delta=1.0)  # scaled units: x1 = x, t1 = t
    for (y1, x1), value in FIG3_EQ8.items():
        c = b.concentration(x1, y1 * 1e3, 100.0)  # y = y1 delta (D'/D)^(1/2)
        assert abs(1000 * c - value) <= 0.5 + 1e-9 * value, (y1, x1)


@pytest.fixture(scope="module")
def numerical():
    return solve_fisher(100.0, dx=0.25, dy=0.125, steps=1600)


def test_numerical_solution_matches_the_exact_one(numerical):
    for y1, x1 in ((0, 16), (4, 0), (4, 16), (8, 0), (8, 16), (12, 0)):
        assert numerical.at(x1, y1) == pytest.approx(exact(x1, y1, 100.0), rel=5e-3, abs=2e-5), (y1, x1)


def test_eq8_overestimates_penetration_at_depth(numerical):
    """At y1 = 8 next to the boundary eq. (8) gives 0.068, the exact solution 0.028."""
    b = FisherBoundary(1.0, 1e6, 1.0)
    assert b.concentration(0.0, 8e3, 100.0) == pytest.approx(0.0681, abs=5e-4)
    assert exact(0.0, 8.0, 100.0) == pytest.approx(0.0282, abs=5e-4)


def test_eq14_is_the_slope_of_the_eq8_layer_average():
    b = FisherBoundary(d_lattice=1e-17, d_boundary=1e-11, delta=5e-10)
    t = 3600.0
    x = np.linspace(0, 20 * np.sqrt(1e-17 * t), 4001)
    depths = np.array([2e-6, 4e-6])
    avg = [np.trapezoid(b.concentration(x, y, t), x) for y in depths]
    assert np.diff(np.log10(avg))[0] / np.diff(depths)[0] == pytest.approx(b.penetration_slope(t), rel=1e-6)


def test_slope_inversion_round_trip_and_fishers_silver_numbers():
    b = FisherBoundary(d_lattice=2e-18, d_boundary=3e-12, delta=5e-10)
    t = 63 * 3600.0
    assert boundary_product_from_slope(b.penetration_slope(t), b.d_lattice, t) == pytest.approx(3e-12 * 5e-10, rel=1e-12)
    # eqs. (15)-(18): D / (D' delta) = 21.3 cm^-1 and delta = 5e-8 cm give D'/D = 0.94e6
    assert (1 / 21.3) / 5e-8 == pytest.approx(0.94e6, rel=1e-2)


def test_dimensional_solve_uses_eq5_scaling():
    b = FisherBoundary(d_lattice=1e-17, d_boundary=1e-11, delta=5e-10)
    sol = b.solve(0.25 * 100 * (5e-10) ** 2 / 1e-17 * 4)  # t1 = 100
    assert sol.t1 == pytest.approx(100.0)
    assert sol.layer_average()[0] == pytest.approx(np.trapezoid(sol.c[0], sol.x1))


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        FisherBoundary(1e-11, 1e-17, 5e-10)
    with pytest.raises(ValueError, match="negative"):
        boundary_product_from_slope(0.1, 1e-17, 100.0)


# --- Whipple's solution as Le Claire, Brit. J. Appl. Phys. 14 (1963) 351, writes and uses it --------------------

def test_boundary_concentration_depends_on_eta_over_root_beta_alone_at_large_beta():
    """Le Claire's Fig. 5 and p. 353: above beta of about 10 the concentration at the boundary is a function of
    eta beta^(-1/2) only, and curves for beta above 100 cannot be told apart; at beta = 1 it is not (1.45 times the
    large-beta value at eta beta^(-1/2) = 1, rising to 4.8 at 8)."""
    z = np.array([1.0, 2.0, 4.0, 6.0, 8.0])
    c = {b: np.array([whipple_concentration(zz * np.sqrt(b), 0.0, b) for zz in z]) for b in (1.0, 10.0, 100.0, 1e3)}
    np.testing.assert_allclose(c[100.0], c[1e3], rtol=0.02)
    np.testing.assert_allclose(c[10.0], c[1e3], rtol=0.2)
    ratio = c[1.0] / c[1e3]
    assert np.all(ratio > 1.4) and np.all(np.diff(ratio) > 0) and ratio[-1] > 4.5


def test_exact_fisher_equations_are_the_large_beta_limit_of_whipple():
    """Fisher's equations drop the y-diffusion in the grains and the D'/D that sets beta; their exact solution
    approaches Whipple's at large beta as t1 grows, at the boundary at eta beta^(-1/2) = 3.58 within 9.5, 4.8 and
    2.3 % for t1 = 100, 400 and 1600."""
    z = np.sqrt(2) * 8 / 100**0.25
    whipple = whipple_concentration(z * 100.0, 0.0, 1e4)
    gaps = []
    for t1 in (100.0, 400.0, 1600.0):
        y1 = z * t1**0.25 / np.sqrt(2)
        sol = solve_fisher(t1, dx=0.025 * np.sqrt(t1), dy=0.079 * t1**0.25, steps=800)
        gaps.append(abs(sol.at(0.0, y1) / whipple - 1))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 0.03


def test_layer_average_slope_meets_eq_14_and_the_six_fifths_law():
    """The log slope of eq. (13) approaches Le Claire's asymptotic eq. (14) with depth (0.02 apart at eta beta^(-1/2)
    = 12 for beta = 100), and against (eta beta^(-1/2))^(6/5) it stays within 0.75-0.81 over 2-12, the near-constant
    0.78 of p. 356 behind eq. (15)."""
    beta = 100.0
    z = np.linspace(1.5, 12.0, 43)
    log_c = np.log([whipple_layer_integral(zz * np.sqrt(beta), beta) for zz in z])
    slope = -np.gradient(log_c, z)
    eq14 = 4 / 3 / z + 0.6307 * z ** (1 / 3) - 0.264 * z ** (-1 / 3) / beta
    gap = np.abs(slope - eq14)
    assert gap[-1] < 0.025 and gap[-1] < gap[20] < gap[5]
    six_fifths = -np.gradient(log_c, z**1.2)
    inside = (z >= 2) & (z <= 12)
    assert np.all((six_fifths[inside] > 0.75) & (six_fifths[inside] < 0.81))


def test_le_claire_reads_back_the_boundary_product_and_fisher_does_not():
    """A sectioning profile from Whipple's eq. (13) for D = 1e-17 m^2/s, D' delta = 2e-21 m^3/s and t = 1e5 s (beta =
    100), fitted over eta beta^(-1/2) = 2-10: eq. (15) returns D' delta 3.4 % high, the profile's own coefficient being
    0.765 against the rounded 0.78. Fisher's straight line in ln c_bar against y, read with his eq. (14), returns a
    third of the true value: Le Claire's 'about three times too small' (p. 355)."""
    d, product, t = 1e-17, 2e-21, 1e5
    beta = product / (2 * d * np.sqrt(d * t))
    assert beta == pytest.approx(100.0, rel=1e-12)
    assert whipple_beta(d, product / 5e-10, 5e-10, t) == pytest.approx(beta, rel=1e-12)
    y = np.linspace(2.0, 10.0, 33) * np.sqrt(beta) * np.sqrt(d * t)
    log_c = np.log([whipple_layer_integral(yy / np.sqrt(d * t), beta) for yy in y])
    slope_65 = np.polyfit(y**1.2, log_c, 1)[0]
    assert le_claire_boundary_product(slope_65, d, t) == pytest.approx(product, rel=0.04)
    fisher = boundary_product_from_slope(np.polyfit(y, log_c / np.log(10), 1)[0], d, t)
    assert product / fisher == pytest.approx(3.0, rel=0.05)


def test_le_claire_inputs_are_checked():
    with pytest.raises(ValueError, match="negative"):
        le_claire_boundary_product(0.1, 1e-17, 1e5)


def test_harrison_regimes_follow_his_conditions():
    """Harrison (1961): type A once a_g^2 / D_g << t (eq. 1), type C while (D_g t)^(1/2) << delta (eq. 5), type B
    between; '<<' taken as a factor 10. A 1 um grain with D = 1e-17 m^2/s needs 1e5 s to cross, so it is A beyond
    1e6 s; a 0.5 nm boundary keeps it C below 2.5e-4 s."""
    d, a_g, delta = 1e-17, 1e-6, 0.5e-9
    assert harrison_regime(d, a_g, delta, 2e6) == "A"
    assert harrison_regime(d, a_g, delta, 1e4) == "B"
    assert harrison_regime(d, a_g, delta, 1e-4) == "C"
    assert harrison_regime(d, a_g, delta, 0.9e6) == "B"  # just short of the factor 10
    assert harrison_regime(d, a_g, delta, 3e-4) == "B"
    assert harrison_regime(d, a_g, delta, 2e6, margin=100.0) == "B"
    with pytest.raises(ValueError, match="positive"):
        harrison_regime(d, a_g, delta, -1.0)


def test_suzuoka_profile_is_the_depth_derivative_of_whipples():
    """Le Claire (p. 356): the thin-source layer average is Whipple's eq. (13) differentiated in eta."""
    for eta, beta in ((5.0, 10.0), (30.0, 100.0)):
        h = 1e-4 * eta
        derivative = -(whipple_layer_integral(eta + h, beta) - whipple_layer_integral(eta - h, beta)) / (2 * h)
        assert suzuoka_layer_integral(eta, beta) == pytest.approx(derivative, rel=1e-6)


@pytest.mark.parametrize("beta", [10.0, 100.0])
def test_suzuoka_six_fifths_slope_is_le_claires_eq_16(beta):
    """Fitted over eta beta^(-1/2) = 2-10 the thin-source 6/5 slope is 0.738 at beta = 10 and 0.744 at 100, against
    0.72 beta^0.008 = 0.733 and 0.747; read with that coefficient, eq. (15) returns D' delta within 2 %. (At beta = 1
    the boundary term is negative near the surface and still rising at eta beta^(-1/2) = 2, peaking near 3, so a
    straight-line fit over 2-10 is not meaningful there.)"""
    d, t = 1e-17, 1e5
    product = beta * 2 * d * np.sqrt(d * t)
    y = np.linspace(2.0, 10.0, 33) * np.sqrt(beta * d * t)
    log_c = np.log([suzuoka_layer_integral(yy / np.sqrt(d * t), beta) for yy in y])
    slope = np.polyfit((y / np.sqrt(beta * d * t)) ** 1.2, log_c, 1)[0]
    assert -slope == pytest.approx(suzuoka_coefficient(beta), rel=0.01)
    read = le_claire_boundary_product(np.polyfit(y**1.2, log_c, 1)[0], d, t, suzuoka_coefficient(beta))
    assert read == pytest.approx(product, rel=0.02)
    assert suzuoka_layer_integral(0.5, 1.0) < 0 < suzuoka_layer_integral(2.0, 1.0) < suzuoka_layer_integral(3.0, 1.0)


# --- Whipple, Phil. Mag. 45 (1954) 1225, read first-hand ---------------------------------------------------------

def _whipple_contour(c, beta):
    from scipy.optimize import brentq

    eta0 = brentq(lambda e: whipple_concentration(e, 0.0, beta) - c, 1e-6, 30.0, xtol=1e-12)
    h = 1e-4  # one-sided in xi: the lattice lies at xi >= 0
    c_xi = (whipple_concentration(eta0, h, beta) - whipple_concentration(eta0, 0.0, beta)) / h
    c_eta = (whipple_concentration(eta0 + h, 0.0, beta) - whipple_concentration(eta0 - h, 0.0, beta)) / (2 * h)
    return eta0, -c_xi / c_eta


def test_whipple_contour_angles_of_figs_1_and_2():
    """Figs. 1-2 print the angle at which the contours C = 0.8, 0.6, 0.4, 0.2 meet the slab: 89.0, 88.0, 86.7, 85.1
    degrees for beta = 0.1 and 81.9, 73.8, 65.8, 57.5 for beta = 1 (90 degrees less the tilt, tan theta = d eta /
    d xi). From the solution here they are within 0.06 degrees for beta = 0.1 and 0.15 for the two upper contours
    at beta = 1; the two deeper ones come to 65.3 and 56.4, 0.5 and 1.1 degrees under the printed angles, which
    Whipple worked out by numerical integration in 1954."""
    printed = {0.1: (89.0, 88.0, 86.7, 85.1), 1.0: (81.9, 73.8, 65.8, 57.5)}
    tolerance = {0.1: (0.1, 0.1, 0.1, 0.1), 1.0: (0.2, 0.2, 0.6, 1.2)}
    for beta, angles in printed.items():
        for c, angle, tol in zip((0.8, 0.6, 0.4, 0.2), angles, tolerance[beta]):
            _, slope = _whipple_contour(c, beta)
            assert 90 - np.degrees(np.arctan(abs(slope))) == pytest.approx(angle, abs=tol), (beta, c)


def test_whipple_penetration_figs_4_and_5():
    """Fig. 4 plots the depth where C = 0.6 meets the slab over its depth far away, labelled 0.742: that is
    2 erfc^-1(0.6), the lattice term erfc(eta / 2) of his eq. (3.12). (His eqs. 5.1-5.2 print the term as erfc eta,
    a misprint the 0.742 confirms.) The ratio is 1.39 at beta = 1 and 2.88 at beta = 10, as the curve reads, and the
    contour's slope where it meets the slab, Fig. 5, is 0.29 and 2.0 (63 degrees)."""
    from scipy.special import erfcinv

    far = 2 * erfcinv(0.6)
    assert far == pytest.approx(0.742, abs=5e-4)
    for beta, ratio, tan in ((1.0, 1.39, 0.29), (10.0, 2.88, 2.0)):
        eta0, slope = _whipple_contour(0.6, beta)
        assert eta0 / far == pytest.approx(ratio, abs=0.05)
        assert abs(slope) == pytest.approx(tan, abs=0.05 * tan)


def test_whipple_steepest_descents_eq_5_3():
    """Eq. (5.3), C_2 = 1.159 beta^(1/3) eta^(-2/3) exp[-(0.473 beta^(-2/3) eta^(4/3) - 0.396 beta^(-4/3) eta^(2/3)
    (1 - beta xi))]: its coefficients are 2^(11/6) (3 pi)^(-1/2), 3 / 2^(8/3) and 2^(-4/3) of eqs. (A.17)-(A.18), and
    on the slab it approaches the exact boundary term from above as eta grows, within 6 % by eta = 32 for beta = 1
    and 10 (22 and 54 % at eta = 4, outside the range Whipple gives for it)."""
    assert 2 ** (11 / 6) / np.sqrt(3 * np.pi) == pytest.approx(1.159, abs=2e-3)
    assert 3 / 2 ** (8 / 3) == pytest.approx(0.473, abs=6e-4)  # 0.47247, printed rounded up
    assert 2 ** (-4 / 3) == pytest.approx(0.396, abs=1e-3)
    for beta in (1.0, 10.0):
        ratios = []
        for eta in (4.0, 8.0, 16.0, 32.0):
            exact = whipple_concentration(eta, 0.0, beta) - erfc(eta / 2)
            asym = 1.159 * beta ** (1 / 3) * eta ** (-2 / 3) * np.exp(-(0.473 * beta ** (-2 / 3) * eta ** (4 / 3)
                                                                         - 0.396 * beta ** (-4 / 3) * eta ** (2 / 3)))
            ratios.append(asym / exact)
        assert np.all(np.diff(ratios) < 0) and 1.0 < ratios[-1] < 1.06
