"""Grain-boundary diffusion from a surface source, Fisher, J. Appl. Phys. 22 (1951) 74."""

import numpy as np
import pytest
from scipy.special import erfc

from calphad_kinetics.grain_boundary import FisherBoundary, boundary_product_from_slope, solve_fisher

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
