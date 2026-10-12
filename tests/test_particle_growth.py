"""Growth of a particle with local equilibrium in a multicomponent matrix, against Chen, Jeppsson and Agren, Acta
Mater. 56 (2008) 1890: eq. (2) and its limits (p. 1891), and eqs. (9), (10) and (13) checked against Zener's exact
sphere solution and this package's sharp-interface solver. The paper's own example uses the TCFE4 and MOB2
commercial databases and is not reproduced."""

import functools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from pycalphad import Database
from scipy.optimize import brentq, root

from janus_kinetics import SolutionPhase
from janus_kinetics.moving_boundary import Layer, LocalEquilibrium, solve_layers
from janus_kinetics.particle_growth import diffusion_distance_factor, growth_parameter, particle_growth_rate

sys.path.insert(0, str(Path(__file__).parent))
from test_geometry import zener_alpha  # noqa: E402


def test_eq_2_is_zeners_sphere():
    """Eq. (2) is Zener's (1949) eq. (12) for a sphere with alpha = 2 lambda, R = alpha (D t)^(1/2), derived
    separately in test_geometry; and n -> 1 as Omega -> 0, n -> Omega (1 - Omega) / 3 as Omega -> 1 (p. 1891),
    the second followed to Omega = 1 - 1e-11, where eq. (2) cancels in floating point."""
    for omega in (1e-6, 1e-3, 0.05, 0.3, 0.7, 0.95):
        assert 2 * growth_parameter(omega) == pytest.approx(zener_alpha(3, omega), rel=1e-9)
    assert diffusion_distance_factor(1e-7) == pytest.approx(1.0, abs=1e-3)
    for omega in (0.999, 1 - 1e-6, 1 - 1e-8, 1 - 1e-11):
        assert diffusion_distance_factor(omega) == pytest.approx(omega * (1 - omega) / 3, rel=2e-3)
    # near Omega = 1 against eq. (2) evaluated through Laplace's continued fraction for erfc,
    # sqrt(pi) erfcx(x) = 1 / (x + K), K = (1/2) / (x + 1 / (x + (3/2) / (x + ...))), which gives
    # f = 2 lambda^2 K / (lambda + K) with no cancellation
    def cf(x, terms=400):
        k = 0.0
        for j in range(terms, 0, -1):
            k = (j / 2) / (x + k)
        return k

    for omega in (0.995, 0.9985, 0.999, 0.99999, 1 - 1e-7):
        lam = brentq(lambda x: 2 * x * x * cf(x) / (x + cf(x)) - omega, 3, 1e6, xtol=1e-12, rtol=1e-15)
        assert growth_parameter(omega) == pytest.approx(lam, rel=1e-9)
    n = diffusion_distance_factor(np.linspace(0.01, 0.99, 99))
    assert np.all(np.diff(n) < 0)
    assert diffusion_distance_factor(-0.2) == 1.0  # no root of eq. (2): the steady-state value, a package choice
    with pytest.raises(ValueError, match="0 < Omega < 1"):
        growth_parameter(1.0)


T = 1000.0
KA, KB, KC = 0.1, 50.0, 30.0  # partition ratios x^P / x^I of the ideal phases


def _tdb(elements, d):
    lines = ["ELEMENT VA VACUUM 0 0 0 !"] + [f"ELEMENT {e} BLANK 1 0 0 !" for e in elements]
    lines += ["TYPE_DEFINITION % SEQ * !", "PHASE FCC % 1 1 !", f"CONSTITUENT FCC :{','.join(elements)}: !",
              "PHASE PPT % 1 1 !", f"CONSTITUENT PPT :{','.join(elements)}: !"]
    for e, k in zip(elements, (KA, KB, KC)):
        lines += [f"PARAMETER G(FCC,{e};0) 298.15 0; 6000 N !", f"PARAMETER G(PPT,{e};0) 298.15 -R*T*LN({k}); 6000 N !",
                  f"PARAMETER MQ(FCC&{e},*;0) 298.15 R*T*LN({d[e]}); 6000 N !",
                  f"PARAMETER MQ(PPT&{e},*;0) 298.15 R*T*LN(1E-25); 6000 N !"]
    return "\n".join(lines)


@functools.lru_cache(maxsize=None)
def _phases(elements: tuple, d: tuple):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(_tdb(elements, dict(zip(elements, d))))
        order = list(elements[1:]) + [elements[0]]
        return SolutionPhase(db, "FCC", order), SolutionPhase(db, "PPT", order)


def test_binary_growth_is_exact():
    """A-B with partition ratios 0.1 and 50 and equal tracer diffusivities (so the interdiffusivity is D): without
    capillarity v R is Zener's D alpha^2 / 2 for the supersaturation of the tie-line, x^I_B = 0.9/49.9. The direct
    form, eq. (13), carries x^I M (mu^M - mu^I) = x^I D ln(x^M / x^I) for the flux, so in this ideal dilute matrix it
    falls short of the exact rate by x^I ln(x^M/x^I) / (x^M - x^I), the price of writing the flux with interface
    quantities alone."""
    d = 1e-14
    matrix, particle = _phases(("A", "B"), (d, d))
    xi, xp = 0.9 / 49.9, 50 * 0.9 / 49.9
    for far in (0.03, 0.1, 0.3):
        st = particle_growth_rate(matrix, particle, [far], 1e-6, T, ([0.02], [0.8]), diffusivity=[[d]])
        assert st.matrix_interface[0] == pytest.approx(xi, rel=1e-9)
        assert st.particle_interface[0] == pytest.approx(xp, rel=1e-9)
        omega = (far - xi) / (xp - xi)
        assert st.velocity * st.radius == pytest.approx(d * zener_alpha(3, omega) ** 2 / 2, rel=1e-8)
        direct = particle_growth_rate(matrix, particle, [far], 1e-6, T, ([0.02], [0.8]), form="mobility")
        assert direct.velocity / st.velocity == pytest.approx(xi * np.log(far / xi) / (far - xi), rel=1e-6)


TERNARY = (1e-13, 1e-14, 4e-14)  # tracer diffusivities of A, B, C in the matrix


@functools.lru_cache(maxsize=None)
def _sharp_interface(far, cells=750):
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    dm = np.diag(TERNARY[1:])
    st = particle_growth_rate(matrix, particle, far, 1e-6, T, ([0.01, 0.013], [0.5, 0.4]), diffusivity=dm)
    k = st.velocity * st.radius
    r0, r1 = 0.1e-6, 2e-6
    t_end = (r1**2 - r0**2) / (2 * k)
    layers = [Layer(np.tile(st.particle_interface[:2], (20, 1)), np.diag([1e-25, 1e-25])),
              Layer(np.tile(np.asarray(far), (cells, 1)), dm)]
    cond = [LocalEquilibrium(particle, matrix, T, st.particle_interface[:2], st.matrix_interface[:2])]
    res = solve_layers([0.0, r0, 40 * r1], layers, cond, t_end, t_end / 800, save_times=t_end * np.linspace(0.6, 1, 3),
                       geometry="spherical")
    return st, res


@pytest.mark.parametrize("far", [(0.02, 0.02), (0.012, 0.03)])
def test_ternary_growth_matches_the_sharp_interface_solver(far):
    """A-B-C, ideal, with B and C partitioned 50 and 30 times into the particle and C diffusing 4 times faster:
    each element gets its own factor (n = 0.79 and 0.89 for the first alloy). With a diagonal D and no capillarity
    every field is the binary one, so eq. (10) should be exact for growth from zero size; a sphere grown from
    0.1 to 2 um by solve_layers, with local equilibrium at its interface, reaches d(R^2)/dt = 2 v R within 0.1 %
    (0.05 % left whatever the grid, from its step-profile start) and the operating tie-line within 2e-6. With
    n = 1 for both, the two flux balances would ask for growth rates 20 % and 11 % below it."""
    st, res = _sharp_interface(far)
    k = st.velocity * st.radius
    r = res.interfaces[:, 0]
    assert (r[-1] ** 2 - r[-2] ** 2) / (res.times[-1] - res.times[-2]) == pytest.approx(2 * k, rel=1e-3)
    np.testing.assert_allclose(res.interface_right[-1, 0], st.matrix_interface[:2], atol=2e-6)
    np.testing.assert_allclose(res.interface_left[-1, 0], st.particle_interface[:2], atol=1e-4)
    assert np.all(st.factor < 0.92) and st.factor[0] < st.factor[1]
    steady = np.diag(TERNARY[1:]) @ (np.asarray(far) - st.matrix_interface[:2]) / (st.particle_interface[:2] - st.matrix_interface[:2])
    np.testing.assert_allclose(steady / k, st.factor, rtol=1e-6)


def test_cross_diffusion_enters_eq_10_column_by_column():
    """With off-diagonal D_ij the paper's sum divides each element's driving difference by its own factor n_j
    before the matrix acts, sum_j D_ij (c^M_j - c^I_j) / (n_j R); the returned state satisfies that, and not the
    variant dividing row by row, which would differ by 3 % here."""
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    far = np.array([0.012, 0.03])
    d = np.array([[1e-14, 0.4e-14], [-0.6e-14, 4e-14]])
    st = particle_growth_rate(matrix, particle, far, 1e-6, T, ([0.01, 0.013], [0.5, 0.4]), diffusivity=d)
    xi, xp = st.matrix_interface[:2], st.particle_interface[:2]
    lhs = st.velocity * st.radius * (xp - xi)
    np.testing.assert_allclose(lhs, d @ ((far - xi) / st.factor), rtol=1e-8)
    assert np.max(np.abs(lhs / ((d @ (far - xi)) / st.factor) - 1)) > 0.01


def test_capillarity_sets_the_critical_radius():
    """With sigma V_m on the particle the growth rate changes sign at R* = 2 sigma V_m / dG, dG the driving force
    of the parallel tangent to the far-field matrix (found here independently); the rate vanishes there within
    1e-6 of its size at 2 R*, and the operating tie-line is then the parallel tangent itself."""
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    far = np.array([0.02, 0.02])
    mu = matrix.chemical_potentials(far, T)

    def tangent(u):
        return particle.chemical_potentials(u[:2], T) - mu - u[2]

    sol = root(tangent, [0.5, 0.4, 0.0])
    xp, d_mu = sol.x[:2], sol.x[2]
    drive = -d_mu  # per mole of particle: sum_k x^P_k mu^M_k - G^P = -d_mu
    gamma, vm = 0.3, 7e-6
    r_star = 2 * gamma * vm / drive
    kw = dict(diffusivity=np.diag(TERNARY[1:]), gamma=gamma, molar_volume=vm)
    at = particle_growth_rate(matrix, particle, far, r_star, T, (far * 0.99, xp), **kw)
    big = particle_growth_rate(matrix, particle, far, 2 * r_star, T, ([0.01, 0.013], [0.5, 0.4]), **kw)
    small = particle_growth_rate(matrix, particle, far, 0.5 * r_star, T, (far * 0.99, xp), **kw)
    assert abs(at.velocity) < 1e-6 * big.velocity and big.velocity > 0 > small.velocity
    np.testing.assert_allclose(at.particle_interface[:2], xp, atol=1e-6)
    np.testing.assert_allclose(at.matrix_interface[:2], far, atol=1e-8)


def test_input_checks():
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    start = ([0.01, 0.013], [0.5, 0.4])
    with pytest.raises(ValueError, match="needs the matrix interdiffusion"):
        particle_growth_rate(matrix, particle, [0.02, 0.02], 1e-6, T, start)
    with pytest.raises(ValueError, match="2 x 2"):
        particle_growth_rate(matrix, particle, [0.02, 0.02], 1e-6, T, start, diffusivity=np.eye(3))
    with pytest.raises(ValueError, match="simplex"):
        particle_growth_rate(matrix, particle, [0.02, 0.02], 1e-6, T, ([0.01, 0.013], [0.7, 0.4]), diffusivity=np.eye(2))
    with pytest.raises(ValueError, match="one fewer"):
        particle_growth_rate(matrix, particle, [0.02], 1e-6, T, start, diffusivity=np.eye(2))
    with pytest.raises(ValueError, match="form"):
        particle_growth_rate(matrix, particle, [0.02, 0.02], 1e-6, T, start, form="other")
