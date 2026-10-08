"""Two-phase planar diffusion with a moving interface.

Exact references:

* Zener, J. Appl. Phys. 20 (1949) 950: a phase of fixed composition n0 growing
  into a matrix with constant D, eqs. (10)-(12) with lambda = 1. Eq. (3) states
  that K1 = alpha (n0 - n_inf)^1/2 (n0 - n1)^1/2 / (n_inf - n1) lies between 1.13
  and 1.41, which checks the transcription of eq. (12).
* Both phases diffusing with constant D, derived below by the same similarity
  argument Zener uses, and checked to reduce to his eq. (12) as D_L -> 0.
"""

import numpy as np
import pytest
from pycalphad import Database, equilibrium, variables as v
from scipy.optimize import brentq
from scipy.special import erfc, erfcx

from janus_kinetics import SolutionPhase
from janus_kinetics.moving_boundary import (
    FixedInterface,
    LocalEquilibrium,
    PhaseDissolvedError,
    solve_moving_boundary,
)
from conftest import HEADER

T = 1000.0

# ---------------------------------------------------------------- Zener (1949)

N0, N1, NINF, D = 0.95, 0.05, 0.20, 1e-14


def phi1(a):
    """Zener eq. (11) for lambda = 1, the integral of exp(-z^2/4) from a to infinity."""
    return np.sqrt(np.pi) * erfc(a / 2)


def zener_alpha(n0, n1, ninf):
    xi = (ninf - n1) / (n0 - n1)  # eq. (2)
    return brentq(lambda a: a / xi - 2 * np.exp(-a * a / 4) / phi1(a), 1e-9, 50)  # eq. (12)


def test_zener_growth_coefficient_lies_in_the_stated_range():
    for ninf in (0.06, 0.2, 0.5, 0.9):
        a = zener_alpha(N0, N1, ninf)
        k1 = a * np.sqrt(N0 - ninf) * np.sqrt(N0 - N1) / (ninf - N1)
        assert 1.13 - 0.01 < k1 < 1.41 + 0.01


def zener_run(n_right, steps):
    alpha = zener_alpha(N0, N1, NINF)
    t0, t1 = 1e3, 1e5
    s0 = alpha * np.sqrt(D * t0)
    length = 40 * np.sqrt(D * t1)
    zr = s0 + (length - s0) * (np.arange(n_right) + 0.5) / n_right
    x_right = NINF + (N1 - NINF) * phi1(zr / np.sqrt(D * t0)) / phi1(alpha)  # eq. (10) at t0
    res = solve_moving_boundary(length, s0, np.full(20, N0), x_right, 0.0, D,
                                FixedInterface([N0], [N1]), t1 - t0, (t1 - t0) / steps)
    return res, alpha, t1


@pytest.fixture(scope="module")
def zener():
    return zener_run(800, 400)


def test_zener_interface_position(zener):
    res, alpha, t1 = zener
    assert res.interface[-1] == pytest.approx(alpha * np.sqrt(D * t1), rel=3e-3)


def test_zener_matrix_profile(zener):
    res, alpha, t1 = zener
    z = 0.5 * (res.z_right[-1, 1:] + res.z_right[-1, :-1])
    exact = NINF + (N1 - NINF) * phi1(z / np.sqrt(D * t1)) / phi1(alpha)
    assert np.max(np.abs(res.x_right[-1, :, 0] - exact)) < 3e-3


def test_zener_conserves_solute(zener):
    res, _, _ = zener
    inv = res.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-12 * inv[0, 0]


def test_zener_error_falls_with_refinement(zener):
    coarse, alpha, t1 = zener
    fine, _, _ = zener_run(1600, 400)
    exact = alpha * np.sqrt(D * t1)
    assert abs(fine.interface[-1] / exact - 1) < 0.6 * abs(coarse.interface[-1] / exact - 1)


# ------------------------------------------- both phases diffusing, constant D

def two_sided_k(dl, dr, xl_inf, xr_inf, xl_i, xr_i):
    """Parabolic constant k of S = k sqrt(t) for a couple welded at z = 0.

    With x_L = xl_inf + (xl_i - xl_inf) erfc(-z / 2 sqrt(D_L t)) / erfc(-k / 2 sqrt(D_L))
    and x_R = xr_inf + (xr_i - xr_inf) erfc(z / 2 sqrt(D_R t)) / erfc(k / 2 sqrt(D_R)),
    the flux balance w (x_R^I - x_L^I) = J_R - J_L with w = k / 2 sqrt(t) becomes
    k/2 (xr_i - xl_i) = (xr_i - xr_inf) sqrt(D_R/pi) exp(-k^2/4D_R) / erfc(k/2 sqrt(D_R))
                      + (xl_i - xl_inf) sqrt(D_L/pi) exp(-k^2/4D_L) / erfc(-k/2 sqrt(D_L)).
    """
    def f(k):
        # exp(-u^2) / erfc(u) = 1 / erfcx(u), stable where both factors underflow
        right = (xr_i - xr_inf) * np.sqrt(dr / np.pi) / erfcx(k / (2 * np.sqrt(dr)))
        left = (xl_i - xl_inf) * np.sqrt(dl / np.pi) / erfcx(-k / (2 * np.sqrt(dl)))
        return k / 2 * (xr_i - xl_i) - right - left
    kmax = 10 * np.sqrt(max(dl, dr))
    return brentq(f, -kmax, kmax)


def test_two_sided_solution_reduces_to_zener():
    """D_L -> 0 with x_L = n0 everywhere is Zener's problem, k = alpha sqrt(D)."""
    k = two_sided_k(1e-22, D, N0, NINF, N0, N1)
    assert k / np.sqrt(D) == pytest.approx(zener_alpha(N0, N1, NINF), rel=1e-6)


def test_two_sided_couple_matches_similarity_solution():
    dl, dr = 2e-15, 1e-14
    xl_inf, xr_inf, xl_i, xr_i = 0.9, 0.1, 0.7, 0.4
    k = two_sided_k(dl, dr, xl_inf, xr_inf, xl_i, xr_i)
    t1 = 1e5
    half = 10 * np.sqrt(dr * t1)
    res = solve_moving_boundary(2 * half, half, np.full(400, xl_inf), np.full(800, xr_inf), dl, dr,
                                FixedInterface([xl_i], [xr_i]), t1, t1 / 800, save_times=[t1 / 4])
    shift = res.interface - half
    assert shift[1] == pytest.approx(k * np.sqrt(t1 / 4), rel=1e-2)
    assert shift[2] == pytest.approx(k * np.sqrt(t1), rel=5e-3)
    inv = res.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-12 * inv[0, 0]


# --------------------------------------------- local equilibrium from pycalphad

TWO_PHASE = HEADER + """
PHASE ALPHA % 1 1 !
CONSTITUENT ALPHA :A,B,C: !
PHASE BETA % 1 1 !
CONSTITUENT BETA :A,B,C: !
PARAMETER G(ALPHA,A;0) 298.15 0; 6000 N !
PARAMETER G(ALPHA,B;0) 298.15 0; 6000 N !
PARAMETER G(ALPHA,C;0) 298.15 0; 6000 N !
PARAMETER G(BETA,A;0) 298.15 5000; 6000 N !
PARAMETER G(BETA,B;0) 298.15 -5000; 6000 N !
PARAMETER G(BETA,C;0) 298.15 1000; 6000 N !
PARAMETER G(ALPHA,A,B;0) 298.15 -2000; 6000 N !
PARAMETER G(BETA,A,B;0) 298.15 1500; 6000 N !
PARAMETER G(ALPHA,A,C;0) 298.15 -1000; 6000 N !
PARAMETER G(BETA,B,C;0) 298.15 2000; 6000 N !
""" + "".join(
    f"PARAMETER MQ({ph}&{el},{host};0) 298.15 {q}+R*T*LN({d0}); 6000 N !\n"
    for ph, base in (("ALPHA", 0), ("BETA", -15000))
    for el, d0 in (("A", "1E-4"), ("B", "2E-4"), ("C", "5E-5"))
    for host, q in (("A", -150000 + base), ("B", -160000 + base), ("C", -155000 + base))
)


@pytest.fixture(scope="module")
def two_phase_db():
    return Database(TWO_PHASE)


def pycalphad_tie_line(db, comps, conds):
    eq = equilibrium(db, comps + ["VA"], ["ALPHA", "BETA"], {v.T: T, v.P: 101325, v.N: 1, **conds})
    phases = list(np.squeeze(eq.Phase.values))
    x = np.squeeze(eq.X.values)
    mu = np.squeeze(eq.MU.values)
    return x[phases.index("ALPHA")], x[phases.index("BETA")], mu


def test_binary_tie_line_matches_pycalphad_equilibrium(two_phase_db):
    alpha = SolutionPhase(two_phase_db, "ALPHA", ["A", "B"])
    beta = SolutionPhase(two_phase_db, "BETA", ["A", "B"])
    xa_ref, xb_ref, mu_ref = pycalphad_tie_line(two_phase_db, ["A", "B"], {v.X("A"): 0.45})
    cond = LocalEquilibrium(alpha, beta, T, [0.6], [0.3])
    xl, xr = cond.tie_line()
    assert xl[0] == pytest.approx(xa_ref[0], abs=1e-6)
    assert xr[0] == pytest.approx(xb_ref[0], abs=1e-6)
    np.testing.assert_allclose(alpha.chemical_potentials([xl[0]], T), mu_ref, rtol=1e-6)
    np.testing.assert_allclose(beta.chemical_potentials([xr[0]], T), mu_ref, rtol=1e-6)


def test_chemical_potentials_obey_gibbs_duhem(two_phase_db):
    alpha = SolutionPhase(two_phase_db, "ALPHA", ["A", "B", "C"])
    x = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    mu = alpha.chemical_potentials(x, T)
    np.testing.assert_allclose((x * mu).sum(axis=1), alpha.gibbs.gm(x, T), rtol=1e-12)


@pytest.fixture(scope="module")
def ternary_couple(two_phase_db):
    """alpha (A-rich) welded to beta (B-rich) with C on both sides."""
    alpha = SolutionPhase(two_phase_db, "ALPHA", ["A", "B", "C"])
    beta = SolutionPhase(two_phase_db, "BETA", ["A", "B", "C"])
    x_alpha, x_beta = np.array([0.75, 0.15]), np.array([0.10, 0.75])
    xa, xb, _ = pycalphad_tie_line(two_phase_db, ["A", "B", "C"], {v.X("A"): 0.45, v.X("C"): 0.1})
    cond = LocalEquilibrium(alpha, beta, T, xa[:2], xb[:2])
    half, t1 = 4e-4, 400.0  # sqrt(D t1) is about 6e-5 m in beta, so the far ends stay untouched
    res = solve_moving_boundary(2 * half, half, np.tile(x_alpha, (200, 1)), np.tile(x_beta, (200, 1)),
                                lambda x: alpha.interdiffusion_matrix(x, T), lambda x: beta.interdiffusion_matrix(x, T),
                                cond, t1, t1 / 200, save_times=[t1 / 4])
    return res, alpha, beta, half


def test_ternary_interface_stays_in_local_equilibrium(ternary_couple):
    res, alpha, beta, _ = ternary_couple
    for xl, xr in zip(res.interface_left[1:], res.interface_right[1:]):
        np.testing.assert_allclose(alpha.chemical_potentials(xl, T), beta.chemical_potentials(xr, T), atol=1e-6 * T)


def test_ternary_operating_tie_line_is_constant_in_time(ternary_couple):
    """In a semi-infinite couple the profiles depend on z/sqrt(t) only, so the
    interface compositions are fixed once diffusion has started; the operating
    tie-line differs from the one through the mean alloy composition."""
    res, *_ = ternary_couple
    np.testing.assert_allclose(res.interface_left[2], res.interface_left[1], atol=1e-3)
    np.testing.assert_allclose(res.interface_right[2], res.interface_right[1], atol=1e-3)
    assert np.max(np.abs(res.interface_left[2] - res.interface_left[0])) > 1e-2


def test_ternary_couple_conserves_every_element(ternary_couple):
    res, *_ = ternary_couple
    inv = res.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-10 * np.abs(inv[0]).max()


def test_ternary_interface_moves_parabolically(ternary_couple):
    res, _, _, half = ternary_couple
    shift = res.interface - half
    assert abs(shift[2]) > 1e-6  # the interface really moved
    assert shift[2] / shift[1] == pytest.approx(2.0, rel=2e-2)


def test_dissolving_phase_is_reported():
    """A thin left layer that the flux balance consumes must stop with a clear error."""
    with pytest.raises(PhaseDissolvedError):
        solve_moving_boundary(1e-4, 2e-6, np.full(10, 0.6), np.full(200, 0.0), 1e-14, 1e-13,
                              FixedInterface([0.6], [0.3]), 1e5, 1e3)


def test_fixed_interface_refuses_ternaries():
    with pytest.raises(ValueError, match="binaries"):
        FixedInterface([0.1, 0.2], [0.3, 0.4])
