"""Microsegregation against Kobayashi, J. Cryst. Growth 88 (1988) 87, and the Scheil-Gulliver algorithm of
Bocklund et al., Materialia 11 (2020) 100689, against the Scheil equation for a constant partition ratio."""

import functools
import warnings

import numpy as np
import pytest
from pycalphad import Database
from scipy.integrate import quad

from calphad_kinetics.solidification import (
    brody_flemings,
    clyne_kurz,
    clyne_kurz_beta,
    kobayashi,
    lever_rule,
    scheil,
    scheil_gulliver,
    solve_back_diffusion,
)

K_P, GAMMA_P = 0.13, 0.8  # P in delta-Fe, Kobayashi Table 1 (p. 92)


# --- the closed forms ---------------------------------------------------------------------------------------

def test_brody_flemings_spans_scheil_and_the_lever_rule():
    """Eq. (2) is Scheil (eq. 1) at gamma = 0 and the lever rule (eq. 5) at gamma = 1 (p. 87)."""
    fs = np.linspace(0, 0.95, 20)
    np.testing.assert_allclose(brody_flemings(0.3, 0.0, fs), scheil(0.3, fs), rtol=1e-14)
    np.testing.assert_allclose(brody_flemings(0.3, 1.0, fs), lever_rule(0.3, fs), rtol=1e-12)


def test_clyne_kurz_beta_limits():
    """Eq. (7): beta -> gamma as gamma -> 0 (Brody-Flemings), and beta -> 1 - 1/gamma -> 1 as gamma grows, where
    eq. (6) becomes the lever rule (Kobayashi's gamma is twice Brody and Flemings' alpha, footnote p. 87, so the
    1/2 of the original normalisation reads 1 here)."""
    assert clyne_kurz_beta(0.01) == pytest.approx(0.01, rel=1e-6)
    assert clyne_kurz_beta(1e4) == pytest.approx(1 - 1e-4, rel=1e-6)
    np.testing.assert_allclose(clyne_kurz(0.3, 1e8, np.array([0.2, 0.7])), lever_rule(0.3, np.array([0.2, 0.7])),
                               rtol=1e-6)


def test_approximations_against_kobayashi_at_full_solid():
    """P in delta-Fe (k = 0.13, gamma = 0.8), p. 92: Clyne-Kurz and Ohnaka give C_s/C0 of about 1.8 at f_s = 1."""
    assert clyne_kurz(K_P, GAMMA_P, 1.0) == pytest.approx(1.8, abs=0.02)
    assert clyne_kurz(K_P, GAMMA_P, 1.0, beta=GAMMA_P / (1 + GAMMA_P)) == pytest.approx(1.8, abs=0.02)


def test_exact_series_at_full_solid():
    """The paper's exact value at f_s = 1 is 4.48 (p. 92, Fig. 2). The series (eqs. 39-40) converges there to
    4.5130 by 5000 terms (10000 and 200000 agree to 2e-9); 1000 terms give 4.494 and 400 give 4.31, so the
    printed value is taken as a sum stopped short. Within 1 % of it."""
    exact = kobayashi(K_P, GAMMA_P, 1.0, terms=10000)
    assert exact == pytest.approx(kobayashi(K_P, GAMMA_P, 1.0, terms=20000), rel=2e-9)
    assert exact == pytest.approx(4.5130, abs=1e-4)
    assert exact == pytest.approx(4.48, rel=0.01)


def test_exact_profile_conserves_solute():
    """Eq. (46): at f_s = 1 the solid holds all the solute, int_0^1 C_s dx* = C0; the paper's 0.01-step sum
    gave 0.99944 (Fig. 3), an adaptive quadrature of eq. (38) gives 1 within 1e-6."""
    total = quad(lambda x: float(kobayashi(K_P, GAMMA_P, 1.0, x=np.array([x]))[0]), 0, 1, limit=200)[0]
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("gamma", [0.05, 3.2])
def test_exact_series_recovers_its_limits(gamma):
    """Section 5: slow diffusion approaches Brody-Flemings (eq. 45) and fast diffusion the lever rule plus
    k^2 (1 - k) f_s^2 / (3 gamma [1 - (1 - k) f_s]^3) (eq. 42)."""
    k, fs = 0.6, np.array([0.2, 0.5, 0.8])
    exact = kobayashi(k, gamma, fs)
    if gamma < 1:
        np.testing.assert_allclose(exact, brody_flemings(k, gamma, fs), rtol=0.02)
    else:
        approx = lever_rule(k, fs) + k**2 * (1 - k) * fs**2 / (3 * gamma * (1 - (1 - k) * fs) ** 3)
        np.testing.assert_allclose(exact, approx, rtol=0.02)


# --- the numerical back-diffusion model ---------------------------------------------------------------------

L, TF = 1e-4, 10.0


@functools.lru_cache(maxsize=None)
def _bd(k, gamma, n, steps, geometry="planar"):
    return solve_back_diffusion(k, gamma * L**2 / (2 * TF), L, TF, n_cells=n, steps=steps, geometry=geometry)


def test_back_diffusion_matches_the_exact_solution():
    """gamma = 2 D t_f / L^2 = 0.8 and k = 0.13: the interface solid follows eq. (40) within 0.2 % up to
    f_s = 0.95 and reaches 4.49 at f_s = 1 against 4.513 exact."""
    res = _bd(K_P, GAMMA_P, 400, 16000)
    for f in (0.3, 0.6, 0.9, 0.95):
        i = int(np.argmin(np.abs(res.fs - f)))
        assert res.interface_solid[i] == pytest.approx(kobayashi(K_P, GAMMA_P, res.fs[i]), rel=2e-3)
    assert res.interface_solid[-1] == pytest.approx(4.513, rel=0.01)
    assert res.mean_solid() == pytest.approx(1.0, abs=1e-12)


def test_back_diffusion_error_falls_with_refinement():
    errors = []
    for n, steps in ((100, 1000), (200, 4000)):
        res = _bd(K_P, GAMMA_P, n, steps)
        i = int(np.argmin(np.abs(res.fs - 0.9)))
        errors.append(abs(res.interface_solid[i] / kobayashi(K_P, GAMMA_P, res.fs[i]) - 1))
    assert errors[1] < 0.5 * errors[0]


def test_final_profile_matches_eq_38():
    """At f_s = 1 the solid profile follows eq. (38) (Fig. 3) within 2 % of the local value away from the
    steep last tenth."""
    res = _bd(K_P, GAMMA_P, 400, 16000)
    z = 0.5 * (res.z_faces[1:] + res.z_faces[:-1]) / L
    keep = z < 0.9
    np.testing.assert_allclose(res.solid[keep], kobayashi(K_P, GAMMA_P, 1.0, x=z[keep]), rtol=0.02)


@pytest.mark.parametrize("geometry", ["planar", "cylindrical", "spherical"])
def test_limits_in_every_geometry(geometry):
    """With the fraction solid counted by volume, no diffusion in the solid gives Scheil and fast diffusion the
    lever rule whatever the shape of the volume element."""
    slow = _bd(0.3, 1e-6, 100, 1000, geometry)
    fast = _bd(0.3, 1e3, 100, 1000, geometry)
    for f in (0.5, 0.9):
        i = int(np.argmin(np.abs(slow.fs - f)))
        assert slow.interface_solid[i] == pytest.approx(scheil(0.3, slow.fs[i]), rel=5e-3)
        assert fast.interface_solid[i] == pytest.approx(lever_rule(0.3, fast.fs[i]), rel=5e-3)
    assert slow.mean_solid() == pytest.approx(1.0, abs=1e-12)


def test_back_diffusion_inputs_are_checked():
    with pytest.raises(ValueError, match="partition ratio"):
        solve_back_diffusion(1.5, 1e-12, L, TF)


# --- Scheil-Gulliver from pycalphad -------------------------------------------------------------------------

# Ideal liquid and solid with G_liq(B) - G_sol(B) = R T ln k: the solute partitions as x_s / x_l = k exactly
# at every composition and temperature, so the Scheil equation holds exactly for the liquid.
CONSTANT_K = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT A BLANK 1 0 0 !
ELEMENT B BLANK 1 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE LIQUID % 1 1 !
CONSTITUENT LIQUID :A,B: !
PARAMETER G(LIQUID,A;0) 298.15 10000-10*T; 6000 N !
PARAMETER G(LIQUID,B;0) 298.15 R*T*LN(0.2); 6000 N !
PHASE FCC % 1 1 !
CONSTITUENT FCC :A,B: !
PARAMETER G(FCC,A;0) 298.15 0; 6000 N !
PARAMETER G(FCC,B;0) 298.15 0; 6000 N !
"""


@functools.lru_cache(maxsize=None)
def _scheil(step):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scheil_gulliver(Database(CONSTANT_K), ["B", "A"], ["LIQUID", "FCC"], [0.05], 1005.0, step, t_min=800.0)


def test_scheil_gulliver_follows_the_scheil_equation():
    """The liquid follows x_L = x0 (1 - f_s)^(k - 1) with k = 0.2, within 1.5 % at 4 K steps, and the error
    falls about in proportion to the step."""
    errs = []
    for step in (8.0, 4.0):
        r = _scheil(step)
        exact = 0.05 * (1 - r.fraction_solid) ** (0.2 - 1)
        errs.append(np.max(np.abs(r.liquid_composition[:, 0] / exact - 1)))
    assert errs[1] < 0.015
    assert 1.6 < errs[0] / errs[1] < 2.4


def test_scheil_gulliver_keeps_the_solute():
    """Solid set aside plus the remaining liquid hold the alloy's B."""
    r = _scheil(4.0)
    solid_b = sum(a * x[0] for step in r.solids for _, a, x in step)
    assert solid_b + (1 - r.fraction_solid[-1]) * r.liquid_composition[-1, 0] == pytest.approx(0.05, rel=1e-6)
    assert r.solid_phase_fractions()["FCC"] == pytest.approx(r.fraction_solid[-1], rel=1e-9)


# --- homogenization of a segregation profile ----------------------------------------------------------------
from scipy.special import j0, jn_zeros  # noqa: E402

from calphad_kinetics import Boundary, solve_planar  # noqa: E402
from calphad_kinetics.solidification import segregation_index, segregation_profile  # noqa: E402
from calphad_kinetics.solver import cell_measures  # noqa: E402


def _slowest_mode(geometry):
    """With no flux at r = 0 and r = L the slowest mode of the diffusion equation and its decay constant
    alpha (amplitude exp(-alpha^2 D t / L^2)): cos(pi r / L) in a plate; J0(a r / L) with J0'(a) = -J1(a) = 0,
    a = 3.8317, in a cylinder; sin(a r / L) / (a r / L) with tan(a) = a, a = 4.4934, in a sphere."""
    if geometry == "planar":
        return np.pi, lambda u: np.cos(np.pi * u)
    if geometry == "cylindrical":
        a = jn_zeros(1, 1)[0]
        return a, lambda u: j0(a * u)
    a = 4.493409457909064
    return a, lambda u: np.sinc(a * u / np.pi)


@pytest.mark.parametrize("geometry", ["planar", "cylindrical", "spherical"])
def test_homogenization_follows_the_slowest_mode(geometry):
    """A profile in the slowest mode keeps its shape and its segregation index falls as exp(-alpha^2 D t / L^2)
    within 1 %, with the solute conserved, in each geometry."""
    alpha, mode = _slowest_mode(geometry)
    length, d, n = 50e-6, 1e-14, 300
    faces = np.linspace(0.0, length, n + 1)
    u = 0.5 * (faces[1:] + faces[:-1]) / length
    x0 = 0.10 + 0.02 * mode(u)
    tau = length**2 / (alpha**2 * d)
    times = np.array([0.25, 0.5, 1.0]) * tau
    res = solve_planar(faces, x0, np.array([[d]]), times[-1], tau / 400, save_times=times, geometry=geometry)
    index = segregation_index(res.x[:, :, 0])
    np.testing.assert_allclose(index[1:], np.exp(-times / tau), rtol=0.01)
    inv = res.inventory()[:, 0]
    assert abs(inv[-1] - inv[0]) < 1e-11 * inv[0]


@pytest.mark.parametrize("geometry", ["planar", "cylindrical", "spherical"])
def test_segregation_profile_keeps_the_solute(geometry):
    """Laid out across the element, a Scheil record keeps exactly the solute it holds (its increments, plus the
    unrecorded last 0.1 % at the last composition), and the solid nearest the spine is the first to form. The
    record does not hold all the alloy's solute: the Scheil curve diverges in the last liquid."""
    k, x0 = 0.3, 0.05
    edges = np.linspace(0.0, 0.999, 2001)
    mid = 0.5 * (edges[1:] + edges[:-1])
    comp = x0 * scheil(k, mid)
    faces, prof = segregation_profile(edges, comp, 1e-4, 200, geometry)
    w = cell_measures(faces, geometry)
    held = np.diff(edges) @ comp + (1 - edges[-1]) * comp[-1]
    assert (w @ prof[:, 0]) / w.sum() == pytest.approx(held, rel=1e-12)
    assert prof[0, 0] == pytest.approx(k * x0, rel=0.01)
    assert np.all(np.diff(prof[:, 0]) >= -1e-15)  # cells within one increment share its value


def test_back_diffusion_profile_homogenizes():
    """The end of solidification (k = 0.13, gamma = 0.8) annealed in the same plate loses its segregation
    and conserves the solute; after one decay time of the slowest mode the index is below exp(-1)."""
    res = _bd(K_P, GAMMA_P, 200, 4000)
    d = 1e-14
    tau = L**2 / (np.pi**2 * d)
    ann = solve_planar(res.z_faces, res.solid, np.array([[d]]), tau, tau / 200, save_times=[tau / 4, tau])
    index = segregation_index(ann.x[:, :, 0])
    assert index[-1] < np.exp(-1)
    inv = ann.inventory()[:, 0]
    assert abs(inv[-1] - inv[0]) < 1e-11 * inv[0]
