"""Cr2O3 scale growth on Cr after Hallstrom, Halvarsson, Hoglund, Jonsson and Agren, Solid State
Ionics 240 (2013) 41, with the Cr-O thermodynamics of Kjellqvist, Selleby and Sundman, Calphad 32
(2008) 577.

Figure readings: pages rendered at 400 dpi with PyMuPDF, axes calibrated on the
frame lines (the darkest full-length rows and columns of each plot). Straight
model lines were fitted over about 1000 columns each (rms 0.3 px, about 0.005
decades); curves were read as the centre of the dark run in single columns.
"""

import functools
import warnings

import numpy as np
import pytest
from scipy.integrate import simpson
from pycalphad import equilibrium, variables as v

from janus_kinetics import R, kjellqvist2008_cro, metal_oxide_potential, parabolic_scale_growth
from janus_kinetics.oxides import (HALLSTROM2013_CHROMIUM, HALLSTROM2013_OXYGEN, Arrhenius, GrainBoundaryMobility,
                                     chromia)

ATM = 1.01325  # bar; the paper's pressures are in atm
DAY = 86400.0


@pytest.fixture(scope="module")
def db():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kjellqvist2008_cro()


@pytest.fixture(scope="module")
def bulk(db):
    return chromia(db)


@functools.lru_cache(maxsize=None)
def _growth(T, grain_size, alpha=0.5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oxide = chromia(kjellqvist2008_cro(), grain_size=grain_size, alpha=alpha)
        return parabolic_scale_growth(oxide, T, ATM, metal_phase="BCC_A2")


# --- the shipped Cr-O file against Kjellqvist et al.'s Table 5 --------------------------------------

def table5(T):
    """Table 5 (pp. 588-589) transcribed again, independently of the .tdb file."""
    ln = np.log(T)
    if T < 1000:
        gcr2o3 = -1177497.8 + 814.9138 * T - 132.046 * T * ln + 0.005256015 * T**2 - 1.38885e-6 * T**3 + 1501761 / T
        go2 = (-6961.74451 - 51.0057202 * T - 22.2710136 * T * ln - 0.0101977469 * T**2 + 1.32369208e-6 * T**3
               - 76729.7484 / T)
    else:
        gcr2o3 = -1166947.9 + 701.5624 * T - 115.5381 * T * ln - 0.00620492 * T**2 + 1.00698e-7 * T**3 + 239949 / T
        go2 = (-13137.5203 + 25.3200332 * T - 33.627603 * T * ln - 0.00119159274 * T**2 + 1.35611111e-8 * T**3
               + 525809.556 / T)
    ghsercr = -8856.94 + 157.48 * T - 26.908 * T * ln + 0.00189435 * T**2 - 1.47721e-6 * T**3 + 139250 / T
    return {
        ("CORUNDUM", "G", ("CR+2", "CR+3", "O-2")): gcr2o3 + 665910,
        ("CORUNDUM", "G", ("CR+3", "CR+3", "O-2")): gcr2o3 - 232227.2 + 241.3793 * T,
        ("CORUNDUM", "G", ("CR+2", "VA", "O-2")): gcr2o3,
        ("CORUNDUM", "G", ("CR+3", "VA", "O-2")): gcr2o3,
        **{("CORUNDUM", "TC", (c, s, "O-2")): -918.0 for c in ("CR+2", "CR+3") for s in ("CR+3", "VA")},
        **{("CORUNDUM", "BMAGN", (c, s, "O-2")): -5.814 for c in ("CR+2", "CR+3") for s in ("CR+3", "VA")},
        ("BCC_A2", "G", ("CR", "VA")): ghsercr,
        ("BCC_A2", "G", ("CR", "O")): ghsercr + 1.5 * go2 + 195 * T,
        ("BCC_A2", "L", ("CR", ("O", "VA"))): -673435 + 27.86 * T,
        ("BCC_A2", "TC", ("CR", "VA")): -311.5,
        ("BCC_A2", "BMAGN", ("CR", "VA")): -0.008,
        ("GAS", "G", ("O2",)): go2,
    }


def test_shipped_cr_o_is_table_5(db):
    """Every parameter of the file, and no other, equals the table at 700, 900 and 1500 K
    (the gas at 1 bar, where its pressure term vanishes)."""
    import symengine as se

    def value(expr, T):
        e = se.sympify(expr)
        for _ in range(8):
            e = e.subs({se.Symbol(k): val for k, val in db.symbols.items()})
        return float(e.subs({s: T for s in e.free_symbols if s.name == "T"} | {s: 1e5 for s in e.free_symbols
                                                                                  if s.name == "P"}))

    found = {}
    for prm in db._parameters.all():
        subl = tuple(tuple(str(x.name) for x in sub) for sub in prm["constituent_array"])
        key = tuple(s[0] if len(s) == 1 else s for s in subl)
        found[(prm["phase_name"], prm["parameter_type"], key)] = prm["parameter"]
    for T in (700.0, 900.0, 1500.0):
        expected = table5(T)
        assert set(found) == set(expected)
        for key, val in expected.items():
            assert value(found[key], T) == pytest.approx(val, rel=1e-12, abs=1e-6), (key, T)


def test_table_5_functions_join_at_their_breakpoints():
    """A typo in a coefficient shows as a step; the printed ones join within rounding."""
    lo, hi = table5(1000 - 1e-9), table5(1000 + 1e-9)
    for key in (("CORUNDUM", "G", ("CR+2", "VA", "O-2")), ("GAS", "G", ("O2",))):
        assert abs(lo[key] - hi[key]) < 3.0
        slope_lo = (lo[key] - table5(1000 - 1e-3)[key]) / 1e-3
        slope_hi = (table5(1000 + 1e-3)[key] - hi[key]) / 1e-3
        assert abs(slope_lo - slope_hi) < 0.01


def test_cr_and_chromia_coexist_at_the_computed_potential(db, bulk):
    """At the metal/oxide oxygen potential the oxide's mu_Cr equals that of bcc Cr, found
    by an equilibrium of the metal alone at the same mu_O."""
    T = 898.15
    mu_o = metal_oxide_potential(db, T, "CR", "BCC_A2", "CORUNDUM", 0.3)
    metal = equilibrium(db, ["CR", "O", "VA"], ["BCC_A2"], {v.T: T, v.P: 1e5, v.N: 1, v.MU("O"): mu_o})
    mu_cr_metal = float(dict(zip(metal.component.values, metal.MU.values.squeeze()))["CR"])
    assert bulk.state_at_potential(T, mu_o)["mu"]["CR"] == pytest.approx(mu_cr_metal, abs=1.0)


def test_coexistence_refuses_a_composition_outside_the_two_phase_field(db):
    """Below the O solubility of Cr only bcc is stable; beyond Cr2O3 pycalphad pairs the
    oxide with bcc filled to x_O = 0.75, which must not pass for the metal side."""
    with pytest.raises(ValueError, match="stable phases"):
        metal_oxide_potential(db, 898.15, "CR", "BCC_A2", "CORUNDUM", 1e-9)
    with pytest.raises(ValueError, match="not the metal side"):
        metal_oxide_potential(db, 898.15, "CR", "BCC_A2", "CORUNDUM", 0.7)


# --- transport coefficients -------------------------------------------------------------------------

def test_chromia_tracer_diffusivities_are_eqs_4_and_7(bulk):
    """D*_Cr = RT (1/3) y''_Va y''_Cr M''_CrVa / n_Cr with eq. (11), D*_O = RT k y''_Cr with eq. (12)."""
    T, p = 1300.0, 1e-6
    s = bulk.state(T, p)
    y = s["y"]
    m_cr = 2.72e-7 * np.exp(-428200 / (R * T))
    k_o = 2.47e-10 * np.exp(-351170 / (R * T))
    assert bulk.tracer_diffusivity(T, p) == pytest.approx(
        R * T / 3 * y[(1, "VA")] * y[(1, "CR+3")] * m_cr / s["n_metal"], rel=1e-12)
    assert bulk.anion_tracer_diffusivity(T, p) == pytest.approx(R * T * k_o * y[(1, "CR+3")], rel=1e-12)


# Fig. 7 (p. 47): frame x 845-2086 px = 1/T 50-100 (1e-5/K), y 362.5-1603.5 px = log D -12 to -32.
# Lines as log10 D*_O = a (1e5/T) + b.
FIG7 = {1e-12: (-0.22696, -5.4223), 0.1: (-0.22866, -7.4249)}
# Fig. 6 (p. 46): frame x 758-1951 px = 50-100, y 360-1553.5 px = -10 to -30.
FIG6 = {5e-5: (-0.25819, -5.3394), 0.1: (-0.25595, -6.1175)}


@pytest.mark.parametrize("T", [1000.0, 1176.0, 1429.0, 1667.0, 1818.0])
@pytest.mark.parametrize("p", sorted(FIG7))
def test_oxygen_tracer_diffusion_against_fig_7(bulk, T, p):
    """Eq. (12) with the Kjellqvist corundum reproduces both lines of Fig. 7 within 0.1
    decades (0.03-0.07 found), so this Cr-O description gives the interstitial Cr
    fraction of the Taylor-Dinsdale one the paper used, over 11 decades of p_O2."""
    a, b = FIG7[p]
    ours = np.log10(bulk.anion_tracer_diffusivity(T, p * ATM))
    assert abs(ours - (a * 1e5 / T + b)) < 0.1


@pytest.mark.parametrize("T", [1000.0, 1250.0, 1818.0])
def test_chromium_tracer_pressure_dependence_against_fig_6(bulk, T):
    """The spacing of the two lines of Fig. 6 (0.55-0.66 decades) matches ours (0.62)
    within 0.07 decades."""
    ours = np.log10(bulk.tracer_diffusivity(T, 5e-5 * ATM) / bulk.tracer_diffusivity(T, 0.1 * ATM))
    theirs = (FIG6[5e-5][0] - FIG6[0.1][0]) * 1e5 / T + FIG6[5e-5][1] - FIG6[0.1][1]
    assert abs(ours - theirs) < 0.07


@pytest.mark.skip(reason="Hallstrom et al. (2013) Fig. 6: with eq. (11) and the thermodynamics that reproduce "
                         "Fig. 7, D*_Cr lies 0.2 decades above their lines at 1000 K and 0.75 at 1818 K, as if "
                         "the plotted lines had a 23 kJ/mol lower activation energy than eq. (11). The defect "
                         "fraction cancels in D*_Cr / D*_O at one p_O2, and their Figs. 6 and 7 give a ratio off "
                         "eqs. (11)-(12) by 0.01 decades at 898 K and 0.79 at 2000 K, so the paper is "
                         "inconsistent there whatever the thermodynamics. "
                         "doi 10.1016/j.ssi.2013.02.017")
def test_chromium_tracer_level_against_fig_6():
    pass


def test_grain_boundary_mobility_is_eq_8():
    bulk = Arrhenius(2e-7, 400e3)
    gb = GrainBoundaryMobility(bulk, grain_size=2e-7, alpha=0.4, width=6e-10)
    T = 950.0
    f = 6e-10 / 2e-7
    expected = f * 2e-7 * np.exp(-0.4 * 400e3 / (R * T)) + (1 - f) * 2e-7 * np.exp(-400e3 / (R * T))
    assert gb(T) == pytest.approx(expected, rel=1e-12)
    assert GrainBoundaryMobility(bulk, grain_size=1.0, alpha=1.0)(T) == pytest.approx(bulk(T), rel=1e-12)


@pytest.mark.parametrize("kwargs", [dict(grain_size=1e-10), dict(grain_size=1e-7, alpha=0.0),
                                    dict(grain_size=1e-7, alpha=1.5), dict(grain_size=1e-7, width=0.0)])
def test_grain_boundary_mobility_refuses_bad_input(kwargs):
    with pytest.raises(ValueError):
        GrainBoundaryMobility(HALLSTROM2013_CHROMIUM, **kwargs)


# --- scale growth -----------------------------------------------------------------------------------

@pytest.mark.parametrize("T, grain, paper", [(898.15, 100e-9, 340e-9), (898.15, 150e-9, 280e-9),
                                             (973.15, 200e-9, 660e-9), (973.15, 250e-9, 590e-9)])
def test_24_h_thickness_against_the_paper(T, grain, paper):
    """Section 4.5.2 (p. 48): with alpha = 1/2, 340 and 280 nm at 625 C for 100 and 150 nm
    grains, 660 and 590 nm at 700 C for 200 and 250 nm. Ours are 345, 282, 658 and 588 nm,
    partly because their curves cross ours near 24 h (see the next two tests)."""
    assert _growth(T, grain).thickness(DAY) == pytest.approx(paper, rel=0.02)


# Fig. 8 (p. 48): frame x 885.5-2017.5 px = 0-9e4 s, y 2750.5-3882.5 px = 0-35e-8 m.
# Fig. 9 (p. 49): frame x 937.5-2033.5 px = 0-9e4 s, y 336.0-1432.5 px = 0-7e-7 m.
CURVES = {
    (898.15, 100e-9): [22.540e-8, 25.477e-8, 28.167e-8, 30.640e-8, 32.959e-8],
    (898.15, 150e-9): [18.350e-8, 20.777e-8, 22.973e-8, 24.998e-8, 26.868e-8],
    (973.15, 200e-9): [4.363e-7, 4.928e-7, 5.449e-7, 5.927e-7, 6.374e-7],
    (973.15, 250e-9): [3.904e-7, 4.411e-7, 4.874e-7, 5.299e-7, 5.701e-7],
}
CURVE_TIMES = [4e4, 5e4, 6e4, 7e4, 8e4]


@pytest.mark.parametrize("case", sorted(CURVES))
def test_thickness_against_figs_8_and_9(case):
    """Between 4e4 and 8e4 s the parabolas run within 5 % of the curves, from 4.5 % high
    at 4e4 s to within 1 % at 8e4 s."""
    g = _growth(*case)
    ratio = g.thickness(np.array(CURVE_TIMES)) / np.array(CURVES[case])
    assert np.all(np.abs(ratio - 1) < 0.05)


@pytest.mark.parametrize("case", sorted(CURVES))
def test_late_growth_rate_against_figs_8_and_9(case):
    """A line through their X^2 against t from 5e4 to 8e4 s gives k 5 % above ours at 625 C
    and 8 % above at 700 C, and meets X = 0 at t0 = 5300-5500 s in all four runs: their
    numerical runs start slower than any parabola, then grow faster. The quasi-steady
    law has no such offset; within 10 % on k is what the comparison supports."""
    t = np.array(CURVE_TIMES[1:])
    slope, intercept = np.polyfit(t, np.array(CURVES[case][1:]) ** 2, 1)
    assert 5000 < -intercept / slope < 6000
    assert _growth(*case).k == pytest.approx(slope / 2, rel=0.10)


def test_alpha_one_third_against_fig_10():
    """Fig. 10 (p. 49; frame x 470.5-1574.5 px = 0-9e4 s, y 2782.5-3885.5 px = 0-40e-6 m):
    alpha = 0.33 gives 39.5 um at 8.6e4 s, a hundred times the alpha = 1/2 scale. The
    value lies between ours for alpha = 1/3 (37.6 um) and alpha = 0.33 (41.4 um)."""
    t = 8.6e4
    assert _growth(898.15, 100e-9, 1 / 3).thickness(t) < 39.53e-6 < _growth(898.15, 100e-9, 0.33).thickness(t)
    assert _growth(898.15, 100e-9, 0.33).thickness(t) / _growth(898.15, 100e-9).thickness(t) > 100


@pytest.mark.skip(reason="Hallstrom et al. (2013) Fig. 11: oxygen transport thickens their scale by 5.2 % "
                         "(625 C, 100 nm), ours by 10.2 %; in the rate constant oxygen carries about 10 % for "
                         "them and 18 % here. That share is set by eqs. (11)-(12) alone, nearly independent of "
                         "the thermodynamics, and the paper does not show where the factor of 2 comes from. "
                         "doi 10.1016/j.ssi.2013.02.017")
def test_oxygen_share_against_fig_11():
    pass


@pytest.mark.slow
def test_rate_constant_by_wagners_form(db):
    """k from the tracer diffusivities, (1/2) int (D*_Cr / n_g + D*_O) d ln p_O2 with n_g the Cr
    per O at the gas side, by the
    Simpson rule on a dense grid, through the public D* methods and pressures."""
    T, grain = 898.15, 100e-9
    g = _growth(T, grain)
    oxide = chromia(db, grain_size=grain)
    ref = oxide.oxygen_potential(T, 1.0)
    ln_p = np.linspace(2 * (g.mu_inner - ref) / (R * T), 2 * (g.mu_outer - ref) / (R * T), 81)
    n_g = oxide.state(T, ATM)["n_metal"]
    vals = []
    for lp in ln_p:
        p = np.exp(lp)
        vals.append(oxide.tracer_diffusivity(T, p) / n_g + oxide.anion_tracer_diffusivity(T, p))
    assert 0.5 * simpson(vals, x=ln_p) == pytest.approx(g.k, rel=1e-3)


@pytest.mark.slow
def test_cation_part_through_mu_cr(db):
    """The cation part integrated over mu_Cr instead of mu_O, (1/n_g) int (1/3) y''_Va y''_Cr M
    (-dmu_Cr), the form of eq. (3) without the Gibbs-Duhem step."""
    T, grain = 973.15, 200e-9
    g = _growth(T, grain)
    oxide = chromia(db, grain_size=grain)
    m = oxide.paths[0].mobility(T)
    mus = np.linspace(g.mu_inner, g.mu_outer, 81)
    states = [oxide.state_at_potential(T, mu) for mu in mus]
    f = np.array([s["y"][(1, "VA")] * s["y"][(1, "CR+3")] * m / 3 for s in states])
    mu_cr = np.array([s["mu"]["CR"] for s in states])
    assert simpson(f, x=-mu_cr) / states[-1]["n_metal"] == pytest.approx(g.cation, rel=1e-3)


def test_scale_stays_close_to_stoichiometry():
    """The quasi-steady law neglects what the scale stores as its composition shifts, an
    error of the order of the change in Cr per O across it: under 2 % here (interstitial
    Cr reaches 3.6 % at the metal side)."""
    lo, hi = _growth(898.15, 100e-9).n_metal_range
    assert lo == pytest.approx(2 / 3, rel=1e-6)
    assert hi / lo - 1 < 0.02


def test_quadrature_has_converged(db):
    oxide = chromia(db, grain_size=100e-9)
    coarse = parabolic_scale_growth(oxide, 898.15, ATM, metal_phase="BCC_A2", nodes=12)
    assert coarse.k == pytest.approx(_growth(898.15, 100e-9).k, rel=1e-4)


def test_fixed_inner_pressure_matches_the_metal_side(db):
    T = 898.15
    g = _growth(T, 100e-9)
    oxide = chromia(db, grain_size=100e-9)
    p_inner = np.exp(2 * (g.mu_inner - oxide.oxygen_potential(T, 1.0)) / (R * T))
    fixed = parabolic_scale_growth(oxide, T, ATM, p_inner=p_inner)
    assert fixed.k == pytest.approx(g.k, rel=1e-6)
    assert fixed.thickness(4 * DAY) == pytest.approx(2 * g.thickness(DAY), rel=1e-6)


def test_without_an_anion_path_only_the_cation_part_remains(db):
    from janus_kinetics import OxideCationDiffusion
    from janus_kinetics.oxides import JumpPath

    T, grain = 898.15, 100e-9
    g = _growth(T, grain)
    cr_only = OxideCationDiffusion(db, "CORUNDUM", [JumpPath(1, GrainBoundaryMobility(HALLSTROM2013_CHROMIUM, grain))],
                                   metal="CR")
    alone = parabolic_scale_growth(cr_only, T, ATM, metal_phase="BCC_A2")
    assert alone.anion == 0.0
    assert alone.k == pytest.approx(g.cation, rel=1e-6)
    assert 0 < g.inward_fraction < 0.5


@pytest.mark.parametrize("kwargs, match", [(dict(), "exactly one"),
                                           (dict(metal_phase="BCC_A2", p_inner=1e-30), "exactly one"),
                                           (dict(p_inner=10.0), "more oxidizing")])
def test_scale_growth_refuses_bad_boundaries(bulk, kwargs, match):
    with pytest.raises(ValueError, match=match):
        parabolic_scale_growth(bulk, 898.15, ATM, **kwargs)


def test_oxygen_mobility_constants_are_eq_12():
    T = 1000.0
    assert HALLSTROM2013_OXYGEN(T) == pytest.approx(2.47e-10 * np.exp(-351170 / (R * T)), rel=1e-12)
    assert HALLSTROM2013_CHROMIUM(T) == pytest.approx(2.72e-7 * np.exp(-428200 / (R * T)), rel=1e-12)
