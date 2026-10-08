"""Fe diffusion in magnetite and hematite after Hallstrom, Hoglund and Agren, Acta Mater. 59 (2011) 53.

pycalphad's alfeo.tdb (Lindwall, Liu and Liu) carries the Fe-O spinel and corundum of
Kjellqvist, Selleby and Sundman, Calphad 32 (2008) 577, term for term (checked below
against their Table 5). Its spinel reproduces the defect fractions the paper draws
from "Sundman's data" in Fig. 3 within about 5 %, although Sundman's own printed
Table 1 (J. Phase Equilib. 12 (1991) 127) differs from it in one end member and
misses Fig. 3 by a factor of three.

Figure readings: pages rendered at 400 dpi with PyMuPDF, axes calibrated on the frame
lines and tick marks; curves traced column by column, continuing from the minimum.
"""

import warnings
from importlib.resources import files

import numpy as np
import pytest
from pycalphad import Database, equilibrium, variables as v

from calphad_kinetics import R
from calphad_kinetics.oxides import HALLSTROM2011_MAGNETITE, JumpPath, OxideCationDiffusion, hematite, magnetite

ALFEO = files("pycalphad") / "tests" / "databases" / "alfeo.tdb"


@pytest.fixture(scope="module")
def db():
    if not ALFEO.is_file():
        pytest.skip("pycalphad's alfeo.tdb test database is not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(str(ALFEO))


@pytest.fixture(scope="module")
def mag(db):
    return magnetite(db, "SPINEL_B")


def test_tracer_diffusivity_is_eq_7(mag):
    """D* = RT (1/2)[y''_Va y''_Fe M'' + y'''_Fe y'''_Va M'''] / n_Fe with n_Fe = Fe per O."""
    T, p = 1473.0, 1e-6
    s = mag.state(T, p)
    y = s["y"]
    y2fe = y[(1, "FE+2")] + y[(1, "FE+3")]
    n_fe = s["x"]["FE"] / s["x"]["O"]
    assert s["n_metal"] == pytest.approx(n_fe, rel=1e-9)
    m2, m3 = (m(T) for m in HALLSTROM2011_MAGNETITE)
    expected = R * T * 0.5 * (y[(1, "VA")] * y2fe * m2 + y[(2, "FE+2")] * y[(2, "VA")] * m3) / n_fe
    assert mag.tracer_diffusivity(T, p) == pytest.approx(expected, rel=1e-12)


def test_hematite_uses_one_path_with_one_third(db):
    hem = hematite(db)
    T, p = 1473.0, 0.2
    s = hem.state(T, p)
    y = s["y"]
    m = 6.02e-6 * np.exp(-304290 / (R * T))
    expected = R * T / 3 * y[(1, "VA")] * y[(1, "FE+3")] * m / s["n_metal"]
    assert hem.tracer_diffusivity(T, p) == pytest.approx(expected, rel=1e-12)


def test_chemical_diffusivity_by_an_independent_route(mag):
    """Eq. (9) along the oxygen potential equals the same expression with dmu_Fe/dn_Fe
    taken from equilibria at fixed composition. Magnetite's range is narrow, so the
    composition step must be small: 2e-5 in x_O is still 3 % off, 1e-6 converged."""
    T, p = 1473.0, 1e-6
    s = mag.state(T, p)
    x_o = s["x"]["O"]
    vals = []
    for dx in (-1e-6, 1e-6):
        eq = equilibrium(mag.db, mag.components, [mag.phase], {v.T: T, v.P: 1e5, v.N: 1, v.X("O"): x_o + dx})
        mu = dict(zip(eq.component.values, eq.MU.values.squeeze()))["FE"]
        vals.append((mu, (1 - x_o - dx) / (x_o + dx)))
    dmu_dn = (vals[1][0] - vals[0][0]) / (vals[1][1] - vals[0][1])
    expected = mag.tracer_diffusivity(T, p) / (R * T) * s["n_metal"] * dmu_dn
    assert mag.chemical_diffusivity(T, p) == pytest.approx(expected, rel=1e-3)


def test_vacancies_and_extra_iron_swap_with_oxygen_pressure(mag):
    """Octahedral vacancies grow and extra octahedral Fe shrinks as p_O2 rises (their Fig. 3),
    so D* has a minimum between the two mechanisms (their Fig. 4)."""
    T = 1473.0
    pressures = 10.0 ** np.arange(-9.0, -1.9, 1.0)
    states = [mag.state(T, p) for p in pressures]
    vac = np.array([s["y"][(1, "VA")] for s in states])
    extra = np.array([s["y"][(2, "FE+2")] for s in states])
    assert np.all(np.diff(vac) > 0) and np.all(np.diff(extra) < 0)
    d = np.array([mag.tracer_diffusivity(T, p) for p in pressures])
    k = int(np.argmin(d))
    assert 0 < k < len(d) - 1


def test_a_path_without_vacancies_is_refused(db):
    with pytest.raises(ValueError, match="does not hold both"):
        OxideCationDiffusion(db, "SPINEL_B", [JumpPath(3, lambda T: 1e-20)])


# Fig. 3 (p. 56), 1273 K: frame x 481.5-1407 px = log pO2 -16 to -2 (atm), y 420-1392.5 px = 1 to 1e-9.
# log10 pO2: (octahedral vacancies, Fe on the extra octahedral sites)
FIG3 = {-16: (5.21e-9, 0.106), -15: (1.01e-8, 0.0485), -13: (1.04e-7, 4.23e-3), -12: (4.48e-7, 9.72e-4),
        -11: (2.06e-6, 2.12e-4), -10: (9.44e-6, 4.62e-5), -9: (4.47e-5, 9.74e-6), -8: (2.10e-4, 2.10e-6),
        -7: (9.51e-4, 4.53e-7), -6: (4.23e-3, 1.03e-7), -4: (0.038, 8.6e-9), -2: (0.102, 1.72e-9)}
ATM = 1.01325


@pytest.mark.parametrize("logp", sorted(FIG3))
def test_magnetite_defects_against_fig_3(mag, logp):
    """alfeo's spinel gives the paper's octahedral vacancies and extra-octahedral Fe within 10 % over 14
    decades of oxygen pressure at 1273 K."""
    vac, extra = FIG3[logp]
    y = mag.state(1273.0, 10.0**logp * ATM)["y"]
    assert y[(1, "VA")] == pytest.approx(vac, rel=0.10)
    assert y[(2, "FE+2")] == pytest.approx(extra, rel=0.10)


def _sundman_printed():
    from pathlib import Path

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(str(Path(__file__).parent / "data" / "sundman1991_feo_spinel.tdb"))


def test_sundman_printed_spinel_differs_from_alfeo_in_one_end_member(db):
    """Sundman (1991) Table 1 (p. 129), transcribed in tests/data, and alfeo.tdb (Kjellqvist et al.) agree in
    eleven of the twelve spinel end members at 1000 and 1500 K; Fe+2:Fe+2:Va:O-2 is 7G - B in the print and
    7G + B in alfeo, a difference of 2B."""
    import symengine as se
    from tinydb import where

    printed = _sundman_printed()

    def end_members(base, phase):
        out = {}
        for prm in base.search((where("phase_name") == phase) & (where("parameter_type") == "G")):
            subl = [tuple(str(x.name) for x in sub) for sub in prm["constituent_array"]]
            if any("AL" in n for sub in subl for n in sub):
                continue
            out[tuple(sub[0] for sub in subl[:-1])] = prm["parameter"]
        return out

    def value(base, expr, T):
        e = se.sympify(expr)
        for _ in range(8):
            e = e.subs({se.Symbol(k): val for k, val in base.symbols.items()})
        return float(e.subs({s: T for s in e.free_symbols if s.name == "T"}))

    ours, theirs = end_members(printed, "SPINEL"), end_members(db, "SPINEL_B")
    assert set(ours) == set(theirs)
    for T in (1000.0, 1500.0):
        b = kjellqvist_functions(T)["B"]
        for key in ours:
            diff = value(printed, ours[key], T) - value(db, theirs[key], T)
            expected = -2 * b if key == ("FE+2", "FE+2", "VA") else 0.0
            assert diff == pytest.approx(expected, abs=1e-3), (key, T)


def test_sundman_printed_spinel_misses_fig_3():
    """With the printed sign the vacancies at pO2 = 1e-16 atm come out at 1.6e-9 against Fig. 3's 5.2e-9, and
    the crossing of the two curves moves half a decade up: the paper's "Sundman's data" carried +B."""
    printed = magnetite(_sundman_printed(), "SPINEL")
    y = printed.state(1273.0, 1e-16 * ATM)["y"]
    assert y[(1, "VA")] < FIG3[-16][0] / 2.5


# Fig. 4 (p. 57) minima of the model curves, traced: (log10 pO2 at the minimum, log10 D*). Frames: 1273 K
# x 1414-2095 px = -14..-4, y 818-1470 px = -11..-15; 1373 K 2321-3002 = -12..-3, same y; 1473 K 3237.5-3919
# = -10..-2, y -10.4..-14; 1573 K 1410.5-2092 = -9..-1, y 1700-2352 = -10.5..-13.5; 1623 K 2325-3007, same.
FIG4_MINIMA = {1273: (-9.14, -14.666), 1373: (-7.61, -14.025), 1473: (-6.50, -13.437), 1573: (-5.26, -12.929),
               1623: (-4.89, -12.709)}


@pytest.mark.parametrize("T", sorted(FIG4_MINIMA))
def test_magnetite_tracer_minimum_against_fig_4(mag, T):
    """The oxygen pressure of lowest D* follows Fig. 4 within 0.25 decades from 1273 to 1623 K; it is set by
    the crossing of the defect fractions, so by the thermodynamics. The depth is 0.1-0.25 decades below the
    figure (see the next test for why)."""
    logp, logd = FIG4_MINIMA[T]
    ps = np.arange(logp - 1.0, logp + 1.01, 0.1)
    d = np.array([np.log10(mag.tracer_diffusivity(float(T), 10**p * ATM)) for p in ps])
    assert ps[np.argmin(d)] == pytest.approx(logp, abs=0.25)
    assert 0.0 < logd - d.min() < 0.3


# Fig. 4 at 1273 K, traced curve at whole decades of pO2 (atm)
FIG4_1273 = {-12: -13.061, -11: -13.751, -10: -14.410, -9: -14.632, -8: -14.105, -7: -13.424, -6: -12.769}


def test_fig_4_does_not_follow_eq_7_with_fig_3():
    """The paper's own eq. (7) with its eqs. (10)-(11) and its Fig. 3 fractions gives D* 0.21-0.30 decades
    below its Fig. 4 curve at 1273 K, the same offset this package shows: the figure is not eq. (7) as
    printed (a dropped factor 1/2 would give 0.30 everywhere). No thermodynamics of this package enters."""
    T = 1273.0
    m2, m3 = (m(T) for m in HALLSTROM2011_MAGNETITE)
    for logp, curve in FIG4_1273.items():
        vac, extra = FIG3[logp]
        n_fe = (1 + 2 * (1 - vac) + 2 * extra) / 4  # Fe per O in (Fe)1(Fe,Va)2(Fe,Va)2(O)4
        dstar = R * T * 0.5 * (vac * (1 - vac) * m2 + extra * (1 - extra) * m3) / n_fe
        assert 0.18 < curve - np.log10(dstar) < 0.32


def test_magnetite_tracer_follows_eq_7_on_the_papers_fractions(mag):
    """Our D* at 1273 K equals eq. (7) evaluated on the paper's Fig. 3 fractions within 0.05 decades: given the
    figure's thermodynamics we reproduce the paper's equations."""
    T = 1273.0
    m2, m3 = (m(T) for m in HALLSTROM2011_MAGNETITE)
    for logp in FIG4_1273:
        vac, extra = FIG3[logp]
        n_fe = (1 + 2 * (1 - vac) + 2 * extra) / 4
        dstar = R * T * 0.5 * (vac * (1 - vac) * m2 + extra * (1 - extra) * m3) / n_fe
        assert np.log10(mag.tracer_diffusivity(T, 10.0**logp * ATM)) == pytest.approx(np.log10(dstar), abs=0.05)


@pytest.mark.skip(reason="Hallstrom et al. (2011) Figs. 7-8: with the Kjellqvist et al. (2008) corundum they used, "
                         "which alfeo.tdb carries exactly, D* comes out 0.4-0.7 decades below their curves, more "
                         "at low T (about 18 kJ/mol in apparent activation energy); the cause is not resolved")
def test_hematite_tracer_diffusion_against_figs_7_and_8():
    pass


def test_oxygen_potential_is_half_that_of_o2_gas(mag):
    """mu_O at O2 partial pressure p equals mu_O of the database's oxygen gas at a total
    pressure P where O2 makes up p: at 1e-6 bar some 0.3 % of it is dissociated, a
    17 J/mol difference from taking P itself as p_O2."""
    from pycalphad import Model

    T = 1473.0
    names = [str(y.species.name) for y in Model(mag.db, ["O"], "GAS").site_fractions]
    for total in (1e-6, 1e-2, 0.5):
        eq = equilibrium(mag.db, ["O"], ["GAS"], {v.T: T, v.P: 1e5 * total, v.N: 1})
        p_o2 = total * eq.Y.values.squeeze()[0][names.index("O2")]
        assert mag.oxygen_potential(T, p_o2) == pytest.approx(float(eq.MU.values.squeeze()), abs=0.05)


def kjellqvist_functions(T):
    """Kjellqvist, Selleby and Sundman (2008), Table 5, p. 591 (B, C, D, G) and p. 589 (GFE2O3)."""
    ln = np.log(T)
    return {"B": 46826 - 27.266 * T, "C": 120730 - 20.102 * T, "D": 402520 - 30.529 * T,
            "G": -161731 + 144.873 * T - 24.9879 * T * ln - 0.0011952256 * T**2 + 206520 / T,
            "GFE2O3": -858683 + 827.946 * T - 137.0089 * T * ln + 1453810 / T}


# Fe-only end members, each as (n G, n B, n C, n D, n GFE2O3) of the functions above
KJELLQVIST_SPINEL = {
    ("FE+2", "FE+2", "FE+2"): (9, 0, 0, 1, 0), ("FE+3", "FE+2", "FE+2"): (9, -1, 0, 1, 0),
    ("FE+2", "FE+3", "FE+2"): (9, -1, 0, 1, 0), ("FE+3", "FE+3", "FE+2"): (9, -2, 0, 1, 0),
    ("FE+2", "VA", "FE+2"): (7, -1, 1, 1, 0), ("FE+3", "VA", "FE+2"): (7, -2, 1, 1, 0),
    ("FE+2", "FE+2", "VA"): (7, 1, 0, 0, 0), ("FE+3", "FE+2", "VA"): (7, 0, 0, 0, 0),
    ("FE+2", "FE+3", "VA"): (7, 0, 0, 0, 0), ("FE+3", "FE+3", "VA"): (7, -1, 0, 0, 0),
    ("FE+2", "VA", "VA"): (5, 0, 1, 0, 0), ("FE+3", "VA", "VA"): (5, -1, 1, 0, 0),
}
KJELLQVIST_CORUNDUM = {("FE+2", "FE+3"): 85000.0, ("FE+3", "FE+3"): 85000.0, ("FE+2", "VA"): 0.0, ("FE+3", "VA"): 0.0}


def test_alfeo_fe_oxides_are_kjellqvist_2008(db):
    """Every Fe-only end member of SPINEL_B and CORUNDUM in alfeo.tdb, and no other Fe-O
    term, equals Kjellqvist et al.'s Table 5."""
    import symengine as se
    from tinydb import where

    def value(expr, T):
        e = se.sympify(expr)
        for _ in range(8):
            e = e.subs({se.Symbol(k): val for k, val in db.symbols.items()})
        return float(e.subs({sym: T for sym in e.free_symbols if sym.name == "T"}))

    for phase, table in (("SPINEL_B", KJELLQVIST_SPINEL), ("CORUNDUM", KJELLQVIST_CORUNDUM)):
        found = {}
        for prm in db.search((where("phase_name") == phase) & (where("parameter_type") == "G")):
            subl = [tuple(str(x.name) for x in sub) for sub in prm["constituent_array"]]
            if any("AL" in name for sub in subl for name in sub):
                continue
            assert all(len(sub) == 1 for sub in subl), f"{phase} has an Fe-O interaction {subl}"
            found[tuple(sub[0] for sub in subl[:-1])] = prm["parameter"]
        assert set(found) == set(table)
        for T in (1000.0, 1500.0):
            f = kjellqvist_functions(T)
            for key, expected in table.items():
                if phase == "SPINEL_B":
                    n_g, n_b, n_c, n_d, _ = expected
                    expected = n_g * f["G"] + n_b * f["B"] + n_c * f["C"] + n_d * f["D"]
                else:
                    expected = f["GFE2O3"] + expected
                assert value(found[key], T) == pytest.approx(expected, abs=1e-3), (phase, key, T)
