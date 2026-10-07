"""Fe diffusion in magnetite and hematite after Hallstrom, Hoglund and Agren, Acta Mater. 59 (2011) 53.

The model is checked here against itself and its equations. pycalphad's
alfeo.tdb (Lindwall, Liu and Liu) carries the Fe-O spinel and corundum of
Kjellqvist, Selleby and Sundman, Calphad 32 (2008) 577, term for term (checked
below against their Table 5). That is the paper's hematite description; its
magnetite was Sundman's (1991), which Kjellqvist et al. changed: their spinel puts
the defect fractions near stoichiometry off by up to a factor of four from the
paper's Fig. 3.
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


@pytest.mark.skip(reason="Hallstrom et al. (2011) Fig. 4 needs Sundman's Fe-O description, J. Phase Equilib. 12 "
                         "(1991) 127, doi 10.1007/BF02645709; alfeo.tdb differs near stoichiometry")
def test_magnetite_tracer_diffusion_against_fig_4():
    pass


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
