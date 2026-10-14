"""Multicomponent precipitation after Svoboda, Fischer, Fratzl and Kozeschnik (Mater. Sci. Eng. A 385
(2004) 166 and 157), checked against gamma' (L1_2) precipitation in Ni-5.2Al-14.2Cr at.% at 873 K:
Sudbrack, Yoon, Noebe and Seidman, Acta Mater. 54 (2006) 3199 (APT kinetics, Tables 1-2), and
Sudbrack, Noebe and Seidman, Acta Mater. 55 (2007) 119 (Thermo-Calc with the Dupin et al. database,
Tables 3-4; interfacial energy and intrinsic diffusivities, Tables 5-6).

The thermodynamics is pycalphad's alcrni.tdb, which holds the Al-Cr-Ni description of Dupin,
Ansara and Sundman, Calphad 25 (2001) 279; its fcc is pinned to their appendix in
test_campbell2002.py and its L1_2 here. The mobilities are Campbell, Boettinger and Kattner (2002).
"""

import functools
import warnings
from importlib.resources import files

import numpy as np
import pytest
from pycalphad import Database, equilibrium, variables as v

from janus_kinetics import R, SolutionPhase, campbell2002_fcc_ni
from janus_kinetics.multicomponent_precipitation import (
    K_B,
    N_A,
    NucleationSettings,
    Population,
    PrecipitateThermo,
    grain_structure,
    growth_rates,
    nucleation,
    nucleation_sites,
    solve_competing_precipitation,
    solve_multicomponent_precipitation,
)

ALCRNI = files("pycalphad") / "tests" / "databases" / "alcrni.tdb"
ELEMENTS = ["AL", "CR", "NI"]
T = 873.0
ALLOY = np.array([0.052, 0.142, 0.806])
# Sudbrack et al. (2007) section 4.1: V_m = N_A a^3 / 4 with a = 0.3554 nm; Table 5 / text: 22.5 mJ/m^2
SETTINGS = NucleationSettings(gamma=0.0225, molar_volume=6.7584e-6, lattice_parameter=0.3552e-9)
HOURS = np.array([0.167, 0.25, 1, 4, 16, 64, 256, 1024])


@functools.lru_cache(maxsize=None)
def _db():
    if not ALCRNI.is_file():
        pytest.skip("pycalphad's alcrni.tdb test database is not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(str(ALCRNI))


@functools.lru_cache(maxsize=None)
def _matrix():
    # the ferromagnetic-factor warning concerns Fe-rich alloys below Tc; Ni-14Cr at 873 K is well above it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SolutionPhase(_db(), "FCC_A1", ELEMENTS, mobility_db=campbell2002_fcc_ni())


@functools.lru_cache(maxsize=None)
def _precipitate():
    return PrecipitateThermo(_db(), "L12_FCC", ELEMENTS)


@functools.lru_cache(maxsize=None)
def _sudbrack_run():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_multicomponent_precipitation(_matrix(), _precipitate(), ALLOY, T, HOURS[-1] * 3600, SETTINGS,
                                                  HOURS * 3600)


def _mu_and_tracer(x):
    m = _matrix()
    return np.asarray(m.chemical_potentials(x, T), dtype=float), R * T * np.asarray(m.mobility(x, T), dtype=float)


# --- the L1_2 of alcrni.tdb against Dupin et al. (2001) ------------------------------------------------

def dupin_l12_parameters(T):
    """Two-sublattice L1_2 parameters from the four-sublattice ones of the appendix (p. 297) through the
    paper's relations (12)-(28), p. 284, with compound energies by eq. (29): U1_AlNi = (2/3)(-22213 + 4.396 T),
    U1_CrNi = -1980, U1_AlCr = -830, Delta G_AlCrNi2 raised by 6650, L1_AlNi = 7204 - 3.743 T, other L = 0."""
    u = {("AL", "NI"): 2 / 3 * (-22213 + 4.396 * T), ("CR", "NI"): -1980.0, ("AL", "CR"): -830.0}
    l1 = {("AL", "NI"): 7204 - 3.743 * T}

    def U(a, b):
        return 0.0 if a == b else u[tuple(sorted((a, b)))]

    def dg(*atoms):  # Delta G of the four-site compound, eq. (29)
        g = sum(U(atoms[i], atoms[j]) for i in range(4) for j in range(i + 1, 4))
        return g + (6650.0 if sorted(atoms) == ["AL", "CR", "NI", "NI"] else 0.0)

    def L1(a, b):
        return l1.get((a, b), 0.0)

    p = {}
    for a, b in (("AL", "CR"), ("AL", "NI"), ("CR", "NI")):
        a3b, a2b2, ab3 = dg(a, a, a, b), dg(a, a, b, b), dg(a, b, b, b)
        p[(b, a, 0)] = ab3  # eq. (12), G(B:A)
        p[(a, b, 0)] = a3b  # eq. (13), G(A:B)
        p[((a, b), a, 0)] = -1.5 * ab3 + 1.5 * a2b2 + 1.5 * a3b  # (14)
        p[((a, b), b, 0)] = 1.5 * ab3 + 1.5 * a2b2 - 1.5 * a3b  # (15)
        p[((a, b), a, 1)] = 0.5 * ab3 - 1.5 * a2b2 + 1.5 * a3b + 3 * L1(a, b)  # (16)
        p[((a, b), b, 1)] = -1.5 * ab3 + 1.5 * a2b2 - 0.5 * a3b + 3 * L1(a, b)  # (17)
        for i in ELEMENTS:
            p[(i, (a, b), 0)] = 0.0  # (18), L0 = 0
            p[(i, (a, b), 1)] = L1(a, b)  # (19)
    al3ni, cr3ni, alcr2ni, al2crni = dg(*"AL AL AL NI".split()), dg(*"CR CR CR NI".split()), \
        dg(*"AL CR CR NI".split()), dg(*"AL AL CR NI".split())
    al3cr, crni3, alcrni2 = dg(*"AL AL AL CR".split()), dg(*"CR NI NI NI".split()), dg(*"AL CR NI NI".split())
    alcr3, alni3 = dg(*"AL CR CR CR".split()), dg(*"AL NI NI NI".split())
    al2cr2, al2ni2, cr2ni2 = dg(*"AL AL CR CR".split()), dg(*"AL AL NI NI".split()), dg(*"CR CR NI NI".split())
    p[(("AL", "CR"), "NI", 0)] = -1.5 * al3ni - 1.5 * cr3ni + 1.5 * alcr2ni + 1.5 * al2crni  # (20)
    p[(("AL", "NI"), "CR", 0)] = -1.5 * al3cr - 1.5 * crni3 + 1.5 * alcrni2 + 1.5 * al2crni  # (21)
    p[(("CR", "NI"), "AL", 0)] = -1.5 * alcr3 - 1.5 * alni3 + 1.5 * alcrni2 + 1.5 * alcr2ni  # (22)
    p[(("AL", "CR"), "NI", 1)] = -0.5 * al3ni + 0.5 * cr3ni - 1.5 * alcr2ni + 1.5 * al2crni  # (23)
    p[(("AL", "NI"), "CR", 1)] = -0.5 * al3cr + 0.5 * crni3 - 1.5 * alcrni2 + 1.5 * al2crni + 3 * L1("AL", "NI")  # (24)
    p[(("CR", "NI"), "AL", 1)] = -0.5 * alcr3 + 0.5 * alni3 - 1.5 * alcrni2 + 1.5 * alcr2ni  # (25)
    p[(("AL", "CR", "NI"), "AL", 0)] = (alcr3 - 1.5 * al2cr2 - 1.5 * al3cr + alni3 - 1.5 * al2ni2 - 1.5 * al3ni
                                        - 1.5 * alcrni2 - 1.5 * alcr2ni + 6 * al2crni)  # (26)
    p[(("AL", "CR", "NI"), "CR", 0)] = (-1.5 * alcr3 - 1.5 * al2cr2 + al3cr + crni3 - 1.5 * cr2ni2 - 1.5 * cr3ni
                                        - 1.5 * alcrni2 + 6 * alcr2ni - 1.5 * al2crni)  # (27)
    p[(("AL", "CR", "NI"), "NI", 0)] = (-1.5 * alni3 - 1.5 * al2ni2 + al3ni - 1.5 * crni3 - 1.5 * cr2ni2 + cr3ni
                                        + 6 * alcrni2 - 1.5 * alcr2ni - 1.5 * al2crni)  # (28)
    for i in ELEMENTS:
        p[(i, i, 0)] = 0.0
    return p


def test_alcrni_l12_is_dupin_2001():
    """Every L1_2 ordering parameter of alcrni.tdb equals the value relations (12)-(28) give from Dupin's
    four-sublattice parameters; parameters the relations make zero may be missing from the file."""
    import symengine as se
    from tinydb import where

    db = _db()

    def value(expr, T):
        e = se.sympify(expr)
        for _ in range(8):
            e = e.subs({se.Symbol(k): val for k, val in db.symbols.items()})
        return float(e.subs({s: T for s in e.free_symbols if s.name == "T"}))

    found = {}
    for prm in db.search(where("phase_name") == "L12_FCC"):
        subl = [tuple(sorted(str(s.name) for s in sub)) for sub in prm["constituent_array"]]
        key = tuple(s[0] if len(s) == 1 else s for s in subl) + (prm["parameter_order"],)
        found[key] = prm["parameter"]
    for temp in (873.0, 1400.0):
        expected = dupin_l12_parameters(temp)
        assert set(found) <= set(expected), set(found) - set(expected)
        for key, val in expected.items():
            ours = value(found[key], temp) if key in found else 0.0
            assert ours == pytest.approx(val, abs=1e-6), (key, temp)


# --- thermodynamics against Thermo-Calc with the same database --------------------------------------

def test_equilibrium_matches_thermo_calc_with_dupin():
    """Sudbrack et al. (2007) Table 3, row [28]: gamma 3.93 Al / 14.90 Cr, gamma' 14.55 Al / 9.57 Cr at.%
    (that row's Ni, 75.59, leaves the gamma' summing to 99.71 %, so only Al and Cr are compared)."""
    eq = equilibrium(_db(), ELEMENTS + ["VA"], ["FCC_A1", "L12_FCC"],
                     {v.T: T, v.P: 101325, v.N: 1, v.X("AL"): 0.052, v.X("CR"): 0.142})
    names = [str(p) for p in np.ravel(eq.Phase.values)]
    x = eq.X.values.reshape(len(names), -1)
    comps = sorted((x[i][:2] for i, n in enumerate(names) if n), key=lambda c: c[0])
    np.testing.assert_allclose(100 * comps[0], [3.93, 14.90], atol=0.1)
    np.testing.assert_allclose(100 * comps[1], [14.55, 9.57], atol=0.1)


def test_nucleus_matches_thermo_calc_with_dupin():
    """The largest-driving-force gamma' against the as-quenched matrix, Table 3 row [28]: 75.36 Ni,
    16.43 Al, 8.21 Cr at.%; found here within 0.06 at.%, with a driving force of 369 J/mol."""
    mu0, _ = _mu_and_tracer(ALLOY)
    x, dg = _precipitate().nucleus(mu0, T)
    np.testing.assert_allclose(100 * x, [16.43, 8.21, 75.36], atol=0.1)
    assert dg == pytest.approx(369.0, abs=2.0)


def test_nucleus_is_the_parallel_tangent():
    """At the nucleus composition mu_p,i - mu_0,i is the same for every element (pycalphad, phase alone at
    that composition), and the driving force is sum_i x_i mu_0i - G_p."""
    mu0, _ = _mu_and_tracer(ALLOY)
    prec = _precipitate()
    x, dg = prec.nucleus(mu0, T)
    eq = equilibrium(_db(), ELEMENTS + ["VA"], ["L12_FCC"], {v.T: T, v.P: 101325, v.N: 1, v.X("AL"): x[0], v.X("CR"): x[1]})
    mu_p = dict(zip([str(c) for c in eq.component.values], np.ravel(eq.MU.values)))
    shift = np.array([mu_p[e] for e in ELEMENTS]) - mu0
    assert np.ptp(shift) < 1.0
    assert dg == pytest.approx(x @ mu0 - prec.gm(x, T), abs=0.5)


@pytest.mark.skip(reason="Sudbrack et al. (2007) Table 4 lists Thermo-Calc curvatures of Dupin's gamma at x = 0.0393 Al, "
                         "0.1490 Cr: 365962, 113607, 117512 J/mol. pycalphad's FCC_A1, pinned to Dupin's appendix, gives "
                         "404528, 152176, 145098 (its own chemical potentials by differences agree to 1 J/mol). The "
                         "equilibria agree within 0.08 at.%; the cause is not found. doi 10.1016/j.actamat.2006.08.009")
def test_matrix_curvature_against_table_4():
    pass


def test_campbell_intrinsic_diffusivities_against_table_6():
    """Table 6: Campbell mobilities with Saunders' thermodynamics give iD_AlAl = 22.22e-21 and
    iD_CrCr = 6.999e-21 m^2/s at Ni-5.24Al-14.24Cr; with Dupin's thermodynamics 25.8e-21 and 6.98e-21."""
    x = np.array([0.0524, 0.1424, 0.8052])
    d = _matrix().intrinsic_matrix(x, T)
    assert d[0, 0] == pytest.approx(22.22e-21, rel=0.20)
    assert d[1, 1] == pytest.approx(6.999e-21, rel=0.05)


# --- the model's equations ----------------------------------------------------------------------------

def test_growth_rate_is_eq_45():
    radius, comp = np.array([1e-9, 3e-9]), np.array([[0.17, 0.08, 0.75], [0.16, 0.07, 0.77]])
    x0, tracer, f = np.array([0.05, 0.14, 0.81]), np.array([1e-20, 3e-21, 2.8e-21]), np.array([5e7, 4e7])
    omega, g = SETTINGS.molar_volume, SETTINGS.gamma
    res = ((comp - x0) ** 2 / (x0 * tracer)).sum(axis=1) / omega
    expected = (f - 2 * g / radius) / (R * T * radius * res)
    np.testing.assert_allclose(growth_rates(radius, comp, f, x0, tracer, SETTINGS, T), expected, rtol=1e-14)


def test_attachment_rate_reduces_to_eq_42_in_a_dilute_binary():
    """For pure-B nuclei in a dilute A-B matrix eq. (46) becomes eq. (42), beta* = 4 pi rho*^2 D_B x_B / a^4,
    to first order in x_B: the A term adds x_B D_B / ((1 - x_B) D_A) relative."""
    s = NucleationSettings(gamma=0.1, molar_volume=1e-5, lattice_parameter=4e-10)
    x0, tracer, f = np.array([1e-4, 1 - 1e-4]), np.array([1e-19, 1e-19]), 1e9
    j, rho = nucleation(np.array([1.0, 0.0]), f, x0, tracer, s, 700.0, 1e30)
    kt = K_B * 700.0
    zeldovich = s.molar_volume / N_A / (2 * np.pi * rho**2) * np.sqrt(s.gamma / kt)
    beta = j / (s.site_density * zeldovich * np.exp(-16 * np.pi * s.gamma**3 / (3 * f**2) / kt))
    assert rho == pytest.approx(2 * s.gamma / f, rel=1e-14)
    assert beta == pytest.approx(4 * np.pi * rho**2 * 1e-19 * 1e-4 / s.lattice_parameter**4, rel=2e-4)


def test_no_nucleation_without_driving_force():
    j, rho = nucleation(np.array([0.2, 0.8]), -1e6, np.array([0.1, 0.9]), np.array([1e-20, 1e-20]), SETTINGS, T, 10.0)
    assert j == 0.0 and rho == np.inf


@functools.lru_cache(maxsize=None)
def _lsw_run():
    """Dilute binary of Perez et al. (2008) Table 1: X^i = 0.01, X^P = 1, D = 5e-20 m^2/s, gamma = 0.13 J/m^2,
    v_at = 1.6e-29 m^3, 160 C, as an ideal fcc A-B with pure-B precipitates."""
    tdb = """
    ELEMENT VA VACUUM 0 0 0 !
    ELEMENT A BLANK 1 0 0 !
    ELEMENT B BLANK 1 0 0 !
    TYPE_DEFINITION % SEQ * !
    PHASE FCC % 1 1 !
    CONSTITUENT FCC :A,B: !
    PARAMETER G(FCC,A;0) 298.15 0; 6000 N !
    PARAMETER G(FCC,B;0) 298.15 R*T*LN(100); 6000 N !
    PARAMETER MQ(FCC&A,*;0) 298.15 R*T*LN(5E-20); 6000 N !
    PARAMETER MQ(FCC&B,*;0) 298.15 R*T*LN(5E-20); 6000 N !
    PHASE PPT % 1 1 !
    CONSTITUENT PPT :B: !
    PARAMETER G(PPT,B;0) 298.15 0; 6000 N !
    """
    db = Database(tdb)
    matrix = SolutionPhase(db, "FCC", ["B", "A"], mobility_db=db)
    prec = PrecipitateThermo(db, "PPT", ["B", "A"], composition=[1.0, 0.0])
    s = NucleationSettings(gamma=0.13, molar_volume=1.6e-29 * N_A, lattice_parameter=4.04e-10)
    radii = np.linspace(4.5e-9, 5.5e-9, 41)
    num = np.exp(-((radii - 5e-9) / 0.3e-9) ** 2)
    num *= 0.02 / (num @ (4 * np.pi / 3 * radii**3))
    alloy_b = 0.02 + 0.98 * 0.0115
    times = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9])
    res = solve_multicomponent_precipitation(matrix, prec, [alloy_b, 1 - alloy_b], 433.15, times[-1], s, times,
                                             t_start=1.0, max_radius_change=0.005, min_classes=200, max_classes=400,
                                             initial={"number": num, "radius": radii,
                                                      "composition": np.tile([1.0, 0.0], (radii.size, 1))})
    return res, times


def test_coarsening_follows_lsw():
    """With F = (RT/Omega) ln(x/x_eq) and D_A = D_B = D, eq. (45) linearised about x_eq is the LSW growth law
    drho/dt = (x_eq D R0 / (1 - x_eq)) (1/R* - 1/rho) / rho with R0 = 2 gamma v_at / kT, so
    d<R>^3/dt = (4/9) x_eq/(1 - x_eq) R0 D, Perez et al. eq. (11). Averaged over 1e7-3e9 s the rate is within
    7 %; single windows scatter by about as much because classes dissolve one at a time."""
    res, times = _lsw_run()
    r0 = 2 * 0.13 * 1.6e-29 / (K_B * 433.15)
    lsw = 4 / 9 * 0.01 / 0.99 * r0 * 5e-20
    r3 = res.mean_radius**3
    rates = np.diff(r3) / np.diff(times) / lsw
    assert np.mean(rates[1:]) == pytest.approx(1.0, abs=0.07)
    assert np.all((rates[1:] > 0.75) & (rates[1:] < 1.25))


def test_coarsening_distribution_has_the_lsw_shape():
    """The LSW distribution, g(u) = u^2 (3/(3+u))^(7/3) ((3/2)/(3/2-u))^(11/3) exp(-u/(3/2-u)) with u = R/<R>,
    integrated here, has its 99th number percentile at 1.328 and 2.7 % of the particles above 1.3; it ends at
    3/2 but holds only 0.1 % above 1.37, so the largest class is not expected there. From 3e7 s on, once
    the initial distribution has relaxed (at 1e7 s the percentile is still 1.29), both match."""
    from scipy.integrate import quad

    def lsw(u):
        return 0.0 if u >= 1.5 else u**2 * (3 / (3 + u)) ** (7 / 3) * (1.5 / (1.5 - u)) ** (11 / 3) * np.exp(-u / (1.5 - u))

    total = quad(lsw, 0, 1.5, limit=200)[0]
    above = quad(lsw, 1.3, 1.5, limit=200)[0] / total
    assert above == pytest.approx(0.0267, abs=5e-4)
    res, times = _lsw_run()
    for c in res.classes[3:]:
        w, r = c["number"], c["radius"]
        u = r / ((w @ r) / w.sum())
        order = np.argsort(u)
        cumulative = np.cumsum(w[order]) / w.sum()
        assert u[order][np.searchsorted(cumulative, 0.99)] == pytest.approx(1.328, abs=0.03)
        assert w[u > 1.3].sum() / w.sum() == pytest.approx(above, abs=0.02)


def test_merging_and_splitting_conserve_the_population():
    """Number, volume, solute and Gibbs energy of the classes survive a merge and a split; the results-level
    mass balance cannot see a violation, since it is computed from the same classes."""
    from janus_kinetics.multicomponent_precipitation import _merge_closest, _split_most_populated

    rng = np.random.default_rng(3)
    number = rng.uniform(1e20, 1e22, 6)
    radius = np.sort(rng.uniform(1e-9, 5e-9, 6))
    comp = rng.dirichlet([2, 1, 6], 6)
    gibbs = rng.uniform(-6e4, -5e4, 6)

    def totals(c):
        vol = c["number"] * c["radius"] ** 3
        return np.concatenate([[c["number"].sum(), vol.sum()], vol @ c["comp"], [vol @ c["gibbs"]]])

    classes = {"number": number, "radius": radius, "comp": comp, "gibbs": gibbs}
    before = totals(classes)
    for step in (_merge_closest, _split_most_populated):
        np.testing.assert_allclose(totals(step(classes)), before, rtol=1e-12)
    assert len(_merge_closest(classes)["number"]) == 5
    assert len(_split_most_populated(classes)["number"]) == 7


def test_mass_balance_holds_at_every_saved_time():
    res = _sudbrack_run()
    phi = res.volume_fraction[:, None]
    np.testing.assert_allclose((1 - phi) * res.matrix_composition + phi * res.precipitate_composition,
                               np.tile(ALLOY, (len(res.times), 1)), atol=1e-12)


# --- kinetics against the atom-probe measurements ----------------------------------------------------

def test_kinetics_at_4_h_against_table_1():
    """Table 1 at 4 h: <R> = 1.27 nm, N_v = 3.2e24 m^-3, phi = 5.2 %. With gamma from coarsening (22.5 mJ/m^2)
    and no adjustment, the model gives 1.54 nm, 3.2e24 m^-3 and 5.3 %."""
    res = _sudbrack_run()
    i = list(HOURS).index(4)
    assert res.volume_fraction[i] == pytest.approx(0.052, rel=0.10)
    assert res.number_density[i] == pytest.approx(3.2e24, rel=0.15)
    assert res.mean_radius[i] == pytest.approx(1.27e-9, rel=0.30)


def test_coarsening_rate_against_the_papers_own_model():
    """Sudbrack et al. (2007) p. 126: Campbell mobilities give K(phi) = 1.29e-30 m^3/s with their f(phi) = 2.3,
    so 5.6e-31 for isolated particles, 15 times the measured K = 8.8e-32 (they attribute the gap to diffusion
    through the gamma' as well). Between 256 and 1024 h the model here gives 3.9e-31: within a factor of 1.5 of
    the paper's model and 4.5 times faster than the measurement. A line through <R>^3 from 64 to 1024 h gives
    4.4-5.0 times the measurement whether the step limits are 1, 4 or 8 times tighter (checked when written;
    too slow to repeat here)."""
    res = _sudbrack_run()
    i, j = list(HOURS).index(256), list(HOURS).index(1024)
    k = (res.mean_radius[j] ** 3 - res.mean_radius[i] ** 3) / ((HOURS[j] - HOURS[i]) * 3600)
    assert 5.6e-31 / 1.5 < k < 5.6e-31 * 1.5
    assert 3 < k / 8.8e-32 < 7


def test_late_matrix_approaches_the_dupin_solvus():
    """At 1024 h the matrix (4.06 Al, 14.79 Cr at.%) has nearly reached Dupin's gamma solvus (3.95, 14.82);
    the measured matrix (3.27, 15.57) sits on a different solvus, Table 2."""
    res = _sudbrack_run()
    np.testing.assert_allclose(100 * res.matrix_composition[-1][:2], [3.95, 14.82], atol=0.2)


@pytest.mark.skip(reason="Sudbrack et al. (2006) p. 3205 measure J_SS = 5.9e21 m^-3 s^-1 and tau = 540 s. Classical "
                         "nucleation with gamma = 22.5 mJ/m^2 gives 2.6e25 and 2330 s at a barrier of 5.3 kT, where the "
                         "theory is not expected to hold; the authors also cite quenched-in short-range order and warn "
                         "against the comparison. The model nucleates too early (5e24 m^-3 at 0.167 h against 3.6e23). "
                         "doi 10.1016/j.actamat.2006.03.015")
def test_nucleation_rate_against_measurement():
    pass


@pytest.mark.skip(reason="Sudbrack et al. (2006) Table 2: the measured equilibrium volume fraction is 15.6 %; Dupin's "
                         "description gives 11.8 % (Thermo-Calc 12.34 %, Sudbrack et al. 2007 Table 3), so the late volume "
                         "fraction (9.8 % at 1024 h) cannot reach the measurement. doi 10.1016/j.actamat.2006.03.015")
def test_late_volume_fraction_against_measurement():
    pass


# --- input checks ------------------------------------------------------------------------------------

def test_inputs_are_checked():
    with pytest.raises(ValueError, match="lattice parameter"):
        NucleationSettings(gamma=0.02, molar_volume=7e-6, lattice_parameter=3.5)
    with pytest.raises(ValueError, match="one mole fraction"):
        PrecipitateThermo(_db(), "L12_FCC", ELEMENTS, composition=[0.2, 0.8])
    with pytest.raises(ValueError, match="alloy"):
        solve_multicomponent_precipitation(_matrix(), _precipitate(), [0.05, 0.95], T, 10.0, SETTINGS, [1.0])
    with pytest.raises(ValueError, match="min_classes"):
        solve_multicomponent_precipitation(_matrix(), _precipitate(), ALLOY, T, 10.0, SETTINGS, [1.0],
                                           min_classes=300, max_classes=400)


# --- precipitates whose composition evolves (part I eqs. 25-39) -------------------------------------------

def test_minimizer_matches_pycalphad_inside_the_single_phase_field():
    """Where pycalphad keeps gamma' as one phase the minimiser's Gibbs energy and chemical potentials agree
    within 0.01 J/mol; where pycalphad splits it into two composition sets the homogeneous state of a single
    particle lies above that, never below."""
    pm = _precipitate().minimizer
    xs = np.array([[0.1637, 0.0825], [0.1455, 0.0957], [0.15, 0.07], [0.19, 0.10], [0.20, 0.02]])
    x3 = np.column_stack([xs, 1 - xs.sum(axis=1)])
    y0 = np.tile([0.01, 0.01, 0.98, 0.9, 0.05, 0.05], (len(xs), 1))
    y, g, mu = pm.solve(x3, y0, T)
    for x, gk, mk in zip(xs, g, mu):
        eq = equilibrium(_db(), ELEMENTS + ["VA"], ["L12_FCC"], {v.T: T, v.P: 101325, v.N: 1, v.X("AL"): x[0], v.X("CR"): x[1]})
        sets = int(np.sum(np.ravel(eq.Phase.values) != ""))
        ref_g = float(np.ravel(eq.GM.values)[0])
        if sets == 1:
            ref_mu = dict(zip([str(c) for c in eq.component.values], np.ravel(eq.MU.values)))
            assert gk == pytest.approx(ref_g, abs=0.01)
            np.testing.assert_allclose(mk, [ref_mu[e] for e in ELEMENTS], atol=0.01)
        else:
            assert gk > ref_g


def _class_state():
    rng = np.random.default_rng(7)
    radius = np.array([0.9e-9, 2.5e-9, 6e-9])
    comp = np.array([[0.17, 0.085, 0.745], [0.16, 0.09, 0.75], [0.15, 0.094, 0.756]])
    mu_k = rng.uniform(-1, 1, (3, 3)) * 2000 + np.array([-178000.0, -40000.0, -40300.0])
    x0 = np.array([0.045, 0.146, 0.809])
    mu0 = np.array([-177500.0, -39400.0, -40400.0])
    tracer = np.array([1.06e-20, 2.97e-21, 2.85e-21])
    return radius, comp, mu_k, x0, mu0, tracer


def test_evolution_rates_satisfy_the_extremum_principle():
    """Part I eqs. (27)-(28), dG/dq = -(1/2) dQ/dq_dot, with dG/drho and dG/dc_ki from eqs. (8)-(9) and Q from
    eqs. (15) and (23) (infinite interface mobility), differentiated numerically here; Q is quadratic in the
    rates, so central differences are exact. This checks the linear system independently of its printed
    coefficients, including the misprinted factor in eq. (32)."""
    from janus_kinetics.multicomponent_precipitation import evolution_rates

    radius, comp, mu_k, x0, mu0, tracer = _class_state()
    dk = np.array([3e-23, 1.5e-23, 2e-23])
    omega, g = SETTINGS.molar_volume, SETTINGS.gamma
    rate, dx = evolution_rates(radius, comp, mu_k, x0, mu0, tracer, dk, SETTINGS, T)
    # the solver renormalises compositions each step, which would hide a wrong rate for the last element
    np.testing.assert_allclose(dx.sum(axis=1), 0.0, atol=1e-12 * np.abs(dx).max())
    c0, d0 = x0 / omega, tracer
    for k in range(3):
        rho, ck, mk = radius[k], comp[k] / omega, mu_k[k]
        ref = -1  # the paper's element 1 is the last one here

        def dissipation(q):
            rdot, cdot = q[0], np.append(q[1:], -q[1:].sum())
            q2 = 4 * np.pi * R * T * rho**5 / 45 * (cdot**2 / (ck * dk)).sum()
            q3 = 4 * np.pi * R * T * rho**3 * ((rdot * (ck - c0) + rho * cdot / 3) ** 2 / (c0 * d0)).sum()
            return q2 + q3

        dgd_rho = 8 * np.pi * rho * g + 4 * np.pi * rho**2 * (
            (mk[ref] - mu0[ref]) / omega + (ck[:-1] * (mk[:-1] - mu0[:-1] - mk[ref] + mu0[ref])).sum())
        dgd_c = 4 * np.pi * rho**3 / 3 * (mk[:-1] - mu0[:-1] - mk[ref] + mu0[ref])
        q = np.concatenate([[rate[k]], dx[k, :-1] / omega])
        grad = np.empty_like(q)
        for i in range(q.size):
            h = np.zeros_like(q)
            h[i] = 1e-6 * max(abs(q[i]), 1e-30)
            grad[i] = (dissipation(q + h) - dissipation(q - h)) / (2 * h[i])
        np.testing.assert_allclose(-0.5 * grad, np.concatenate([[dgd_rho], dgd_c]), rtol=1e-6)


def test_frozen_composition_recovers_eq_45():
    """As the diffusivities in the precipitate go to zero the composition stops changing and the growth
    rate becomes eq. (45)."""
    from janus_kinetics.multicomponent_precipitation import evolution_rates

    radius, comp, mu_k, x0, mu0, tracer = _class_state()
    rate, dx = evolution_rates(radius, comp, mu_k, x0, mu0, tracer, np.full(3, 1e-40), SETTINGS, T)
    f = (comp @ mu0 - (comp * mu_k).sum(axis=1)) / SETTINGS.molar_volume
    np.testing.assert_allclose(rate, growth_rates(radius, comp, f, x0, tracer, SETTINGS, T), rtol=1e-9)
    assert np.abs(dx).max() < 1e-15


@functools.lru_cache(maxsize=None)
def _sudbrack_evolving_run():
    """Tracer diffusivities in gamma': Al 2.7e-23 and Cr 1.4e-23 m^2/s in Ni3Al at 873 K, as Sudbrack et al.
    (2007) p. 126 quote them; Ni is not given and takes the Al value (1.4e-23 changes <R> at 1024 h by 4 %)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_multicomponent_precipitation(_matrix(), _precipitate(), ALLOY, T, HOURS[-1] * 3600, SETTINGS,
                                                  HOURS * 3600, precipitate_tracer=[2.7e-23, 1.4e-23, 2.7e-23])


def test_evolving_precipitates_relax_to_the_dupin_solvus():
    """The gamma' leaves its nucleus composition (16.3 Al, 8.3 Cr at.%) for Dupin's gamma' solvus (14.56,
    9.54): 14.70 and 9.45 at 1024 h. Al falls as the atom probe shows (19.1 to 16.7, Table 2); Cr rises where
    the measured Cr falls (9.7 to 6.9), since Dupin's description puts the gamma' Cr at 9.5 rather than the
    measured 6.8."""
    res = _sudbrack_evolving_run()
    xp = 100 * res.precipitate_composition
    assert xp[0][0] > 16.0 and xp[-1][0] < xp[0][0] - 1.0
    np.testing.assert_allclose(xp[-1][:2], [14.56, 9.54], atol=0.3)
    phi = res.volume_fraction[:, None]
    np.testing.assert_allclose((1 - phi) * res.matrix_composition + phi * res.precipitate_composition,
                               np.tile(ALLOY, (len(res.times), 1)), atol=1e-12)


def test_evolving_inputs_are_checked():
    with pytest.raises(ValueError, match="one positive diffusivity"):
        solve_multicomponent_precipitation(_matrix(), _precipitate(), ALLOY, T, 10.0, SETTINGS, [1.0],
                                           precipitate_tracer=[1e-23, 1e-23])


def test_second_alloy_shows_the_same_coarsening_gap():
    """Booth-Morrison et al., Acta Mater. 56 (2008) 3422, Table 1 and p. 3432: Ni-7.5Al-8.5Cr at 873 K,
    gamma = 24 mJ/m^2, measured K = 1.84e-31 m^3/s. The model matches <R> at 1 h (1.22 against 1.24 nm) and
    coarsens 5-6 times faster than measured, as for Ni-5.2Al-14.2Cr with evolving compositions (5.6-6.5 times,
    from <R>^3 over 64-1024 h with 1-8 times tighter step limits and 100-800 classes): the gap follows the
    mobilities, not the alloy."""
    s = NucleationSettings(gamma=0.024, molar_volume=SETTINGS.molar_volume, lattice_parameter=SETTINGS.lattice_parameter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_multicomponent_precipitation(_matrix(), _precipitate(), [0.075, 0.085, 0.840], T, HOURS[-1] * 3600,
                                                 s, HOURS * 3600, precipitate_tracer=[2.7e-23, 1.4e-23, 2.7e-23])
    i1, i, j = list(HOURS).index(1), list(HOURS).index(256), list(HOURS).index(1024)
    assert res.mean_radius[i1] == pytest.approx(1.24e-9, rel=0.10)
    k = (res.mean_radius[j] ** 3 - res.mean_radius[i] ** 3) / ((HOURS[j] - HOURS[i]) * 3600)
    assert 4 < k / 1.84e-31 < 8


# --- misfit energy (part I eq. 37 and Appendix A) ----------------------------------------------------------

def test_misfit_energy_follows_appendix_a():
    """Eq. (A.1) lambda0 = eps0^2 E / (1 - nu); with yielding f(S) = 2 S (1 - ln S) - S^2, which the appendix
    says reaches 1.0 at S = 0.199 and its maximum 1.456 at S = 0.567 (p. 173)."""
    from janus_kinetics.multicomponent_precipitation import misfit_energy

    e, nu, eps = 200e9, 0.3, 0.01
    lam0 = eps**2 * e / (1 - nu)
    assert misfit_energy(eps, e, nu) == pytest.approx(lam0, rel=1e-14)

    def sigma_for(s):  # yield stress giving a softening factor S, eq. (A.2)
        return s * e * eps / (1 - nu)

    assert misfit_energy(eps, e, nu, sigma_for(0.199)) / lam0 == pytest.approx(1.0, abs=2e-3)
    assert misfit_energy(eps, e, nu, sigma_for(0.567)) / lam0 == pytest.approx(1.456, abs=1e-3)
    s = np.linspace(0.05, 0.99, 400)
    f = np.array([misfit_energy(eps, e, nu, sigma_for(x)) for x in s]) / lam0
    assert s[np.argmax(f)] == pytest.approx(0.567, abs=0.005)
    assert misfit_energy(eps, e, nu, sigma_for(2.0)) == pytest.approx(lam0, rel=1e-14)  # no yielding


def test_misfit_lowers_the_driving_force_everywhere():
    """Eq. (37) F = -lambda - sum c (mu_k - mu_0): the frozen-composition rate with misfit equals eq. (45)
    with F - lambda, and a misfit above the chemical driving force (5.5e7 J/m^3 for the Ni-Al-Cr alloy)
    suppresses nucleation."""
    from janus_kinetics.multicomponent_precipitation import evolution_rates

    radius, comp, mu_k, x0, mu0, tracer = _class_state()
    lam = 2e7
    s = NucleationSettings(SETTINGS.gamma, SETTINGS.molar_volume, SETTINGS.lattice_parameter, misfit=lam)
    rate, _ = evolution_rates(radius, comp, mu_k, x0, mu0, tracer, np.full(3, 1e-40), s, T)
    f = (comp @ mu0 - (comp * mu_k).sum(axis=1)) / SETTINGS.molar_volume - lam
    np.testing.assert_allclose(rate, growth_rates(radius, comp, f, x0, tracer, SETTINGS, T), rtol=1e-9)

    def run(misfit):
        s = NucleationSettings(SETTINGS.gamma, SETTINGS.molar_volume, SETTINGS.lattice_parameter, misfit=misfit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return solve_multicomponent_precipitation(_matrix(), _precipitate(), ALLOY, T, 3600.0, s, [3600.0])

    assert run(6e7).volume_fraction[-1] == 0.0
    assert 0 < run(2e7).number_density[-1] < 0.5 * run(0.0).number_density[-1]


def test_misfit_must_not_be_negative():
    with pytest.raises(ValueError, match="misfit"):
        NucleationSettings(0.02, 7e-6, 3.5e-10, misfit=-1.0)


# --- nucleation on grain boundaries, edges and corners (Clemm and Fisher 1955) ------------------------------
from janus_kinetics.multicomponent_precipitation import junction_barrier_factor, junction_factors  # noqa: E402

# Fig. 6 (p. 72): ferrite in austenite, gamma_AA = 850, gamma_AB = 600 erg/cm^2, Delta f_v = -4.35e8 erg/cm^3
CF_GAA, CF_GAB, CF_DF = 850.0, 600.0, 4.35e8
CF_FIG6 = {"bulk": 1.913e-8, "boundary": 0.220e-8, "edge": 0.0436e-8, "corner": 0.0096e-8}  # ergs


def _cf_barrier(site):
    a, b, c = junction_factors(site, CF_GAA / (2 * CF_GAB))
    return 4 / 27 * (b * CF_GAB - a * CF_GAA) ** 3 / (c * c * CF_DF**2)


@pytest.mark.parametrize("site", ["bulk", "boundary", "edge"])
def test_junction_barriers_against_fig_6(site):
    """W* = (4/27)(b gamma_AB - a gamma_AA)^3 / (c^2 Delta f^2), p. 70, with the paper's a, b, c."""
    assert _cf_barrier(site) == pytest.approx(CF_FIG6[site], rel=0.005)


@pytest.mark.skip(reason="Clemm and Fisher (1955) Fig. 6 gives 0.0096e-8 erg for the four-grain junction; their printed "
                         "a, b, c give 0.0109e-8 (13 % more). The same coefficients pass every internal check (barrier "
                         "ratio equal to volume ratio, zero barrier at k = (2/3)^1/2, the sphere at k = 0), so the "
                         "figure's value is taken as a computing slip. doi 10.1016/0001-6160(55)90014-6")
def test_corner_barrier_against_fig_6():
    pass


@pytest.mark.parametrize("site", ["boundary", "edge", "corner"])
def test_junction_coefficients_are_consistent(site):
    """For caps of one curvature the barrier ratio (b - 2ka)^3 / (36 pi c^2) equals the volume ratio
    c / (4 pi / 3) at every k, the sphere is recovered at k = 0, and the barrier vanishes at the critical
    k the paper lists (1, sqrt(3)/2, sqrt(2/3))."""
    for k in (0.0, 0.2, 0.5, 0.7):
        a, b, c = junction_factors(site, k)
        assert junction_barrier_factor(site, k) == pytest.approx(c / (4 * np.pi / 3), rel=1e-9, abs=1e-12)
    assert junction_barrier_factor(site, 0.0) == pytest.approx(1.0, rel=1e-12)
    critical = {"boundary": 1.0, "edge": np.sqrt(3) / 2, "corner": np.sqrt(2 / 3)}[site]
    assert junction_barrier_factor(site, critical * (1 - 1e-6)) < 1e-8
    with pytest.raises(ValueError, match="no nucleation barrier"):
        junction_barrier_factor(site, critical)


def test_boundary_nucleation_lowers_the_barrier_in_a_run():
    """Same site density, nuclei on grain boundaries with gamma_gb = gamma (k = 0.5, barrier factor 0.31):
    the barrier at the start falls from 5.3 kT to 1.7 kT, so more particles form in the first minute."""
    def run(site, k):
        s = NucleationSettings(SETTINGS.gamma, SETTINGS.molar_volume, SETTINGS.lattice_parameter, sites=1e22,
                               site=site, boundary_ratio=k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return solve_multicomponent_precipitation(_matrix(), _precipitate(), ALLOY, T, 60.0, s, [60.0])

    assert junction_barrier_factor("boundary", 0.5) == pytest.approx(0.3125, rel=1e-12)
    assert run("boundary", 0.5).number_density[-1] > 2 * run("bulk", 0.0).number_density[-1]


# --- temperature schedules ---------------------------------------------------------------------------------
from janus_kinetics import TemperatureSchedule  # noqa: E402


def _frozen_run(T, t_end, save, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_multicomponent_precipitation(_matrix(), _precipitate(), ALLOY, T, t_end, SETTINGS, save, **kwargs)


def test_isothermal_schedule_equals_a_constant_temperature():
    a = _frozen_run(T, 3600.0, [600.0, 3600.0])
    b = _frozen_run(TemperatureSchedule.isothermal(T), 3600.0, [600.0, 3600.0])
    np.testing.assert_allclose(b.mean_radius, a.mean_radius, rtol=1e-12)
    np.testing.assert_allclose(b.volume_fraction, a.volume_fraction, rtol=1e-12)


@pytest.mark.parametrize("T2", [848.0, 893.0])
def test_temperature_step_matches_two_chained_runs(T2):
    """873 K for 1 h, then T2 for 1 h, as one schedule (a 1 s ramp) and as two runs, the second started
    from the classes the first leaves: volume fraction, number and mean radius agree within 2 %. The class
    Gibbs energies must follow the temperature for this to hold. At 893 K most of the gamma' dissolves
    (Dupin's equilibrium fraction drops and the small particles fall below the new critical radius)."""
    first = _frozen_run(T, 3600.0, [3600.0])
    c = first.classes[-1]
    second = _frozen_run(T2, 7200.0, [7200.0], t_start=3601.0,
                         initial={"number": c["number"], "radius": c["radius"], "composition": c["composition"]})
    sched = _frozen_run(TemperatureSchedule([0.0, 3600.0, 3601.0], [T, T, T2]), 7200.0, [3600.0, 7200.0])
    assert sched.volume_fraction[-1] == pytest.approx(second.volume_fraction[-1], rel=0.02)
    assert sched.number_density[-1] == pytest.approx(second.number_density[-1], rel=0.02)
    assert sched.mean_radius[-1] == pytest.approx(second.mean_radius[-1], rel=0.02)
    if T2 > T:
        assert sched.volume_fraction[-1] < 0.7 * first.volume_fraction[-1]



def test_late_stage_keeps_a_size_distribution():
    """Late in a run the nucleation rate is astronomically small; nuclei are held back until they would make
    up a millionth of the population, so no class is empty and the distribution keeps coarsening. Before this,
    a class of 1e-120 nuclei per step filled the array and merging collapsed the real particles into one class,
    which stopped coarsening for hundreds of hours."""
    res = _sudbrack_evolving_run()
    c = res.classes[-1]
    assert c["number"].min() / c["number"].sum() > 1e-9
    assert len(c["radius"]) >= 100
    r3 = res.mean_radius**3
    late = HOURS >= 16
    assert np.all(np.diff(r3[late]) > 0)


# --- populations competing for one matrix (part I eq. 4, part II section 3.5) ------------------------------

def _ideal_binary(sites=None):
    """The dilute binary of _lsw_run as (matrix, precipitate, settings)."""
    db = Database("""
    ELEMENT VA VACUUM 0 0 0 !
    ELEMENT A BLANK 1 0 0 !
    ELEMENT B BLANK 1 0 0 !
    TYPE_DEFINITION % SEQ * !
    PHASE FCC % 1 1 !
    CONSTITUENT FCC :A,B: !
    PARAMETER G(FCC,A;0) 298.15 0; 6000 N !
    PARAMETER G(FCC,B;0) 298.15 R*T*LN(100); 6000 N !
    PARAMETER MQ(FCC&A,*;0) 298.15 R*T*LN(5E-20); 6000 N !
    PARAMETER MQ(FCC&B,*;0) 298.15 R*T*LN(5E-20); 6000 N !
    PHASE PPT % 1 1 !
    CONSTITUENT PPT :B: !
    PARAMETER G(PPT,B;0) 298.15 0; 6000 N !
    """)
    matrix = SolutionPhase(db, "FCC", ["B", "A"], mobility_db=db)
    prec = PrecipitateThermo(db, "PPT", ["B", "A"], composition=[1.0, 0.0])
    s = NucleationSettings(gamma=0.13, molar_volume=1.6e-29 * N_A, lattice_parameter=4.04e-10, sites=sites)
    return matrix, prec, s


def _binary_seeds(take=slice(None)):
    radii = np.linspace(4.5e-9, 5.5e-9, 40)
    num = np.exp(-((radii - 5e-9) / 0.3e-9) ** 2)
    num *= 0.02 / (num @ (4 * np.pi / 3 * radii**3))
    return {"number": num[take], "radius": radii[take], "composition": np.tile([1.0, 0.0], (radii[take].size, 1))}


BINARY_ALLOY = [0.02 + 0.98 * 0.0115, 1 - 0.02 - 0.98 * 0.0115]
BINARY_TIMES = np.array([1e6, 1e7, 1e8])


def test_a_population_split_in_two_runs_as_one():
    """The same particles held as one population or dealt alternately into two give the same run: the two
    halves see one matrix, depleted by both. No nucleation, and no merging or splitting of classes, so the
    classes match one for one."""
    matrix, prec, s = _ideal_binary(sites=0.0)
    kw = dict(t_start=1.0, min_classes=1, max_classes=400)
    one = solve_competing_precipitation(matrix, [Population(prec, s, _binary_seeds())], BINARY_ALLOY, 433.15,
                                        BINARY_TIMES[-1], BINARY_TIMES, **kw)[0]
    two = solve_competing_precipitation(matrix, [Population(prec, s, _binary_seeds(slice(0, None, 2))),
                                                 Population(prec, s, _binary_seeds(slice(1, None, 2)))],
                                        BINARY_ALLOY, 433.15, BINARY_TIMES[-1], BINARY_TIMES, **kw)
    np.testing.assert_allclose(two[0].matrix_composition, one.matrix_composition, rtol=1e-9)
    np.testing.assert_allclose(two[0].volume_fraction + two[1].volume_fraction, one.volume_fraction, rtol=1e-9)
    np.testing.assert_allclose(two[0].number_density + two[1].number_density, one.number_density, rtol=1e-9)
    for i in range(BINARY_TIMES.size):
        merged = np.sort(np.concatenate([two[0].classes[i]["radius"], two[1].classes[i]["radius"]]))
        np.testing.assert_allclose(merged, np.sort(one.classes[i]["radius"]), rtol=1e-9)
    assert one.volume_fraction[-1] > 0.02  # the run did something: the seeds grew


def test_a_population_that_never_forms_changes_nothing():
    """A second population with no seeds and no nucleation sites leaves the first run unchanged to the bit."""
    matrix, prec, s = _ideal_binary(sites=0.0)
    kw = dict(t_start=1.0, min_classes=20, max_classes=60)
    alone = solve_competing_precipitation(matrix, [Population(prec, s, _binary_seeds())], BINARY_ALLOY, 433.15,
                                          BINARY_TIMES[-1], BINARY_TIMES, **kw)
    both = solve_competing_precipitation(matrix, [Population(prec, s, _binary_seeds()), Population(prec, s)],
                                         BINARY_ALLOY, 433.15, BINARY_TIMES[-1], BINARY_TIMES, **kw)
    for field in ("mean_radius", "number_density", "volume_fraction", "matrix_composition"):
        assert np.array_equal(getattr(both[0], field), getattr(alone[0], field))
    assert both[0].steps == alone[0].steps
    assert np.all(both[1].number_density == 0) and np.all(both[1].volume_fraction == 0)


THREE_PHASE_ALLOY = np.array([0.08, 0.30, 0.62])


@functools.lru_cache(maxsize=None)
def _three_phase_equilibrium():
    """Dupin's gamma + gamma' + alpha(bcc) equilibrium of Ni-8Al-30Cr at.% at 1073 K, from pycalphad:
    {phase: (mole fraction of phase, composition)} with keys "gamma", "gamma_prime", "bcc"."""
    eq = equilibrium(_db(), ELEMENTS + ["VA"], ["FCC_A1", "L12_FCC", "BCC_A2"],
                     {v.T: 1073.0, v.P: 101325, v.N: 1, v.X("AL"): 0.08, v.X("CR"): 0.30})
    names = [str(p) for p in np.ravel(eq.Phase.values)]
    fractions = np.ravel(eq.NP.values)
    comps = eq.X.values.reshape(len(names), -1)
    sets = [(n, f, x / x.sum()) for n, f, x in zip(names, fractions, comps) if n]
    assert sorted(n for n, _, _ in sets) == ["BCC_A2", "L12_FCC", "L12_FCC"]
    fcc = sorted([s for s in sets if s[0] == "L12_FCC"], key=lambda s: s[2][0])
    bcc = next(s for s in sets if s[0] == "BCC_A2")
    return {"gamma": fcc[0][1:], "gamma_prime": fcc[1][1:], "bcc": bcc[1:]}


def _three_phase_populations(gamma_bcc=0.1):
    eq = _three_phase_equilibrium()
    pops = []
    for key, phase, gamma, a in (("gamma_prime", "L12_FCC", 0.02, 0.3552e-9), ("bcc", "BCC_A2", gamma_bcc, 0.288e-9)):
        x = eq[key][1]
        pops.append(Population(PrecipitateThermo(_db(), phase, ELEMENTS, composition=x),
                               NucleationSettings(gamma=gamma, molar_volume=SETTINGS.molar_volume, lattice_parameter=a,
                                                  sites=0.0),
                               initial={"number": [1e20], "radius": [5e-9], "composition": [x]}))
    return pops


def test_two_phases_settle_on_the_three_phase_equilibrium():
    """gamma' and bcc seeds (1e20 m^-3 of each, 5 nm, at their equilibrium compositions, gamma = 0.02 and
    0.1 J/m^2) in Ni-8Al-30Cr at 1073 K. They grow from one matrix until each satisfies its own
    Gibbs-Thomson condition, F_k = 2 gamma_k / rho_k (from 1e5 s on within 1e-3; reversals under the step
    guard's threshold leave a scatter of up to 5e-4 between runs), and then hold the lever-rule
    fractions of the three-phase equilibrium, short by the capillary shift (gamma' 22.13 % for 22.26 %, bcc
    6.71 % for 6.77 %). Halving all step limits moves the matrix by 4e-8. Before steps were cut on a
    reversal of the matrix rate, the matrix swung about the fixed point by up to the step limit and F_k
    rho_k / (2 gamma_k) wandered between 0.72 and 1.09."""
    eq = _three_phase_equilibrium()
    pops = _three_phase_populations()
    times = np.array([1e3, 1e4, 1e5, 1e6])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_competing_precipitation(_matrix(), pops, THREE_PHASE_ALLOY, 1073.0, times[-1], times,
                                            min_classes=1)
    phi = sum(r.volume_fraction for r in res)[:, None]
    held = (1 - phi) * res[0].matrix_composition + sum(r.volume_fraction[:, None] * r.precipitate_composition
                                                       for r in res)
    np.testing.assert_allclose(held, np.tile(THREE_PHASE_ALLOY, (times.size, 1)), rtol=0, atol=1e-12)
    for i in (2, 3):
        x0 = res[0].matrix_composition[i]
        mu0 = np.asarray(_matrix().chemical_potentials(x0, 1073.0), dtype=float)
        for pop, r in zip(pops, res):
            p = pop.precipitate
            f = (p.fixed @ mu0 - p.gm(p.fixed, 1073.0)) / pop.settings.molar_volume
            assert f * r.mean_radius[i] / (2 * pop.settings.gamma) == pytest.approx(1.0, rel=1e-3)
    assert res[0].volume_fraction[-1] == pytest.approx(eq["gamma_prime"][0], abs=0.002)
    assert res[1].volume_fraction[-1] == pytest.approx(eq["bcc"][0], abs=0.001)
    np.testing.assert_allclose(res[0].matrix_composition[-1], eq["gamma"][1], rtol=0, atol=5e-4)


def test_competing_inputs_are_checked():
    matrix, prec, s = _ideal_binary()
    other = NucleationSettings(gamma=0.13, molar_volume=1.1 * s.molar_volume, lattice_parameter=4.04e-10)
    with pytest.raises(ValueError, match="same molar volume"):
        solve_competing_precipitation(matrix, [Population(prec, s), Population(prec, other)], BINARY_ALLOY,
                                      433.15, 10.0, [1.0])
    with pytest.raises(ValueError, match="at least one"):
        solve_competing_precipitation(matrix, [], BINARY_ALLOY, 433.15, 10.0, [1.0])
    with pytest.raises(ValueError, match="same elements"):
        solve_competing_precipitation(_matrix(), [Population(prec, s)], ALLOY, T, 10.0, [1.0])


# --- nucleation sites from the microstructure (Cahn 1956 eq. 1; part II p. 162) ----------------------------

def _bcc_voronoi_cell(d):
    """Face area, edge length and corner count of one Voronoi cell of a bcc lattice of cube edge d, the
    space-filling tetrakaidecahedron with square faces d apart, measured from scipy's construction."""
    from itertools import product

    from scipy.spatial import Voronoi

    pts = np.array([p for p in product(range(-2, 3), repeat=3)], dtype=float)
    pts = np.vstack([pts, pts + 0.5]) * d
    vor = Voronoi(pts)
    centre = int(np.argmin(np.linalg.norm(pts, axis=1)))
    area, edges = 0.0, set()
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if centre not in (p, q):
            continue
        poly = vor.vertices[rv]
        c = poly.mean(axis=0)
        normal = pts[q if p == centre else p] - pts[centre]
        u = (poly[0] - c) / np.linalg.norm(poly[0] - c)
        w = np.cross(normal / np.linalg.norm(normal), u)
        order = np.argsort(np.arctan2((poly - c) @ w, (poly - c) @ u))
        poly, ring = poly[order], np.asarray(rv)[order]
        area += 0.5 * np.linalg.norm(sum(np.cross(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))))
        edges |= {tuple(sorted((ring[i], ring[(i + 1) % len(ring)]))) for i in range(len(ring))}
    length = sum(np.linalg.norm(vor.vertices[a] - vor.vertices[b]) for a, b in edges)
    return area, length, len(vor.regions[vor.point_region[centre]])


def test_grain_structure_is_cahn_eq_1():
    """Cahn's S = 3.35/D, L = 8.5/D^2, C = 12.0/D^3 to his printed digits, and the exact values against a
    tetrakaidecahedron measured independently, as a bcc Voronoi cell (2/D^3 grains per volume; faces shared
    by two grains, edges by three, corners by four)."""
    d = 40e-6
    s, l, c = grain_structure(d)
    assert (round(s * d, 2), round(l * d**2, 1), round(c * d**3, 1)) == (3.35, 8.5, 12.0)
    area, length, corners = _bcc_voronoi_cell(d)
    grains = 2 / d**3
    assert corners == 24
    assert s == pytest.approx(area / 2 * grains, rel=1e-12)
    assert l == pytest.approx(length / 3 * grains, rel=1e-12)
    assert c == pytest.approx(corners / 4 * grains, rel=1e-12)


def test_nucleation_sites_count_the_atoms_on_each_defect():
    """The fraction of atoms on boundaries is the boundary area per volume times one atomic distance, and the
    edge and corner counts fall as 1/D^2 and 1/D^3."""
    vm = SETTINGS.molar_volume
    w = (vm / N_A) ** (1 / 3)
    bulk = nucleation_sites("bulk", vm)
    assert bulk == pytest.approx(N_A / vm, rel=1e-14)
    for d in (10e-6, 100e-6):
        s, l, c = grain_structure(d)
        assert nucleation_sites("boundary", vm, grain_size=d) / bulk == pytest.approx(s * w, rel=1e-12)
        assert nucleation_sites("edge", vm, grain_size=d) / bulk == pytest.approx(l * w**2, rel=1e-12)
        assert nucleation_sites("corner", vm, grain_size=d) == pytest.approx(12 / d**3, rel=1e-14)
    ratios = [nucleation_sites(site, vm, grain_size=10e-6) / nucleation_sites(site, vm, grain_size=100e-6)
              for site in ("boundary", "edge", "corner")]
    np.testing.assert_allclose(ratios, [10, 100, 1000], rtol=1e-12)
    assert nucleation_sites("dislocation", vm, dislocation_density=1e14) == pytest.approx(1e14 / w, rel=1e-14)


def test_nucleation_site_inputs_are_checked():
    vm = SETTINGS.molar_volume
    with pytest.raises(ValueError, match="grain_size"):
        nucleation_sites("edge", vm)
    with pytest.raises(ValueError, match="dislocation_density"):
        nucleation_sites("dislocation", vm)
    with pytest.raises(ValueError, match="site must be"):
        nucleation_sites("surface", vm, grain_size=1e-5)
    with pytest.raises(ValueError, match="grain size"):
        grain_structure(0.0)


# --- precipitate composition trajectories: Mao, Booth-Morrison, Sudbrack, Martin and Seidman, Acta Mater. 60
# (2012) 1871, Table 4 (APT at 873 K), against Philippe and Voorhees' eq. (45) with Dupin's Hessians -----------

from janus_kinetics.coarsening import precipitate_composition_shift  # noqa: E402

MAO_ALLOYS = {"A": (0.075, 0.085), "B": (0.052, 0.142), "C": (0.065, 0.095)}
# gamma'-precipitate Al and Cr (at.%) from 16 h on, with the long-time (infinity) row last
MAO_GP_LATE = {
    "A": ([18.6, 18.3, 18.18, 18.02, 17.8], [6.1, 6.0, 5.90, 5.87, 5.9]),
    "B": ([17.2, 17.2, 17.0, 16.7, 16.7], [7.3, 7.2, 7.1, 6.9, 6.8]),
    "C": ([18.0, 17.9, 17.8, 17.65, 17.54, 17.5], [6.9, 6.6, 6.4, 6.30, 6.22, 6.1]),
}
ORDERED_START = np.array([0.01, 0.01, 0.98, 0.9, 0.05, 0.05])


def _dupin_tie(alloy):
    eq = equilibrium(_db(), ELEMENTS + ["VA"], ["L12_FCC"],
                     {v.T: T, v.P: 101325, v.N: 1, v.X("AL"): alloy[0], v.X("CR"): alloy[1]})
    names = [str(p) for p in np.ravel(eq.Phase.values)]
    comp = eq.X.values.reshape(len(names), -1)
    order = [list(map(str, eq.component.values)).index(c) for c in ELEMENTS]
    sets = sorted((comp[i][order] for i, n in enumerate(names) if n), key=lambda x: x[0])
    return sets[0], sets[-1]  # gamma, gamma'


def _hessians(alloy):
    pm = _precipitate().minimizer
    xa, xb = _dupin_tie(alloy)
    yb, _, _ = pm.solve(xb, ORDERED_START, T)
    return xa, xb, _matrix().reduced_hessian(xa[:2], T), pm.hessian(yb, T)[0], yb[0]


def test_relaxed_hessian_of_ordered_gamma_prime():
    """The minimiser's Hessian, with the L1_2 order relaxed as x moves, equals central differences of its own
    chemical potentials (to 1e-6) at Dupin's gamma' of alloy C, which is ordered (Al and Cr on the minority
    sublattice); on disordered gamma it is the solution-phase Hessian to round-off."""
    pm = _precipitate().minimizer
    xa, xb, g_a, g_b, yb = _hessians(MAO_ALLOYS["C"])
    assert yb[3] > 0.5 and yb[0] < 0.01
    h, cols = 1e-5, []
    for j in range(2):
        d = np.zeros(3)
        d[j], d[2] = h, -h
        mu = [pm.solve(xb + s * d, yb, T)[2][0] for s in (1, -1)]
        cols.append(((mu[0][:2] - mu[0][2]) - (mu[1][:2] - mu[1][2])) / (2 * h))
    np.testing.assert_allclose(g_b, np.array(cols).T, rtol=1e-6)
    ya, _, _ = pm.solve(xa, pm.disordered(xa), T)
    np.testing.assert_allclose(pm.hessian(ya, T)[0], g_a, rtol=1e-12)


def test_composition_drift_direction_ignores_the_mobilities():
    """Eq. (45)'s direction G_beta^-1 G_alpha dC / (dC^T G_alpha dC) holds no mobility: a diagonal M and one with
    strong Al-Cr coupling give parallel drifts, so flux coupling sets only how far the precipitate composition
    lies off equilibrium, never in which direction, in the late-stage regime the formula describes."""
    xa, xb, g_a, g_b, _ = _hessians(MAO_ALLOYS["B"])
    dc = (xb - xa)[:2]
    plain = precipitate_composition_shift(3.6e6, dc, np.diag([1e-25, 1e-25]), g_a, g_b, 6.7584e-6, 0.0225)[0]
    coupled = precipitate_composition_shift(3.6e6, dc, [[1e-25, -0.9e-25], [-0.9e-25, 1e-25]], g_a, g_b, 6.7584e-6,
                                            0.0225)[0]
    assert plain @ coupled / (np.linalg.norm(plain) * np.linalg.norm(coupled)) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(coupled) != pytest.approx(np.linalg.norm(plain), rel=0.1)


def test_mean_field_drift_against_mao_table_4():
    """Mao et al. find the gamma' Al and Cr both above their long-time values and falling (Table 4), and note the
    Kuehmann-Voorhees mean-field model gets Al right but not Cr. Eq. (45) with Dupin's Hessians at each alloy's
    tie-line agrees: gamma' is Al-rich (dAl > 0) and Cr-poor (dCr < 0) while it coarsens, in all three alloys.
    In B and C the measured Cr is above its long-time value at every time from 16 h and falls toward it, the
    opposite sign; in A it lies within 0.2 at.% of it throughout. Since the direction holds no mobility, the
    flux coupling Mao et al. invoke cannot reverse it in this regime. (Table 4 prints gamma' Al in alloy C at 16 h
    as 11.0 at.%; the row sums to 100 only with 18.0, used here.)"""
    for name, alloy in MAO_ALLOYS.items():
        xa, xb, g_a, g_b, _ = _hessians(alloy)
        dc = (xb - xa)[:2]
        drift = np.linalg.solve(g_b, g_a @ dc) / float(dc @ g_a @ dc)
        assert drift[0] > 0 > drift[1], name
        al, cr = (np.array(v) for v in MAO_GP_LATE[name])
        assert np.all(al[:-1] >= al[-1]) and al[0] > al[-1], name
        if name == "A":
            assert np.all(np.abs(cr - cr[-1]) <= 0.2)
        else:
            assert np.all(cr[:-1] > cr[-1]) and np.all(np.diff(cr) < 0), name
    assert 75.09 + 18.0 + 6.9 == pytest.approx(100.0, abs=0.05)


# --- composition constraints: Kozeschnik, Svoboda and Fischer, Calphad 28 (2004) 379 ------------------------

from janus_kinetics.multicomponent_precipitation import constrained_evolution_rates, evolution_rates  # noqa: E402

_KSF = NucleationSettings(gamma=0.2, molar_volume=7e-6, lattice_parameter=0.36e-9, misfit=2e7)


def _four_element_classes():
    """Three classes of a precipitate (A,B)2(C,D) in an A-B-C-D matrix, its sublattice holding A + B at 2/3."""
    radius = np.array([1e-9, 3e-9, 8e-9])
    comp = np.array([[0.40, 0.26667, 0.20, 0.13333], [0.50, 0.16667, 0.10, 0.23333], [0.30, 0.36667, 0.25, 0.08333]])
    comp[:, 1] = 2 / 3 - comp[:, 0]
    comp[:, 3] = 1 / 3 - comp[:, 2]
    x0 = np.array([0.05, 0.08, 0.04, 0.83])
    rng = np.random.default_rng(7)
    mu_k = rng.normal(-3e4, 4e3, (3, 4))
    mu_0 = np.array([-4.1e4, -3.6e4, -3.9e4, -2.2e4])
    tracer = np.array([3e-19, 1e-19, 5e-20, 2e-19])
    ptracer = np.array([1e-21, 4e-22, 2e-21, 1e-21])
    return radius, comp, mu_k, x0, mu_0, tracer, ptracer


def _gibbs_rate_and_dissipation(rho, x, mu_k, x0, mu_0, tracer, ptracer, drho, dx, settings, mobility=np.inf):
    """dG/dt by eqs. (9)-(10) and Q by eqs. (11), (13), (14), per class, divided by 4 pi."""
    omega = settings.molar_volume
    c, c0, dc = x / omega, x0 / omega, dx / omega
    dmu = mu_k - mu_0
    g_rho = 2 * rho * settings.gamma + rho**2 * (settings.misfit + (c * dmu).sum(axis=1))
    g_c = rho[:, None] ** 3 / 3 * dmu
    gdot = g_rho * drho + (g_c * dc).sum(axis=1)
    q1 = rho**2 * drho**2 / mobility
    q2 = R * T * rho**5 * (dc**2 / (c * ptracer)).sum(axis=1) / 45
    q3 = R * T * rho**3 * ((drho[:, None] * (c - c0) + rho[:, None] * dc / 3) ** 2 / (c0 * tracer)).sum(axis=1)
    return gdot, q1 + q2 + q3


def test_constrained_rates_with_one_balance_are_part_one():
    """With the single row of ones, eqs. (24)-(32) give part I's rates, :func:`evolution_rates`, to round-off,
    in this Ni-Al-Cr matrix with three gamma' classes."""
    rho = np.array([0.8e-9, 2e-9, 5e-9])
    comp = np.array([[0.16, 0.08, 0.76], [0.18, 0.06, 0.76], [0.15, 0.09, 0.76]])
    x0 = np.array([0.045, 0.145, 0.81])
    mu_k = np.array([[-1.5e5, -4.0e4, -2.9e4], [-1.45e5, -4.2e4, -2.95e4], [-1.52e5, -3.9e4, -2.88e4]])
    mu_0 = np.array([-1.48e5, -4.1e4, -2.92e4])
    tracer, ptracer = np.array([3e-21, 1e-21, 2e-22]), np.array([1e-23, 2e-23, 1e-23])
    plain = evolution_rates(rho, comp, mu_k, x0, mu_0, tracer, ptracer, SETTINGS, T)
    general = constrained_evolution_rates(rho, comp, mu_k, x0, mu_0, tracer, ptracer, np.ones((1, 3)), SETTINGS, T)
    np.testing.assert_allclose(general[0], plain[0], rtol=1e-10)
    np.testing.assert_allclose(general[1], plain[1], rtol=1e-9, atol=1e-12 * np.abs(plain[1]).max())


def test_constrained_rates_keep_the_stoichiometry():
    """A precipitate (A,B)2(C,D): with constraints (1,1,1,1) and (1,1,0,0) its classes change composition only
    within each sublattice, A + B and C + D each fixed (eq. 19), to round-off."""
    radius, comp, mu_k, x0, mu_0, tracer, ptracer = _four_element_classes()
    a = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    drho, dx = constrained_evolution_rates(radius, comp, mu_k, x0, mu_0, tracer, ptracer, a, _KSF, T)
    scale = np.abs(dx).max()
    assert scale > 0
    np.testing.assert_allclose(dx[:, 0] + dx[:, 1], 0, atol=1e-12 * scale)
    np.testing.assert_allclose(dx[:, 2] + dx[:, 3], 0, atol=1e-12 * scale)
    free = constrained_evolution_rates(radius, comp, mu_k, x0, mu_0, tracer, ptracer, a[:1], _KSF, T)[1]
    assert np.abs(free[:, 0] + free[:, 1]).max() > 1e-3 * scale


@pytest.mark.parametrize("mobility", [np.inf, 1e-16])
def test_constrained_rates_dissipate_what_the_gibbs_energy_loses(mobility):
    """The rates make dG/dt + Q = 0 (eq. 17), with G's derivatives of eqs. (9)-(10) and the dissipation of eqs.
    (11), (13) and (14), for an infinite and a finite interface mobility."""
    radius, comp, mu_k, x0, mu_0, tracer, ptracer = _four_element_classes()
    a = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    drho, dx = constrained_evolution_rates(radius, comp, mu_k, x0, mu_0, tracer, ptracer, a, _KSF, T, mobility)
    gdot, q = _gibbs_rate_and_dissipation(radius, comp, mu_k, x0, mu_0, tracer, ptracer, drho, dx, _KSF, mobility)
    assert np.all(q > 0)
    np.testing.assert_allclose(gdot + q, 0, atol=1e-10 * np.abs(q).max())


def test_constrained_rates_depend_only_on_defined_potentials():
    """In a stoichiometric phase only combinations of the chemical potentials are defined: shifting them by w_j a_j
    with sum_j w_j U_j = 0 (U_j = a_j c_k, so that sum_i c_ki mu_ki is kept) leaves every rate unchanged."""
    radius, comp, mu_k, x0, mu_0, tracer, ptracer = _four_element_classes()
    a = np.array([[1.0, 1, 1, 1], [1, 1, 0, 0]])
    base = constrained_evolution_rates(radius, comp, mu_k, x0, mu_0, tracer, ptracer, a, _KSF, T)
    u = comp @ a.T  # U_j per class, in mole fractions
    w = np.column_stack([u[:, 1], -u[:, 0]]) * 5e3
    shifted = constrained_evolution_rates(radius, comp, mu_k + w @ a, x0, mu_0, tracer, ptracer, a, _KSF, T)
    np.testing.assert_allclose(shifted[0], base[0], rtol=1e-10)
    np.testing.assert_allclose(shifted[1], base[1], rtol=1e-8, atol=1e-12 * np.abs(base[1]).max())


def test_interface_mobility_slows_the_interface():
    """A finite interface mobility M adds 1/M to the radius row (eq. 26): every class moves more slowly, and as M
    grows the rates return to part I's."""
    radius, comp, mu_k, x0, mu_0, tracer, ptracer = _four_element_classes()
    a = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    fast = constrained_evolution_rates(radius, comp, mu_k, x0, mu_0, tracer, ptracer, a, _KSF, T)[0]
    slow = constrained_evolution_rates(radius, comp, mu_k, x0, mu_0, tracer, ptracer, a, _KSF, T, 1e-17)[0]
    nearly = constrained_evolution_rates(radius, comp, mu_k, x0, mu_0, tracer, ptracer, a, _KSF, T, 1e-6)[0]
    assert np.all(np.abs(slow) < np.abs(fast)) and np.all(np.sign(slow) == np.sign(fast))
    np.testing.assert_allclose(nearly, fast, rtol=1e-6)
    with pytest.raises(ValueError, match="one coefficient per element"):
        constrained_evolution_rates(radius, comp, mu_k, x0, mu_0, tracer, ptracer, [[1, 1, 1]], _KSF, T)
