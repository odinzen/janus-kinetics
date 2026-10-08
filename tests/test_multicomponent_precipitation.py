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

from calphad_kinetics import R, SolutionPhase, campbell2002_fcc_ni
from calphad_kinetics.multicomponent_precipitation import (
    K_B,
    N_A,
    NucleationSettings,
    PrecipitateThermo,
    growth_rates,
    nucleation,
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
    from calphad_kinetics.multicomponent_precipitation import _merge_closest, _split_most_populated

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
    and no adjustment, the model gives 1.57 nm, 3.0e24 m^-3 and 5.4 %."""
    res = _sudbrack_run()
    i = list(HOURS).index(4)
    assert res.volume_fraction[i] == pytest.approx(0.052, rel=0.10)
    assert res.number_density[i] == pytest.approx(3.2e24, rel=0.15)
    assert res.mean_radius[i] == pytest.approx(1.27e-9, rel=0.30)


def test_coarsening_rate_against_the_papers_own_model():
    """Sudbrack et al. (2007) p. 126: Campbell mobilities give K(phi) = 1.29e-30 m^3/s with their f(phi) = 2.3,
    so 5.6e-31 for isolated particles, 15 times the measured K = 8.8e-32 (they attribute the gap to diffusion
    through the gamma' as well). Between 256 and 1024 h the model here gives 4.0e-31: within a factor of 1.5 of
    the paper's model and 4.5 times faster than the measurement."""
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
    from calphad_kinetics.multicomponent_precipitation import evolution_rates

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
    from calphad_kinetics.multicomponent_precipitation import evolution_rates

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
    coarsens 6 times faster than measured between 256 and 1024 h, as for Ni-5.2Al-14.2Cr (4.5 times): the gap
    follows the mobilities, not the alloy."""
    s = NucleationSettings(gamma=0.024, molar_volume=SETTINGS.molar_volume, lattice_parameter=SETTINGS.lattice_parameter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_multicomponent_precipitation(_matrix(), _precipitate(), [0.075, 0.085, 0.840], T, HOURS[-1] * 3600,
                                                 s, HOURS * 3600, precipitate_tracer=[2.7e-23, 1.4e-23, 2.7e-23])
    i1, i, j = list(HOURS).index(1), list(HOURS).index(256), list(HOURS).index(1024)
    assert res.mean_radius[i1] == pytest.approx(1.24e-9, rel=0.10)
    k = (res.mean_radius[j] ** 3 - res.mean_radius[i] ** 3) / ((HOURS[j] - HOURS[i]) * 3600)
    assert 4 < k / 1.84e-31 < 8
