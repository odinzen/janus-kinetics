"""B2 ordering and diffusion, checked against Helander and Agren, Acta Mater. 47 (1999) 1141.

Their Figs. 2 and 7 are read from the published figures at 300 dpi; the readings
carry about 0.03 decades of error. The paper used the Al-Ni thermodynamics of
Ansara et al., J. Alloys Compd. 247 (1997) 20, whose B2 is (Al,Ni)(Ni,Va). Two
descriptions are used here: that model, built below from Ansara's Appendix A
(pp. 28-29), and pycalphad's alni_dupin_2001.tdb, which puts Ansara's parameter
values into Dupin's (Al,Ni,Va)(Al,Ni,Va) reformulation (Dupin, Ansara and
Sundman, Calphad 25 (2001) 279, pp. 297-298, matched within 0.6 J/mol). Both give
the same Ni-rich results. Compositions stop at x_Al = 0.49; the vacancy-rich side
is discussed in calphad_kinetics.ordered.
"""
import re

import warnings
from importlib.resources import files

import numpy as np
import pytest
from pycalphad import Database

from calphad_kinetics import R
from calphad_kinetics.datasets import HELANDER1999_ALNI_ORDERING, helander1999_bcc_alni
from calphad_kinetics.ordered import B2Phase, ordering_energy

# Their calculated log10 D*_Ni (Fig. 7, offsets M removed) and log10 D~ (Fig. 2).
FIG7 = {1573.15: {0.40: -12.24, 0.44: -12.86, 0.48: -13.54, 0.49: -13.70},
        1473.15: {0.40: -12.75, 0.44: -13.40, 0.48: -14.16, 0.49: -14.34},
        1373.15: {0.40: -13.33, 0.44: -14.05, 0.48: -14.87, 0.49: -15.07},
        1273.15: {0.40: -14.02, 0.44: -14.80, 0.48: -15.69, 0.49: -15.92}}
FIG2 = {1373.15: {0.365: -12.27, 0.405: -12.83, 0.445: -13.45, 0.475: -13.86},
        1273.15: {0.365: -12.90, 0.405: -13.53, 0.445: -14.22, 0.475: -14.69}}


DUPIN_TDB = files("pycalphad") / "tests" / "databases" / "alni_dupin_2001.tdb"


def ansara_b2() -> Database:
    """Ansara et al. (1997) Appendix A B2 phase, (Al,Ni)(Ni,Va), on the SGTE pure-element
    functions of pycalphad's Dupin file (GHSERAL, GBCCAL, GHSERNI, GBCCNI)."""
    src = DUPIN_TDB.read_text()
    funcs = [re.search(r"FUNCTION\s+" + name + r"\b.*?!", src, re.S).group(0)
             for name in ("GHSERAL", "GBCCAL", "GHSERNI", "GBCCNI")]
    header = "ELEMENT VA VACUUM 0 0 0 !\nELEMENT AL FCC_A1 26.98 0 0 !\nELEMENT NI FCC_A1 58.69 0 0 !\n"
    return Database(header + "TYPE_DEFINITION % SEQ * !\n" + "\n".join(funcs) + """
PHASE B2 % 2 1 1 !
CONSTITUENT B2 :AL,NI:NI,VA: !
PARAMETER G(B2,AL:NI;0) 298.15 GHSERAL#+GHSERNI#-152397.30+26.40575*T; 6000 N !
PARAMETER G(B2,NI:NI;0) 298.15 2*GBCCNI#; 6000 N !
PARAMETER G(B2,AL:VA;0) 298.15 GBCCAL#+1000-T; 6000 N !
PARAMETER G(B2,NI:VA;0) 298.15 GBCCNI#+162397.3-27.40575*T; 6000 N !
PARAMETER G(B2,AL,NI:NI;0) 298.15 -52440.88+11.30117*T; 6000 N !
PARAMETER G(B2,AL,NI:VA;0) 298.15 -52440.88+11.30117*T; 6000 N !
PARAMETER G(B2,AL:NI,VA;0) 298.15 -64024.38+26.49419*T; 6000 N !
PARAMETER G(B2,NI:NI,VA;0) 298.15 -64024.38+26.49419*T; 6000 N !
""")


@pytest.fixture(scope="module", params=["dupin", "ansara"])
def b2(request):
    if not DUPIN_TDB.is_file():
        pytest.skip("pycalphad's alni_dupin_2001.tdb test database is not installed")
    mob, ordering = helander1999_bcc_alni()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if request.param == "dupin":
            return B2Phase(Database(str(DUPIN_TDB)), "BCC_B2", ["AL", "NI"], mob, "BCC_A2", ordering)
        return B2Phase(ansara_b2(), "B2", ["AL", "NI"], mob, "BCC_A2", ordering)


@pytest.fixture(scope="module")
def alni():
    path = files("pycalphad") / "tests" / "databases" / "alni_dupin_2001.tdb"
    if not path.is_file():
        pytest.skip("pycalphad's alni_dupin_2001.tdb test database is not installed")
    mob, ordering = helander1999_bcc_alni()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return B2Phase(Database(str(path)), "BCC_B2", ["AL", "NI"], mob, "BCC_A2", ordering)


@pytest.mark.parametrize("ya", [0.5, 0.7, 0.95, 1.0])
def test_binary_reduces_to_girifalco(ya):
    """Appendix B: with y_A^a + y_A^b = 2 x_A, eq. 12 gives S^2 dQ^order_AAB / 2."""
    x_a = 0.5
    yb = 2 * x_a - ya
    terms = {("A", "A", "B"): 2.0e5}
    got = ordering_energy(terms, "A", ("A", "B"), [ya, 1 - ya], [yb, 1 - yb])
    assert got == pytest.approx((ya - yb) ** 2 * 2.0e5 / 2, rel=1e-12, abs=1e-9)


def test_eq_14_reproduces_eq_9():
    """dQ^order = 2 Q^dis alpha (eq. 14) turns eq. 12 into Q = Q^dis (1 + alpha S^2) (eq. 9)."""
    q_dis, alpha, ya, yb = 250e3, 0.3, 0.9, 0.1
    d_q = ordering_energy({("A", "A", "B"): 2 * q_dis * alpha}, "A", ("A", "B"), [ya, 1 - ya], [yb, 1 - yb])
    assert q_dis + d_q == pytest.approx(q_dis * (1 + alpha * (ya - yb) ** 2), rel=1e-12)


def test_disorder_adds_nothing():
    y = [0.3, 0.65, 0.05]
    for k in ("AL", "NI"):
        assert ordering_energy(HELANDER1999_ALNI_ORDERING, k, ("AL", "NI", "VA"), y, y) == pytest.approx(0.0, abs=1e-9)


def test_shipped_disordered_parameters_match_appendix_a():
    """Phi = R T Theta - Q: Theta = ln(6.5e-5), Q = 215 kJ for Al and ln(1.8e-5), 204 kJ for Ni,
    with 0Q^{Al,Ni} = -327.4 kJ for both."""
    from calphad_kinetics import MobilityModel
    from calphad_kinetics._lattice import SubstitutionalLattice

    mob, _ = helander1999_bcc_alni()
    model = MobilityModel(mob, SubstitutionalLattice.from_database(mob, "BCC_A2", ["AL", "NI"]))
    T = 1300.0
    for x_al in (0.0, 0.3, 1.0):
        x = np.array([x_al, 1 - x_al])
        mix = x_al * (1 - x_al) * 327400
        expected = [R * T * np.log(6.5e-5) - 215000 + mix, R * T * np.log(1.8e-5) - 204000 + mix]
        np.testing.assert_allclose(model.phi(x, T), expected, rtol=1e-12)


@pytest.mark.parametrize("T", sorted(FIG7))
def test_ni_tracer_diffusivity_against_fig_7(b2, T):
    for x_al, ref in FIG7[T].items():
        assert np.log10(b2.tracer_diffusivity([x_al], T)[1]) == pytest.approx(ref, abs=0.08)


@pytest.mark.parametrize("T", sorted(FIG2))
def test_chemical_diffusivity_against_fig_2(b2, T):
    for x_al, ref in FIG2[T].items():
        assert np.log10(b2.interdiffusion_matrix([x_al], T)[0, 0]) == pytest.approx(ref, abs=0.06)


def test_order_slows_ni_by_orders_of_magnitude(alni):
    """Without the ordering terms Ni would diffuse about four decades faster at the B2 composition."""
    mob, _ = helander1999_bcc_alni()
    disordered = B2Phase(alni.db, "BCC_B2", ["AL", "NI"], mob, "BCC_A2", {})
    gap = np.log10(disordered.tracer_diffusivity([0.49], 1273.15)[1] / alni.tracer_diffusivity([0.49], 1273.15)[1])
    assert gap > 3


def test_site_fractions_are_ordered_and_sum_to_one(alni):
    y_a, y_b = alni.site_fractions([0.45], 1273.15)
    assert y_a.sum() == pytest.approx(1.0) and y_b.sum() == pytest.approx(1.0)
    assert abs(y_a[0] - y_b[0]) > 0.8  # Al mostly on one sublattice
    x_al = 0.5 * (y_a[0] + y_b[0]) / (1 - 0.5 * (y_a[2] + y_b[2]))
    assert x_al == pytest.approx(0.45, abs=1e-6)


def test_unknown_species_in_ordering_terms_raise(alni):
    mob, _ = helander1999_bcc_alni()
    with pytest.raises(ValueError, match="FE"):
        B2Phase(alni.db, "BCC_B2", ["AL", "NI"], mob, "BCC_A2", {("NI", "FE", "AL"): 1e5})


def test_vacancy_rich_compositions_warn(alni):
    """Above x_Al = 0.5 the model runs about 0.3 decades above their Fig. 2 (see the
    module docstring), so it says so; the Ni-rich side stays silent."""
    with pytest.warns(UserWarning, match="vacancy"):
        alni.mobility([0.53], 1373.15)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        alni.mobility([0.45], 1373.15)
