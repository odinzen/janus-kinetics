"""Benchmark against Jonsson, ISIJ International 35 (1995) 1415-1421.

Parameters are transcribed from the Appendix on p. 1421 into
src/calphad_kinetics/data/jonsson1995_bcc_crfeni_mob.tdb. Here eqs. (3)-(6) of the paper are
evaluated independently with the same Appendix values held as plain numbers,
so the test checks the TDB reading, the Redlich-Kister and ternary expansion,
and the split M_B = exp(Theta_B) exp(-Q_B/RT) / RT.
"""

import numpy as np
import pytest

from calphad_kinetics import MobilityModel, R
from calphad_kinetics._lattice import SubstitutionalLattice

R_PAPER = 8.31451  # J/(mol K), stated at the head of the Appendix

ELEMENTS = ("CR", "FE", "NI")


def appendix(r):
    """Appendix values as {B: (Theta end members, Q end members, binary Theta, binary Q, ternary Q)}."""
    return {
        "CR": (
            {"CR": -35.6 / r, "FE": np.log(8.5e-5), "NI": np.log(8.5e-5)},
            {"CR": 407000.0, "FE": 218000.0, "NI": 218000.0},
            {("CR", "FE"): (-116 / r, 37.5 / r)},
            {("CR", "FE"): (-361000.0, -2820.0), ("CR", "NI"): (-350000.0, 0.0), ("FE", "NI"): (-150000.0, 0.0)},
            {"FE": 2400000.0},
        ),
        "FE": (
            {"CR": -17.2 / r, "FE": np.log(4.6e-5), "NI": np.log(4.6e-5)},
            {"CR": 407000.0, "FE": 218000.0, "NI": 218000.0},
            {("CR", "FE"): (-117 / r, 348 / r)},
            {("CR", "FE"): (-267000.0, 416000.0), ("CR", "NI"): (-350000.0, 0.0), ("FE", "NI"): (-150000.0, 0.0)},
            {"FE": 1400000.0},
        ),
        "NI": (
            {"CR": -17.2 / r, "FE": np.log(1.8e-5), "NI": np.log(1.8e-5)},
            {"CR": 407000.0, "FE": 204000.0, "NI": 204000.0},
            {("CR", "FE"): (10 / r, 0.0)},
            {("CR", "FE"): (88000.0, 0.0), ("CR", "NI"): (-350000.0, 0.0), ("FE", "NI"): (-150000.0, 0.0)},
            {"FE": 500000.0},
        ),
    }


def eq5(x, end, binary, ternary):
    """Eq. (5) with m = 1 and eq. (6) for v^s_ijk."""
    xs = dict(zip(ELEMENTS, x))
    val = sum(xs[i] * end[i] for i in ELEMENTS)
    for (i, j), (p0, p1) in binary.items():
        val += xs[i] * xs[j] * (p0 + p1 * (xs[i] - xs[j]))
    rest = (1 - sum(xs.values())) / 3
    for s, p in ternary.items():
        val += xs["CR"] * xs["FE"] * xs["NI"] * (xs[s] + rest) * p
    return val


def mobility_eq3(x, T, r):
    """Eq. (3) with the magnetic factor set to one."""
    out = []
    for b, (theta, q, b_theta, b_q, t_q) in appendix(r).items():
        th = eq5(x, theta, b_theta, {})
        qq = eq5(x, q, b_q, t_q)
        out.append(np.exp(th) * np.exp(-qq / (r * T)) / (r * T))
    return np.array(out)


@pytest.fixture(scope="module")
def model(jonsson_db):
    return MobilityModel(jonsson_db, SubstitutionalLattice.from_database(jonsson_db, "BCC_A2", ELEMENTS, dependent="NI"))


COMPOSITIONS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.25, 0.70, 0.05],
                         [0.50, 0.40, 0.10], [0.18, 0.78, 0.04], [0.3, 0.3, 0.4]])


@pytest.mark.parametrize("T", [1100.0, 1473.15, 1800.0])
def test_mobilities_match_paper_equations(model, T):
    got = model.mobility(COMPOSITIONS, T)
    same_r = np.array([mobility_eq3(x, T, R) for x in COMPOSITIONS])
    np.testing.assert_allclose(got, same_r, rtol=1e-9)
    # With the paper's R = 8.31451 instead of pycalphad's 8.3145 the exponent
    # Q/RT ~ 40 shifts by ~1e-6 relative, a few 1e-5 in M.
    paper_r = np.array([mobility_eq3(x, T, R_PAPER) for x in COMPOSITIONS])
    np.testing.assert_allclose(got, paper_r, rtol=2e-4)


def test_cr_tracer_in_pure_cr_against_fig1(model):
    """Fig. 1 (p. 1417), assessed line read by eye, gives log10 D*_Cr = -21.0 at the right axis end.

    The bottom axis is labelled 10^4/T but its scale matches the top axis only as
    10^5/T, so the axis end at 90 is T = 1111 K and the line start near 46 is 2174 K.
    Reading precision is about 0.2 in log10 D.
    """
    x = np.array([1.0, 0.0, 0.0])
    d_end = model.tracer_diffusivity(x, 1e5 / 90)[0]
    d_start = model.tracer_diffusivity(x, 1e5 / 46)[0]
    assert np.log10(d_end) == pytest.approx(-21.0, abs=0.2)
    assert np.log10(d_start) == pytest.approx(-11.6, abs=0.2)


def test_fe_tracer_in_pure_cr_against_fig2(model):
    """Fig. 2 (p. 1418), assessed line read by eye, gives -12.7 at 10^4/T = 5.55 and -21.15 at 9.5."""
    x = np.array([1.0, 0.0, 0.0])
    assert np.log10(model.tracer_diffusivity(x, 1e4 / 5.55)[1]) == pytest.approx(-12.7, abs=0.2)
    assert np.log10(model.tracer_diffusivity(x, 1e4 / 9.5)[1]) == pytest.approx(-21.15, abs=0.2)


@pytest.mark.skip(reason="needs the ferromagnetic factor mg-Omega of B. Jonsson, Z. Metallkd. 83 (1992) 349, "
                         "which ISIJ Int. 35 (1995) 1415 cites for it; the restatement in Helander and Agren, "
                         "Acta Mater. 47 (1999) 1141, eq. (8), is not dimensionally consistent as printed")
def test_ferromagnetic_factor():
    pass


@pytest.mark.skip(reason="needs the SGTE solution database (1994) used for the thermodynamic factor in "
                         "Jonsson, ISIJ Int. 35 (1995) 1415, Figs. 3-4, plus mg-Omega; neither is available locally")
def test_chemical_diffusivity_fe_cr_against_fig4():
    pass
