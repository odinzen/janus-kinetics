"""Andersson-Agren mobility evaluation from MQ/MF database parameters."""

import numpy as np
import pytest
from pycalphad import Database, calculate

from janus_kinetics import MissingParameterError, MobilityModel, R, UnsupportedPhaseError
from janus_kinetics._lattice import SubstitutionalLattice
from conftest import HEADER

T = 1200.0

# Every parameter is written with constituents out of alphabetical order to pin the
# convention. pycalphad sorts them, so order 1 multiplies (x_A - x_B), not (x_B - x_A).
RK_TDB = HEADER + """
PHASE SOL % 1 1 !
CONSTITUENT SOL :A,B,C: !
PARAMETER MQ(SOL&A,A;0) 298.15 -100000; 6000 N !
PARAMETER MQ(SOL&A,B;0) 298.15 -120000; 6000 N !
PARAMETER MQ(SOL&A,C;0) 298.15 -90000; 6000 N !
PARAMETER MF(SOL&A,A;0) 298.15 R*T*LN(2E-5); 6000 N !
PARAMETER MF(SOL&A,B;0) 298.15 -20*T; 6000 N !
PARAMETER MQ(SOL&A,B,A;0) 298.15 11000; 6000 N !
PARAMETER MQ(SOL&A,B,A;1) 298.15 -7000+2*T; 6000 N !
PARAMETER MQ(SOL&A,B,A;2) 298.15 3000; 6000 N !
PARAMETER MF(SOL&A,C,B;1) 298.15 -6*T; 6000 N !
PARAMETER MQ(SOL&A,C,B,A;1) 298.15 50000; 6000 N !
PARAMETER MQ(SOL&B,A;0) 298.15 -110000; 6000 N !
PARAMETER MQ(SOL&B,B;0) 298.15 -105000; 6000 N !
PARAMETER MQ(SOL&B,C;0) 298.15 -95000; 6000 N !
PARAMETER MQ(SOL&C,A;0) 298.15 -130000; 6000 N !
PARAMETER MQ(SOL&C,B;0) 298.15 -125000; 6000 N !
PARAMETER MQ(SOL&C,C;0) 298.15 -115000; 6000 N !
"""

GRID = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.45, 0.45, 0.1]])


def phi_a_by_hand(x, T):
    """Phi_A written out term by term: Andersson and Agren (1992) eq. (67) plus the ternary term of Jonsson (1995) eq. (5)-(6)."""
    xa, xb, xc = x
    mq = xa * -100000 + xb * -120000 + xc * -90000
    mq += xa * xb * (11000 + (-7000 + 2 * T) * (xa - xb) + 3000 * (xa - xb) ** 2)
    v_b = xb + (1 - xa - xb - xc) / 3  # order 1 of sorted (A, B, C) is B
    mq += xa * xb * xc * v_b * 50000
    mf = xa * R * T * np.log(2e-5) + xb * -20 * T
    mf += xb * xc * (-6 * T) * (xb - xc)
    return mf, mq


@pytest.fixture(scope="module")
def rk_model():
    db = Database(RK_TDB)
    return MobilityModel(db, SubstitutionalLattice.from_database(db, "SOL", ["A", "B", "C"]))


def test_redlich_kister_and_ternary_terms(rk_model):
    mf, mq = rk_model.parameters(GRID, T)
    for i, x in enumerate(GRID):
        mf_ref, mq_ref = phi_a_by_hand(x, T)
        assert mf[i, 0] == pytest.approx(mf_ref, rel=1e-12)
        assert mq[i, 0] == pytest.approx(mq_ref, rel=1e-12)


def test_mobility_definition(rk_model):
    x = GRID[0]
    mf_ref, mq_ref = phi_a_by_hand(x, T)
    m_ref = np.exp((mf_ref + mq_ref) / (R * T)) / (R * T)
    assert rk_model.mobility(x, T)[0] == pytest.approx(m_ref, rel=1e-12)
    assert rk_model.mobility(x, T, magnetic_factor=0.5)[0] == pytest.approx(0.5 * m_ref, rel=1e-12)
    assert rk_model.tracer_diffusivity(x, T)[0] == pytest.approx(R * T * m_ref, rel=1e-12)


def test_independent_input_matches_full_input(rk_model):
    np.testing.assert_allclose(rk_model.phi(GRID[:, :2], T), rk_model.phi(GRID, T), rtol=1e-14)


def test_ordering_convention_matches_pycalphad_excess_energy():
    """The same numbers as G parameters must give the same polynomial through pycalphad's own GM."""
    body = """
PHASE SOL % 1 1 !
CONSTITUENT SOL :A,B,C: !
PARAMETER G(SOL,B,A;0) 298.15 11000; 6000 N !
PARAMETER G(SOL,B,A;1) 298.15 -7000+2*T; 6000 N !
PARAMETER G(SOL,B,A;2) 298.15 3000; 6000 N !
PARAMETER G(SOL,C,B,A;1) 298.15 50000; 6000 N !
"""
    db = Database(HEADER + body)
    rk = Database(RK_TDB)
    model = MobilityModel(rk, SubstitutionalLattice.from_database(rk, "SOL", ["A", "B", "C"]))
    _, mq = model.parameters(GRID, T)
    endmember = GRID @ np.array([-100000.0, -120000.0, -90000.0])
    res = calculate(db, ["A", "B", "C"], "SOL", T=T, P=101325, N=1, points=GRID, output="GM")
    ideal = R * T * (GRID * np.log(GRID)).sum(axis=1)
    excess = np.squeeze(res.GM.values) - ideal
    np.testing.assert_allclose(mq[:, 0] - endmember, excess, rtol=1e-10)


def test_missing_end_member_is_refused():
    db = Database(RK_TDB.replace("PARAMETER MQ(SOL&C,B;0) 298.15 -125000; 6000 N !", ""))
    with pytest.raises(MissingParameterError, match="C diffusing in pure B"):
        MobilityModel(db, SubstitutionalLattice.from_database(db, "SOL", ["A", "B", "C"]))


def test_parameters_of_inactive_elements_are_ignored(rk_model):
    db = Database(RK_TDB)
    binary = MobilityModel(db, SubstitutionalLattice.from_database(db, "SOL", ["A", "B"]))
    x = np.array([[0.3, 0.7], [0.8, 0.2]])
    x3 = np.column_stack([x, np.zeros(len(x))])
    np.testing.assert_allclose(binary.phi(x, T), rk_model.phi(x3, T)[:, :2], rtol=1e-12)


def test_unsupported_constitution_is_refused():
    db = Database(HEADER + "PHASE TWO % 2 1 1 !\nCONSTITUENT TWO :A,B:A,C: !\n")
    with pytest.raises(UnsupportedPhaseError):
        SubstitutionalLattice.from_database(db, "TWO", ["A", "B", "C"])


QUATERNARY = HEADER.replace("ELEMENT C BLANK 1 0 0 !", "ELEMENT C BLANK 1 0 0 !\nELEMENT D BLANK 1 0 0 !") + """
PHASE SOL % 1 1 !
CONSTITUENT SOL :A,B,C,D: !
""" + "".join(f"PARAMETER MQ(SOL&{a},{b};0) 298.15 -100000; 6000 N !\n" for a in "ABCD" for b in "ABCD")


def ternary_term(extra: str, x: np.ndarray, elements=("A", "B", "C", "D")) -> np.ndarray:
    db = Database(QUATERNARY + extra)
    model = MobilityModel(db, SubstitutionalLattice.from_database(db, "SOL", list(elements)))
    return model.parameters(x, T)[1] + 100000.0  # remove the end members, leave the ternary term


def test_one_elements_ternary_terms_do_not_change_another_elements_mobility():
    x = np.array([0.2, 0.3, 0.4, 0.1])
    only_a = ternary_term("PARAMETER MQ(SOL&A,A,B,C;0) 298.15 30000; 6000 N !\n", x)
    with_b = ternary_term("PARAMETER MQ(SOL&A,A,B,C;0) 298.15 30000; 6000 N !\n"
                          "PARAMETER MQ(SOL&B,A,B,C;1) 298.15 5000; 6000 N !\n", x)
    assert only_a[0] == pytest.approx(0.2 * 0.3 * 0.4 * 30000, rel=1e-12)  # order 0 alone: symmetric term
    assert with_b[0] == pytest.approx(only_a[0], rel=1e-12)


def test_ternary_weights_with_a_fourth_element_present():
    """v_s = x_s + (1 - x_i - x_j - x_k)/3 (Jonsson 1995, eq. 6); the 1/3 share only matters with x_D > 0."""
    x = np.array([0.2, 0.3, 0.4, 0.1])
    got = ternary_term("PARAMETER MQ(SOL&C,A,B,C;0) 298.15 1000; 6000 N !\n"
                       "PARAMETER MQ(SOL&C,A,B,C;1) 298.15 2000; 6000 N !\n"
                       "PARAMETER MQ(SOL&C,A,B,C;2) 298.15 4000; 6000 N !\n", x)
    rest = 0.1 / 3
    expected = 0.2 * 0.3 * 0.4 * ((0.2 + rest) * 1000 + (0.3 + rest) * 2000 + (0.4 + rest) * 4000)
    assert got[2] == pytest.approx(expected, rel=1e-12)


def test_quaternary_interaction_parameters_are_refused():
    with pytest.raises(UnsupportedPhaseError, match="4 elements"):
        ternary_term("PARAMETER MQ(SOL&A,A,B,C,D;0) 298.15 1000; 6000 N !\n", np.array([0.25] * 4))


def test_mobility_layout_need_not_match_the_thermodynamic_one(ideal_db, regular_db):
    """SOL is (A,B,C) in ideal_db and (A,B,C)(VA) in regular_db with the same mobility
    numbers; either pairing of thermodynamic and mobility database gives one mobility."""
    from janus_kinetics import SolutionPhase

    x = np.array([[0.2, 0.3], [0.6, 0.1], [0.05, 0.9]])
    ref = SolutionPhase(regular_db, "SOL", ["A", "B", "C"]).mobility(x, T)
    mixed = SolutionPhase(regular_db, "SOL", ["A", "B", "C"], mobility_db=ideal_db).mobility(x, T)
    np.testing.assert_allclose(mixed, ref, rtol=1e-13)
    reverse = SolutionPhase(ideal_db, "SOL", ["A", "B", "C"], mobility_db=regular_db).mobility(x, T)
    np.testing.assert_allclose(reverse, SolutionPhase(ideal_db, "SOL", ["A", "B", "C"]).mobility(x, T), rtol=1e-13)
    np.testing.assert_allclose(reverse, ref, rtol=1e-13)
