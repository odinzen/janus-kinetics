"""Inputs that used to give a silently wrong answer now give the right one or a clear error.

Each case comes from an adversarial review of the package; the comment names what
went wrong before.
"""

import warnings

import numpy as np
import pytest
from pycalphad import Database

from janus_kinetics import (
    Boundary,
    ConvergenceError,
    LocalEquilibrium,
    MissingParameterError,
    MobilityModel,
    PhaseDissolvedError,
    SolutionPhase,
    UnsupportedPhaseError,
    solve_moving_boundary,
    solve_planar,
    uniform_faces,
)
from janus_kinetics._lattice import SubstitutionalLattice
from janus_kinetics.homogenization import LocalEquilibriumTable, solve_homogenized
from janus_kinetics.precipitation import DiluteSolution, Kinetics, solve_kwn
from conftest import HEADER, REGULAR_TERNARY
from test_moving_boundary import TWO_PHASE

T = 1000.0


# --- element order and compositions ---------------------------------------------------------

def test_element_order_is_kept(regular_db):
    """Before, ["B","A","C"] was re-sorted to (A, B, C) and x[0] silently meant A."""
    phase = SolutionPhase(regular_db, "SOL", ["B", "A", "C"])
    assert phase.elements == ("B", "A", "C")
    sorted_phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    np.testing.assert_allclose(phase.chemical_potentials([0.3, 0.2], T),
                               sorted_phase.chemical_potentials([0.2, 0.3], T)[[1, 0, 2]], rtol=1e-12)


@pytest.mark.parametrize("x", [[1.5], [-0.2], [30.0]])
def test_compositions_outside_unit_interval_raise(regular_db, x):
    phase = SolutionPhase(regular_db, "SOL", ["A", "B"])
    with pytest.raises(ValueError, match="must lie in"):
        phase.interdiffusion_matrix(np.array([x]), T)


def test_independent_fractions_summing_above_one_raise(regular_db):
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    with pytest.raises(ValueError, match="must lie in"):
        phase.interdiffusion_matrix(np.array([[0.7, 0.5]]), T)


def test_scalar_composition_for_a_binary(regular_db):
    phase = SolutionPhase(regular_db, "SOL", ["A", "B"])
    np.testing.assert_allclose(phase.mobility(0.3, T), phase.mobility(np.array([0.3]), T))


def test_duplicate_elements_are_refused(regular_db):
    with pytest.raises(UnsupportedPhaseError, match="twice"):
        SolutionPhase(regular_db, "SOL", ["A", "A", "B"])


# --- mobility databases -----------------------------------------------------------------------

WILDCARD = HEADER + """
PHASE SOL % 2 1 1 !
CONSTITUENT SOL :A,B:VA: !
PARAMETER MQ(SOL&A,*:VA;0) 298.15 -150000; 6000 N !
PARAMETER MQ(SOL&B,A:VA;0) 298.15 -160000; 6000 N !
PARAMETER MQ(SOL&B,B:VA;0) 298.15 -140000; 6000 N !
"""


def test_wildcard_mobility_parameters_are_used():
    """Before, MQ(SOL&A,*:VA) was dropped and A had no end members at all."""
    db = Database(WILDCARD)
    model = MobilityModel(db, SubstitutionalLattice.from_database(db, "SOL", ["A", "B"]))
    mq = model.parameters(np.array([[0.3, 0.7], [0.9, 0.1]]), T)[1]
    np.testing.assert_allclose(mq[:, 0], -150000.0, rtol=1e-14)  # composition independent
    np.testing.assert_allclose(mq[:, 1], [-0.3 * 160000 - 0.7 * 140000, -0.9 * 160000 - 0.1 * 140000], rtol=1e-14)


def test_mobility_database_without_the_phase_says_so(regular_db):
    other = Database(HEADER + "PHASE LIQ % 1 1 !\nCONSTITUENT LIQ :A,B,C: !\n")
    with pytest.raises(MissingParameterError, match="has no phase SOL"):
        SolutionPhase(regular_db, "SOL", ["A", "B"], mobility_db=other)


def test_mobility_database_in_another_layout_is_read_in_its_own(regular_db):
    """(A,B,C) mobilities against an (A,B,C)(VA) Gibbs energy are matched on the
    mobility database's own layout, so missing parameters there still raise rather
    than reading as zero (layouts that agree are tested in test_mobility.py)."""
    other = Database(HEADER + "PHASE SOL % 1 1 !\nCONSTITUENT SOL :A,B,C: !\n")
    with pytest.raises(MissingParameterError, match="end-member"):
        SolutionPhase(regular_db, "SOL", ["A", "B"], mobility_db=other)


def test_magnetic_gibbs_energy_warns_that_the_mobility_factor_is_omitted():
    tdb = REGULAR_TERNARY + "PARAMETER TC(SOL,A:VA;0) 298.15 900; 6000 N !\n"
    with pytest.warns(UserWarning, match="ferromagnetic factor"):
        SolutionPhase(Database(tdb), "SOL", ["A", "B"])


def test_unused_parameter_types_are_reported():
    tdb = REGULAR_TERNARY + "PARAMETER DQ(SOL&A,A:VA;0) 298.15 -1000; 6000 N !\n"
    with pytest.warns(UserWarning, match="DQ"):
        SolutionPhase(Database(tdb), "SOL", ["A", "B"])


def test_plain_databases_do_not_warn(regular_db):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SolutionPhase(regular_db, "SOL", ["A", "B", "C"])


# --- boundaries -------------------------------------------------------------------------------

def test_boundary_kinds_and_compositions_are_checked():
    with pytest.raises(ValueError, match="kind"):
        Boundary("Fixed", np.array([0.1]))
    with pytest.raises(ValueError, match="needs a composition"):
        Boundary("fixed")
    assert Boundary("fixed", [0.1, 0.2]).composition.tolist() == [0.1, 0.2]  # lists are accepted
    with pytest.raises(ValueError, match="mole fractions"):
        Boundary.fixed([0.7, 0.5])
    with pytest.raises(ValueError, match="independent mole fractions"):
        solve_planar(uniform_faces(1.0, 4), np.zeros((4, 2)), np.eye(2), 1.0, 0.5, left=Boundary.fixed([0.1, 0.2, 0.7]))


# --- precipitation inputs ---------------------------------------------------------------------

V = 1.6e-29


@pytest.mark.parametrize("kwargs, match", [
    (dict(x_eq=1.0, x_p=1.0), "x_eq"),
    (dict(x_eq=0.01, x_p=1.0, gamma=-0.13), "interface energy"),
    (dict(x_eq=0.01, x_p=1.0, v_at_p=1e-5, v_at_m=1e-5), "atomic volume"),
])
def test_precipitation_thermodynamics_are_checked(kwargs, match):
    args = dict(v_at_p=V, v_at_m=V, gamma=0.13, T=433.0) | kwargs
    with pytest.raises(ValueError, match=match):
        DiluteSolution(**args)


def test_lattice_parameter_in_angstrom_is_refused():
    with pytest.raises(ValueError, match="angstrom"):
        Kinetics(5e-20, 4.04, 1 / V)


def test_gibbs_thomson_has_no_root_below_the_smallest_radius():
    """Before, Newton returned X^i = 1e-298 there and sub-critical particles grew."""
    thermo = DiluteSolution(0.01, 0.25, V, V, 0.13, 433.15)
    r_min = thermo.smallest_radius
    assert np.isnan(thermo.interface_composition(0.9 * r_min))
    xi = float(thermo.interface_composition(1.1 * r_min))
    assert 0.01 < xi < 0.25
    with pytest.raises(ValueError, match="cut-off radius"):
        solve_kwn(thermo, Kinetics(5e-20, 4.04e-10, 1 / V), 0.06, 1e3, np.arange(0.3e-9, 20e-9, 0.05e-9))


# --- homogenization -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_table():
    db = Database(TWO_PHASE)
    alpha = SolutionPhase(db, "ALPHA", ["A", "B"])
    beta = SolutionPhase(db, "BETA", ["A", "B"])
    return LocalEquilibriumTable(db, {"ALPHA": alpha, "BETA": beta}, T)


def test_homogenization_refuses_compositions_beyond_its_table(binary_table):
    """Before, mu was held at the table edge for x_A = 0.999 and the run froze while reporting convergence."""
    with pytest.raises(ValueError, match="outside the tabulated range"):
        binary_table.chemical_potentials(np.array([[0.999]]))
    faces = uniform_faces(4e-4, 200)
    z = 0.5 * (faces[1:] + faces[:-1])
    with pytest.raises(ValueError, match="outside the tabulated range"):
        solve_homogenized(faces, np.where(z < 2e-4, 0.999, 0.70)[:, None], binary_table, 50.0, 0.25)


@pytest.mark.slow
def test_ternary_homogenization_runs_near_the_table_hypotenuse():
    """Before, x_A + x_B = 0.96 was judged outside the table, every step was damped to nothing and
    reported as converged, and the profile never moved."""
    db = Database(TWO_PHASE)
    alpha = SolutionPhase(db, "ALPHA", ["A", "B", "C"])
    beta = SolutionPhase(db, "BETA", ["A", "B", "C"])
    table = LocalEquilibriumTable(db, {"ALPHA": alpha, "BETA": beta}, T, spacing=0.025)
    faces = uniform_faces(8e-4, 80)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z[:, None] < 4e-4, [0.80, 0.16], [0.10, 0.75])
    res = solve_homogenized(faces, x0, table, 100.0, 2.0)
    assert np.max(np.abs(res.x[-1] - x0)) > 0.05
    assert max(res.picard_iterations) > 1


# --- stacks -------------------------------------------------------------------------------------

@pytest.mark.slow
def test_vanishing_layer_under_local_equilibrium_is_reported_as_consumed():
    """Before, the interface Newton failed first and the run ended in a ConvergenceError without results. The
    alpha layer really is used up, near 171 s with 0.1 s, 0.02 s and adaptive steps alike."""
    db = Database(TWO_PHASE)
    alpha = SolutionPhase(db, "ALPHA", ["A", "B"])
    beta = SolutionPhase(db, "BETA", ["A", "B"])
    xl, xr = LocalEquilibrium(alpha, beta, T, [0.56], [0.27]).tie_line()
    with pytest.raises(PhaseDissolvedError) as err:
        solve_moving_boundary(4e-4, 2e-6, np.full(10, 0.7), np.full(200, 0.05),
                              lambda x: alpha.interdiffusion_matrix(x, T), lambda x: beta.interdiffusion_matrix(x, T),
                              LocalEquilibrium(alpha, beta, T, xl, xr), 200.0, 0.1)
    assert err.value.result.times.size >= 1


GAP = HEADER + """
PHASE ALPHA % 1 1 !
CONSTITUENT ALPHA :A,B: !
PHASE BETA % 1 1 !
CONSTITUENT BETA :A,B: !
PARAMETER G(ALPHA,A,B;0) 298.15 22000; 6000 N !
PARAMETER G(BETA,A;0) 298.15 3000; 6000 N !
PARAMETER G(BETA,B;0) 298.15 3000; 6000 N !
PARAMETER G(BETA,A,B;0) 298.15 -20000; 6000 N !
""" + "".join(f"PARAMETER MQ({p}&{e},{h};0) 298.15 -200000; 6000 N !\n" for p in ("ALPHA", "BETA") for e in "AB" for h in "AB")


@pytest.mark.parametrize("guess", [(0.9, 0.78), (0.8, 0.78)])
def test_binary_tie_line_is_damped(guess):
    """Alpha has a miscibility gap; undamped Newton diverged from these starts.
    pycalphad gives alpha 0.99465 / beta 0.77948 for the A-rich field."""
    db = Database(GAP)
    alpha = SolutionPhase(db, "ALPHA", ["A", "B"])
    beta = SolutionPhase(db, "BETA", ["A", "B"])
    xl, xr = LocalEquilibrium(alpha, beta, T, [guess[0]], [guess[1]]).tie_line()
    assert xl[0] == pytest.approx(0.99465, abs=1e-4)
    assert xr[0] == pytest.approx(0.77948, abs=1e-4)
