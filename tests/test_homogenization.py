"""Homogenization model of Larsson and Hoglund, Calphad 33 (2009) 495.

The paper's multiphase examples use commercial databases, so they are not
reproduced. The checks are the mathematical properties of the averaging
functions, the single-phase limit (which must match the phase-1 solver), and a
binary two-phase couple, where no two-phase zone can grow at constant T and the
model must converge to the sharp-interface solution.
"""

import numpy as np
import pytest
from pycalphad import Database

from calphad_kinetics import SolutionPhase, solve_moving_boundary, solve_planar, uniform_faces
from calphad_kinetics.homogenization import LocalEquilibriumTable, average_kinetic_factor, solve_homogenized
from calphad_kinetics.moving_boundary import LocalEquilibrium
from test_moving_boundary import TWO_PHASE

T = 1000.0
BOUNDS = ("wiener_lower", "hashin_shtrikman_lower", "hashin_shtrikman_upper", "wiener_upper")


def test_bounds_are_ordered():
    """Wiener lower <= HS lower <= HS upper <= Wiener upper for any phase mixture."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        p = rng.integers(2, 5)
        gamma = 10.0 ** rng.uniform(-20, -14, size=(p, 3))
        f = rng.dirichlet(np.ones(p))
        vals = [average_kinetic_factor(gamma, f, how) for how in BOUNDS]
        for lo, hi in zip(vals, vals[1:]):
            assert np.all(lo <= hi * (1 + 1e-12))


def test_single_phase_gives_its_own_factor():
    gamma = np.array([[3e-17, 5e-16], [8e-18, 2e-15]])
    for how in BOUNDS:
        np.testing.assert_allclose(average_kinetic_factor(gamma, np.array([1.0, 0.0]), how), gamma[0], rtol=1e-12)


def test_two_phase_hashin_shtrikman_closed_form():
    """Eqs. (8)-(9) with two phases equal G1 + f2 / (1/(G2 - G1) + f1/(3 G1))."""
    g1, g2, f2 = 2e-17, 9e-16, 0.3
    gamma, f = np.array([[g1], [g2]]), np.array([1 - f2, f2])
    lower = average_kinetic_factor(gamma, f, "hashin_shtrikman_lower")[0]
    assert lower == pytest.approx(g1 + f2 / (1 / (g2 - g1) + (1 - f2) / (3 * g1)), rel=1e-12)
    upper = average_kinetic_factor(gamma, f, "hashin_shtrikman_upper")[0]
    assert upper == pytest.approx(g2 + (1 - f2) / (1 / (g1 - g2) + f2 / (3 * g2)), rel=1e-12)


def test_wiener_bounds_are_the_arithmetic_and_harmonic_means():
    gamma, f = np.array([[1e-16], [4e-16]]), np.array([0.25, 0.75])
    assert average_kinetic_factor(gamma, f, "wiener_upper")[0] == pytest.approx(0.25e-16 + 3e-16)
    assert average_kinetic_factor(gamma, f, "wiener_lower")[0] == pytest.approx(1 / (0.25 / 1e-16 + 0.75 / 4e-16))


@pytest.fixture(scope="module")
def binary():
    db = Database(TWO_PHASE)
    alpha = SolutionPhase(db, "ALPHA", ["A", "B"])
    beta = SolutionPhase(db, "BETA", ["A", "B"])
    return alpha, beta, LocalEquilibriumTable(db, {"ALPHA": alpha, "BETA": beta}, T, spacing=0.0025)


def test_table_matches_the_single_phase_model(binary):
    """At grid points inside the alpha field, mu and Gamma* = M x are the phase's own."""
    alpha, _, table = binary
    x = np.array([[0.7], [0.9]])
    np.testing.assert_allclose(table.chemical_potentials(x), alpha.chemical_potentials(x, T), rtol=1e-9)
    full = np.column_stack([x, 1 - x])
    np.testing.assert_allclose(table.kinetic_factor(x), alpha.mobility(x, T) * full, rtol=1e-9)
    assert np.all(table.phase_fractions(x)["ALPHA"] == pytest.approx(1.0))


def test_single_phase_couple_matches_phase1_solver(binary):
    alpha, _, table = binary
    faces = uniform_faces(4e-4, 200)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 2e-4, 0.95, 0.65)[:, None]
    homog = solve_homogenized(faces, x0, table, 200.0, 1.0)
    ref = solve_planar(faces, x0, lambda x: alpha.interdiffusion_matrix(x, T), 200.0, 1.0)
    assert np.max(np.abs(homog.x[-1] - ref.x[-1])) < 2e-3


def alpha_layer_error(binary, n_cells):
    """|alpha thickness from the homogenized run - sharp-interface position| for an alpha/beta couple."""
    alpha, beta, table = binary
    half, t1 = 4e-4, 200.0
    faces = uniform_faces(2 * half, n_cells)
    z = 0.5 * (faces[1:] + faces[:-1])
    homog = solve_homogenized(faces, np.where(z < half, 0.8, 0.1)[:, None], table, t1, t1 / 200)
    inv = homog.inventory()
    assert abs(inv[-1, 0] - inv[0, 0]) < 1e-12 * inv[0, 0]
    thickness = np.sum(table.phase_fractions(homog.x[-1])["ALPHA"] * np.diff(faces))
    cond = LocalEquilibrium(alpha, beta, T, [0.56], [0.27])
    sharp = solve_moving_boundary(2 * half, half, np.full(n_cells // 2, 0.8), np.full(n_cells // 2, 0.1),
                                  lambda x: alpha.interdiffusion_matrix(x, T), lambda x: beta.interdiffusion_matrix(x, T),
                                  cond, t1, t1 / 200)
    shift = sharp.interface[-1] - half
    assert shift > 5e-6  # the interface moved several cells
    return abs(thickness - sharp.interface[-1]) / shift


def test_binary_two_phase_couple_converges_to_sharp_interface(binary):
    coarse, fine = alpha_layer_error(binary, 400), alpha_layer_error(binary, 800)
    assert coarse < 0.12
    assert fine < 0.6 * coarse


@pytest.fixture(scope="module")
def ternary_run():
    db = Database(TWO_PHASE)
    alpha = SolutionPhase(db, "ALPHA", ["A", "B", "C"])
    beta = SolutionPhase(db, "BETA", ["A", "B", "C"])
    table = LocalEquilibriumTable(db, {"ALPHA": alpha, "BETA": beta}, T, spacing=0.025)
    faces = uniform_faces(8e-4, 160)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z[:, None] < 4e-4, [0.75, 0.15], [0.10, 0.75])
    return table, faces, solve_homogenized(faces, x0, table, 400.0, 2.0)


def test_ternary_conserves_every_element(ternary_run):
    _, _, res = ternary_run
    inv = res.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-12 * np.abs(inv[0]).max()


def test_ternary_phase_fractions_are_physical(ternary_run):
    table, _, res = ternary_run
    f = table.phase_fractions(res.x[-1])
    total = f["ALPHA"] + f["BETA"]
    assert np.all(total == pytest.approx(1.0, abs=1e-9))
    assert np.all((f["ALPHA"] >= -1e-12) & (f["ALPHA"] <= 1 + 1e-12))


def test_quaternary_tables_are_refused():
    tdb = (TWO_PHASE.replace("ELEMENT C BLANK 1 0 0 !", "ELEMENT C BLANK 1 0 0 !\nELEMENT D BLANK 1 0 0 !")
           .replace(":A,B,C:", ":A,B,C,D:"))
    tdb += "".join(f"PARAMETER MQ({ph}&{a},{b};0) 298.15 -150000; 6000 N !\n"
                   for ph in ("ALPHA", "BETA") for a in "ABCD" for b in "ABCD" if "D" in (a, b))
    db = Database(tdb)
    phases = {name: SolutionPhase(db, name, ["A", "B", "C", "D"]) for name in ("ALPHA", "BETA")}
    with pytest.raises(ValueError, match="binaries and ternaries"):
        LocalEquilibriumTable(db, phases, T)
