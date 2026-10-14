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

from janus_kinetics import Boundary, ConvergenceError, SolutionPhase, solve_moving_boundary, solve_planar, uniform_faces
from janus_kinetics.homogenization import (
    LocalEquilibriumTable,
    OnDemandEquilibriumTable,
    average_kinetic_factor,
    solve_homogenized,
)
from janus_kinetics.moving_boundary import Layer, LocalEquilibrium, solve_layers
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


@pytest.mark.slow
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


# ------------------------------------------------------------- anneals with a temperature table

from janus_kinetics import LocalEquilibrium, TemperatureSchedule  # noqa: E402
from janus_kinetics.homogenization import TemperatureTable  # noqa: E402

ANNEAL = TemperatureSchedule([0.0, 50.0, 150.0, 200.0], [950.0, 1050.0, 1050.0, 950.0])


@pytest.fixture(scope="module")
def anneal_table():
    db = Database(TWO_PHASE)
    alpha = SolutionPhase(db, "ALPHA", ["A", "B"])
    beta = SolutionPhase(db, "BETA", ["A", "B"])
    return alpha, beta, TemperatureTable(db, {"ALPHA": alpha, "BETA": beta}, np.arange(940.0, 1061.0, 10.0))


def test_temperature_table_reproduces_its_grid_points(anneal_table):
    _, _, table = anneal_table
    x = np.array([[0.7], [0.9]])
    np.testing.assert_allclose(table.at(1000.0).chemical_potentials(x), table.tables[6].chemical_potentials(x), rtol=1e-14)
    with pytest.raises(ValueError, match="outside the tabulated"):
        table.at(1100.0)


def test_single_phase_couple_under_an_anneal_matches_phase1_solver(anneal_table):
    alpha, _, table = anneal_table
    faces = uniform_faces(4e-4, 200)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 2e-4, 0.95, 0.75)[:, None]
    homog = solve_homogenized(faces, x0, table, 200.0, 1.0, temperature=ANNEAL)
    ref = solve_planar(faces, x0, alpha.interdiffusion_function(ANNEAL), 200.0, 1.0)
    assert np.max(np.abs(homog.x[-1] - ref.x[-1])) < 2e-3


@pytest.mark.slow
def test_two_phase_couple_under_an_anneal_matches_sharp_interface(anneal_table):
    """The alpha layer from the homogenized run against the sharp interface whose tie-line follows the schedule."""
    alpha, beta, table = anneal_table
    half, n = 4e-4, 400
    faces = uniform_faces(2 * half, n)
    z = 0.5 * (faces[1:] + faces[:-1])
    homog = solve_homogenized(faces, np.where(z < half, 0.8, 0.1)[:, None], table, 200.0, 1.0, temperature=ANNEAL)
    thickness = np.sum(table.at(950.0).phase_fractions(homog.x[-1])["ALPHA"] * np.diff(faces))
    xl, xr = LocalEquilibrium(alpha, beta, 950.0, [0.56], [0.27]).tie_line()
    sharp = solve_moving_boundary(2 * half, half, np.full(n // 2, 0.8), np.full(n // 2, 0.1),
                                  alpha.interdiffusion_function(ANNEAL), beta.interdiffusion_function(ANNEAL),
                                  LocalEquilibrium(alpha, beta, ANNEAL, xl, xr), 200.0, 1.0)
    shift = sharp.interface[-1] - half
    assert shift > 5e-6
    assert abs(thickness - sharp.interface[-1]) < 0.08 * shift  # 5.5% at 400 cells, halving with refinement
    inv = homog.inventory()
    assert abs(inv[-1, 0] - inv[0, 0]) < 1e-12 * inv[0, 0]


def test_single_temperature_table_refuses_a_schedule(binary):
    _, _, table = binary
    with pytest.raises(ValueError, match="TemperatureTable"):
        solve_homogenized(uniform_faces(1e-4, 10), np.full((10, 1), 0.8), table, 1.0, 0.5, temperature=ANNEAL)


# ------------------------------------------------------------- local equilibrium on demand (section 2.5)

def quaternary_db():
    tdb = (TWO_PHASE.replace("ELEMENT C BLANK 1 0 0 !", "ELEMENT C BLANK 1 0 0 !\nELEMENT D BLANK 1 0 0 !")
           .replace(":A,B,C:", ":A,B,C,D:"))
    tdb += "".join(f"PARAMETER MQ({ph}&{a},{b};0) 298.15 -150000; 6000 N !\n"
                   for ph in ("ALPHA", "BETA") for a in "ABCD" for b in "ABCD" if "D" in (a, b))
    db = Database(tdb)
    return db, {name: SolutionPhase(db, name, ["A", "B", "C", "D"]) for name in ("ALPHA", "BETA")}


@pytest.mark.slow
def test_on_demand_table_equals_the_full_table_in_a_binary(binary):
    """On a 1D grid both interpolate linearly between the same nodes, so values agree
    and a two-phase couple runs identically, with fewer equilibria."""
    alpha, beta, table = binary
    od = OnDemandEquilibriumTable(table_db(), {"ALPHA": alpha, "BETA": beta}, T, spacing=0.0025)
    x = np.array([[0.1234], [0.5011], [0.77777], [0.9013]])
    np.testing.assert_allclose(od.chemical_potentials(x), table.chemical_potentials(x), rtol=1e-12)
    np.testing.assert_allclose(od.kinetic_factor(x), table.kinetic_factor(x), rtol=1e-12)
    faces = uniform_faces(8e-4, 200)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 4e-4, 0.8, 0.1)[:, None]
    full = solve_homogenized(faces, x0, table, 200.0, 1.0)
    lazy = solve_homogenized(faces, x0, od, 200.0, 1.0)
    assert np.max(np.abs(full.x[-1] - lazy.x[-1])) < 1e-10
    assert od.calculations < len(table.points)


def table_db():
    return Database(TWO_PHASE)


def test_on_demand_slopes_are_those_of_its_interpolant():
    db = table_db()
    phases = {name: SolutionPhase(db, name, ["A", "B", "C"]) for name in ("ALPHA", "BETA")}
    od = OnDemandEquilibriumTable(db, phases, T, spacing=0.025)
    x = np.array([[0.4123, 0.2311], [0.7011, 0.1532]])
    g = od.dmu_dx(x)
    for j in range(2):
        e = np.zeros(2)
        e[j] = 1e-6  # small enough to stay on one simplex
        fd = (od.chemical_potentials(x + e) - od.chemical_potentials(x - e)) / 2e-6
        np.testing.assert_allclose(g[:, :, j], fd, rtol=1e-6, atol=1e-6)


@pytest.mark.slow
def test_on_demand_ternary_agrees_with_the_full_table(ternary_run):
    """At grid nodes the two tables hold the same equilibrium; between nodes they
    triangulate differently, so the couple differs by less than the grid allows."""
    table, faces, res = ternary_run
    db = table_db()
    phases = {name: SolutionPhase(db, name, ["A", "B", "C"]) for name in ("ALPHA", "BETA")}
    od = OnDemandEquilibriumTable(db, phases, T, spacing=0.025)
    nodes = table.points[::37]
    np.testing.assert_allclose(od.chemical_potentials(nodes), table.chemical_potentials(nodes), rtol=1e-9)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z[:, None] < 4e-4, [0.75, 0.15], [0.10, 0.75])
    lazy = solve_homogenized(faces, x0, od, 400.0, 2.0)
    assert np.max(np.abs(lazy.x[-1] - res.x[-1])) < 2e-3
    inv = lazy.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-12 * np.abs(inv[0]).max()
    assert od.calculations < len(table.points) / 4


@pytest.mark.slow
def test_quaternary_single_phase_couple_converges_to_the_phase1_solver():
    """Inside the alpha field the homogenized result must approach the ordinary solver as
    the composition grid is refined (2.1e-3 at a 0.02 step, 4.8e-4 at 0.01)."""
    db, phases = quaternary_db()
    faces = uniform_faces(4e-4, 100)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z[:, None] < 2e-4, [0.80, 0.06, 0.06], [0.62, 0.14, 0.10])
    ref = solve_planar(faces, x0, lambda x: phases["ALPHA"].interdiffusion_matrix(x, T), 200.0, 1.0, max_picard=300)
    errors = []
    for spacing in (0.02, 0.01):
        od = OnDemandEquilibriumTable(db, phases, T, spacing=spacing)
        assert np.all(od.phase_fractions(x0[[0, -1]])["ALPHA"] == pytest.approx(1.0))
        res = solve_homogenized(faces, x0, od, 200.0, 1.0, max_iter=300)
        errors.append(np.max(np.abs(res.x[-1] - ref.x[-1])))
    assert errors[1] < 1e-3 and errors[1] < errors[0] / 3


@pytest.mark.slow
def test_quaternary_two_phase_couple_conserves_with_few_equilibria():
    db, phases = quaternary_db()
    od = OnDemandEquilibriumTable(db, phases, T, spacing=0.025)
    faces = uniform_faces(8e-4, 120)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z[:, None] < 4e-4, [0.75, 0.10, 0.05], [0.10, 0.70, 0.10])
    res = solve_homogenized(faces, x0, od, 400.0, 2.0)
    inv = res.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-12 * np.abs(inv[0]).max()
    f = od.phase_fractions(res.x[-1])
    assert np.all(np.abs(f["ALPHA"] + f["BETA"] - 1) < 1e-9)
    assert f["ALPHA"].max() > 0.99 and f["BETA"].max() > 0.99  # both phases are in the couple
    assert od.calculations < 400  # a full table at this step has about 9100 nodes


def test_on_demand_refuses_compositions_beyond_its_grid():
    db = table_db()
    phases = {name: SolutionPhase(db, name, ["A", "B"]) for name in ("ALPHA", "BETA")}
    od = OnDemandEquilibriumTable(db, phases, T, spacing=0.05)
    assert not od.admissible(np.array([[0.02]]))
    with pytest.raises(ValueError, match="smaller spacing"):
        od.chemical_potentials(np.array([[0.02]]))


# ------------------------------------------------------------- ideal flux contribution (section 2.4, eq. 13)

def test_ideal_flux_speeds_an_ideal_solution_by_one_plus_epsilon():
    """With mu = RT ln x, eq. 13 adds epsilon times the whole flux, so a run with
    epsilon equals one without it run (1 + epsilon) times longer."""
    from conftest import IDEAL_TERNARY

    db = Database(IDEAL_TERNARY)
    phase = SolutionPhase(db, "SOL", ["A", "B"])
    table = LocalEquilibriumTable(db, {"SOL": phase}, T, spacing=0.0025)
    faces = uniform_faces(4e-4, 100)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 2e-4, 0.8, 0.2)[:, None]
    eps = 0.05
    from janus_kinetics import Boundary

    for left in (Boundary.no_flux(), Boundary.fixed([0.9])):  # the surface term scales the same way
        with_term = solve_homogenized(faces, x0, table, 100.0, 0.5, left=left, ideal_flux=eps)
        stretched = solve_homogenized(faces, x0, table, 100.0 * (1 + eps), 0.5 * (1 + eps), left=left)
        assert np.max(np.abs(with_term.x[-1] - stretched.x[-1])) < 1e-6


def test_ideal_flux_is_a_small_conservative_perturbation(ternary_run):
    """In a two-phase couple the change from the term grows in proportion to epsilon
    and every element stays conserved."""
    table, faces, base = ternary_run
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z[:, None] < 4e-4, [0.75, 0.15], [0.10, 0.75])
    shifts = []
    for eps in (0.01, 0.02):
        res = solve_homogenized(faces, x0, table, 400.0, 2.0, ideal_flux=eps)
        inv = res.inventory()
        assert np.max(np.abs(inv - inv[0])) < 1e-12 * np.abs(inv[0]).max()
        shifts.append(np.max(np.abs(res.x[-1] - base.x[-1])))
    assert 0 < shifts[0] < 0.02
    assert shifts[1] / shifts[0] == pytest.approx(2.0, rel=0.15)
    with pytest.raises(ValueError, match="non-negative"):
        solve_homogenized(faces, x0, table, 1.0, 1.0, ideal_flux=-0.01)


def test_ideal_flux_keeps_newton_convergence():
    """The term's slope enters the Newton matrix; with it, even a large epsilon leaves
    the iteration count unchanged, as a wrong slope would not."""
    from conftest import IDEAL_TERNARY

    db = Database(IDEAL_TERNARY)
    phase = SolutionPhase(db, "SOL", ["A", "B"])
    table = LocalEquilibriumTable(db, {"SOL": phase}, T, spacing=0.0025)
    faces = uniform_faces(4e-4, 100)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 2e-4, 0.8, 0.2)[:, None]
    plain = solve_homogenized(faces, x0, table, 100.0, 0.5)
    strong = solve_homogenized(faces, x0, table, 100.0, 0.5, ideal_flux=0.5)
    assert np.mean(strong.picard_iterations) <= 1.1 * np.mean(plain.picard_iterations)


# --- cylinders and spheres ------------------------------------------------------------------------------

@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_single_phase_core_matches_phase1_solver_in_curved_geometry(binary, geometry):
    """An alpha core richer than its shell, from the centre out: the homogenized run follows the phase-1
    solver in the same geometry (1.7e-5 apart) and keeps the r^p-weighted inventory to round-off."""
    alpha, _, table = binary
    faces = uniform_faces(4e-4, 200)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 2e-4, 0.95, 0.65)[:, None]
    homog = solve_homogenized(faces, x0, table, 200.0, 1.0, geometry=geometry)
    ref = solve_planar(faces, x0, lambda x: alpha.interdiffusion_matrix(x, T), 200.0, 1.0, geometry=geometry)
    assert np.max(np.abs(homog.x[-1] - ref.x[-1])) < 1e-4
    inv = homog.inventory()
    assert abs(inv[-1, 0] - inv[0, 0]) < 1e-12 * inv[0, 0]
    planar = solve_homogenized(faces, x0, table, 200.0, 1.0)
    assert np.max(np.abs(planar.x[-1] - homog.x[-1])) > 1e-3  # the geometry mattered



@pytest.mark.parametrize("geometry", ["planar", "spherical"])
def test_held_surface_matches_phase1_solver(binary, geometry):
    """Material at x = 0.95 whose surface is held at 0.70 loses B through it as in the phase-1 solver: the
    outer surface of a sphere, or a plate's face at z = 0, whose r^0 area must stay 1."""
    alpha, _, table = binary
    faces = uniform_faces(4e-4, 200)
    x0 = np.full((200, 1), 0.95)
    held = {"right": Boundary.fixed([0.70])} if geometry == "spherical" else {"left": Boundary.fixed([0.70])}
    homog = solve_homogenized(faces, x0, table, 200.0, 1.0, geometry=geometry, **held)
    ref = solve_planar(faces, x0, lambda x: alpha.interdiffusion_matrix(x, T), 200.0, 1.0, geometry=geometry,
                       **held)
    assert np.max(np.abs(homog.x[-1] - ref.x[-1])) < 1e-4
    assert homog.inventory()[-1, 0] < 0.99 * homog.inventory()[0, 0]  # material left through the surface

def alpha_core_error(binary, n_cells):
    """|radius of the homogenized alpha core - sharp-interface radius| over the interface shift, for an
    alpha sphere (x = 0.8, radius 0.4 mm) in a beta shell (x = 0.1) out to 0.8 mm."""
    alpha, beta, table = binary
    r_in, r_out, t1 = 4e-4, 8e-4, 200.0
    faces = uniform_faces(r_out, n_cells)
    z = 0.5 * (faces[1:] + faces[:-1])
    homog = solve_homogenized(faces, np.where(z < r_in, 0.8, 0.1)[:, None], table, t1, t1 / 200, geometry="spherical")
    volume = np.sum(table.phase_fractions(homog.x[-1])["ALPHA"] * homog.widths)  # per steradian, r^3 / 3
    radius = (3 * volume) ** (1 / 3)
    sharp = solve_layers([0.0, r_in, r_out],
                         [Layer(np.full(n_cells // 2, 0.8), lambda x: alpha.interdiffusion_matrix(x, T)),
                          Layer(np.full(n_cells // 2, 0.1), lambda x: beta.interdiffusion_matrix(x, T))],
                         [LocalEquilibrium(alpha, beta, T, [0.56], [0.27])], t1, t1 / 200, geometry="spherical")
    shift = sharp.interfaces[-1, 0] - r_in
    assert shift > 5e-6
    return abs(radius - sharp.interfaces[-1, 0]) / shift


@pytest.mark.slow
def test_two_phase_sphere_converges_to_sharp_interface(binary):
    """As the planar couple does: 9.7 % of the shift at 400 cells, 3.6 % at 800."""
    coarse, fine = alpha_core_error(binary, 400), alpha_core_error(binary, 800)
    assert coarse < 0.12
    assert fine < 0.6 * coarse


def test_curved_geometry_inputs_are_checked(binary):
    _, _, table = binary
    faces = uniform_faces(4e-4, 20)
    x0 = np.full((20, 1), 0.9)
    with pytest.raises(ValueError, match="r = 0"):
        solve_homogenized(faces, x0, table, 1.0, 1.0, left=Boundary.fixed([0.9]), geometry="spherical")
    with pytest.raises(ValueError, match="geometry"):
        solve_homogenized(faces, x0, table, 1.0, 1.0, geometry="conical")
    with pytest.raises(ValueError, match="negative"):
        solve_homogenized(faces - 1e-4, x0, table, 1.0, 1.0, geometry="cylindrical")


# --- adaptive steps ---------------------------------------------------------------------------------------

COUPLE_SAVES = [0.1, 1.0, 10.0]


@pytest.fixture(scope="module")
def two_phase_couple(binary):
    """The alpha/beta couple of alpha_layer_error on 200 cells, saved at 0.1 s to 200 s, with a 0.05 s-step run as
    reference (within about 2e-4 of a 0.01 s-step one)."""
    _, _, table = binary
    faces = uniform_faces(8e-4, 200)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 4e-4, 0.8, 0.1)[:, None]
    return table, faces, x0, solve_homogenized(faces, x0, table, 200.0, 0.05, save_times=COUPLE_SAVES)


def test_adaptive_homogenization_follows_a_couple_over_decades(two_phase_couple):
    """Tolerance 1e-4 takes about 217 steps for a worst error of 9e-4 over the saved times; 200 equal steps
    leave 4e-3 (800 are needed for 1e-3)."""
    table, faces, x0, ref = two_phase_couple
    adaptive = solve_homogenized(faces, x0, table, 200.0, 50.0, save_times=COUPLE_SAVES, tolerance=1e-4)
    fixed = solve_homogenized(faces, x0, table, 200.0, 1.0, save_times=COUPLE_SAVES)
    np.testing.assert_array_equal(adaptive.times, ref.times)
    err = np.abs(adaptive.x - ref.x).max()
    assert err < 1.5e-3
    assert len(adaptive.picard_iterations) < 300
    assert np.abs(fixed.x - ref.x).max() > 3 * err
    inv = adaptive.inventory()
    assert np.abs(inv - inv[0]).max() < 1e-12 * inv[0, 0]


def test_adaptive_homogenization_recovers_from_a_failed_step(two_phase_couple):
    """With eight iterations allowed a single 200 s step does not converge; adaptive steps halve until they do."""
    table, faces, x0, ref = two_phase_couple
    with pytest.raises(ConvergenceError):
        solve_homogenized(faces, x0, table, 200.0, 200.0, max_iter=8)
    res = solve_homogenized(faces, x0, table, 200.0, 200.0, max_iter=8, tolerance=1e-4)
    assert np.abs(res.x[-1] - ref.x[-1]).max() < 1.5e-3
