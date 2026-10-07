"""Planar implicit solver against closed-form diffusion solutions.

The references are the similarity (error-function) solutions of the diffusion
equation with constant D: a step couple in an infinite medium, and a
semi-infinite medium whose surface is held at fixed composition.
"""

import numpy as np
import pytest
from scipy.special import erf

from calphad_kinetics import (
    Boundary,
    ConvergenceError,
    SolutionPhase,
    UnstableCompositionError,
    solve_planar,
    uniform_faces,
)

D = 1e-14  # m^2/s
T_END = 1e5  # s, so sqrt(D t) = 31.6 um
HALF = 5e-4  # half-length (m), about 8 x 2 sqrt(D t), so the far field is untouched


def step_couple(n_cells, left, right, half=HALF):
    faces = uniform_faces(2 * half, n_cells, origin=-half)
    centers = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(centers[:, None] < 0, left, right)
    return faces, centers, x0


def erf_couple(z, t, d, left, right):
    return 0.5 * (left + right) - 0.5 * (left - right) * erf(z / (2 * np.sqrt(d * t)))


def couple_error(n_cells, n_steps):
    faces, z, x0 = step_couple(n_cells, np.array([0.8]), np.array([0.2]))
    res = solve_planar(faces, x0, D, T_END, T_END / n_steps)
    exact = erf_couple(z, T_END, D, 0.8, 0.2)
    return np.max(np.abs(res.x[-1, :, 0] - exact))


def test_binary_constant_d_matches_erf():
    err = couple_error(400, 400)
    assert err < 5e-4, err


def test_binary_error_falls_with_refinement():
    """Backward Euler is first order in time; halving dt and dz should roughly halve the error."""
    coarse, fine = couple_error(200, 100), couple_error(400, 200)
    assert fine < 0.6 * coarse, (coarse, fine)


def test_fixed_surface_composition_matches_erf():
    """Semi-infinite medium with the surface held at xs, x = xs + (x0 - xs) erf(z / 2 sqrt(Dt))."""
    xs, xi = 0.05, 0.6
    faces = uniform_faces(HALF, 400)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.full((400, 1), xi)
    res = solve_planar(faces, x0, D, T_END, T_END / 400, left=Boundary.fixed([xs]))
    exact = xs + (xi - xs) * erf(z / (2 * np.sqrt(D * T_END)))
    assert np.max(np.abs(res.x[-1, :, 0] - exact)) < 5e-4


def test_diagonal_ternary_matches_independent_erfs():
    d = np.diag([1e-14, 4e-15])
    left, right = np.array([0.7, 0.1]), np.array([0.2, 0.5])
    faces, z, x0 = step_couple(400, left, right)
    res = solve_planar(faces, x0, d, T_END, T_END / 400)
    for k in range(2):
        exact = erf_couple(z, T_END, d[k, k], left[k], right[k])
        assert np.max(np.abs(res.x[-1, :, k] - exact)) < 5e-4


def test_coupled_constant_ternary_matches_eigen_solution():
    """Off-diagonal constant D: with D = V diag(lambda) V^-1 the combinations V^-1 x obey independent erf solutions."""
    d = np.array([[1.0e-14, 2.0e-15], [5.0e-15, 4.0e-15]])
    lam, vec = np.linalg.eig(d)
    assert np.all(lam.real > 0) and np.allclose(lam.imag, 0)
    lam, vec = lam.real, vec.real
    left, right = np.array([0.6, 0.15]), np.array([0.3, 0.35])
    faces, z, x0 = step_couple(400, left, right)
    res = solve_planar(faces, x0, d, T_END, T_END / 400)
    ul, ur = np.linalg.solve(vec, left), np.linalg.solve(vec, right)
    u = np.column_stack([erf_couple(z, T_END, lam[k], ul[k], ur[k]) for k in range(2)])
    exact = u @ vec.T
    assert np.max(np.abs(res.x[-1] - exact)) < 5e-4
    # the uphill feature that makes this case a real coupling test
    assert np.ptp(exact[:, 1]) > abs(left[1] - right[1])


def round_off_bound(d_max, dt, widths, n_steps):
    """Inventory drift expected from floating point alone.

    Each implicit step sums stiffness terms that are larger than the mass terms by
    the cell Fourier number Fo = D dt / dz^2, so double-precision round-off
    (about 1e-16) is amplified by Fo per step. In exact arithmetic the scheme
    conserves the inventory, because its stiffness matrix has zero column sums.
    """
    fourier = d_max * dt / widths.min() ** 2
    return 10 * np.finfo(float).eps * fourier * n_steps


def no_flux_case(regular_db):
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    faces = uniform_faces(2e-4, 120)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.column_stack([0.3 + 0.2 * np.tanh((z - 1e-4) / 1e-5), 0.3 - 0.15 * np.sin(z / 3e-5)])
    dfun = lambda x: phase.interdiffusion_matrix(x, 1300.0)
    return faces, x0, dfun


def test_no_flux_conserves_mass_with_composition_dependent_d(regular_db):
    faces, x0, dfun = no_flux_case(regular_db)
    dt, t_end = 2e3, 2e5
    res = solve_planar(faces, x0, dfun, t_end, dt, save_times=[5e4, 1e5])
    inv = res.inventory()
    drift = np.abs(inv - inv[0]) / inv[0]
    d_max = np.abs(dfun(x0)).max()
    bound = round_off_bound(d_max, dt, res.widths, round(t_end / dt))
    print(f"relative inventory drift {drift.max():.3e}, round-off bound {bound:.3e}")
    assert drift.max() < bound
    assert drift.max() < 1e-8
    assert max(res.picard_iterations) > 1  # D really changed within the step
    assert np.ptp(res.x[-1, :, 0]) < 0.5 * np.ptp(x0[:, 0])  # and diffusion happened


def test_conservation_error_is_round_off_not_scheme(regular_db):
    """The drift falls with the Fourier number; a scheme leak would not."""
    faces, x0, dfun = no_flux_case(regular_db)
    drifts = []
    for dt in (2e3, 2e1):
        res = solve_planar(faces, x0, dfun, 10 * dt, dt)
        inv = res.inventory()
        drifts.append(np.abs(inv[-1] - inv[0]).max() / inv[0].max())
    print(f"drift after 10 steps at dt=2e3: {drifts[0]:.3e}, at dt=2e1: {drifts[1]:.3e}")
    assert drifts[1] < drifts[0] / 30
    assert drifts[1] < 1e-12


def test_nonlinear_profile_obeys_boltzmann_scaling():
    """For a step couple in an infinite medium x depends on z / sqrt(t) only, whatever D(x)."""
    dfun = lambda x: (1e-15 * np.exp(4.0 * x))[..., None]
    half = 1.5e-3  # > 10 sqrt(D_max t_end), so the couple stays effectively infinite
    faces, z, x0 = step_couple(1500, np.array([0.9]), np.array([0.1]), half)
    res = solve_planar(faces, x0, dfun, 4e5, 250.0, save_times=[1e5])
    early = res.x[1, :, 0]
    late = res.x[2, :, 0]
    late_at_scaled = np.interp(2 * z, z, late)  # x(2z, 4t) should equal x(z, t)
    inner = np.abs(z) < half / 2
    assert np.max(np.abs(late_at_scaled[inner] - early[inner])) < 1.5e-3
    # and the profile is genuinely asymmetric, so D(x) mattered
    exact_const = erf_couple(z, 1e5, 1e-15 * np.exp(2.0), 0.9, 0.1)
    assert np.max(np.abs(early - exact_const)) > 0.02


def test_profile_entering_spinodal_raises(gap_db):
    phase = SolutionPhase(gap_db, "SOL", ["A", "B"])
    faces, z, x0 = step_couple(40, np.array([0.95]), np.array([0.05]))
    with pytest.raises(UnstableCompositionError, match="multiphase model"):
        solve_planar(faces, x0, lambda x: phase.interdiffusion_matrix(x, 1000.0), 1e3, 10.0)


def test_picard_failure_is_reported():
    dfun = lambda x: (1e-15 * np.exp(8.0 * x))[..., None]
    faces, z, x0 = step_couple(50, np.array([0.9]), np.array([0.1]))
    with pytest.raises(ConvergenceError):
        solve_planar(faces, x0, dfun, 1e5, 1e5, max_picard=2)


def test_graded_grid_matches_erf():
    """Cells refined toward the fixed surface; the face-flux form must stay accurate."""
    xs, xi = 0.9, 0.3
    faces = HALF * np.linspace(0.0, 1.0, 301) ** 1.5
    z = 0.5 * (faces[1:] + faces[:-1])
    res = solve_planar(faces, np.full((300, 1), xi), D, T_END, T_END / 400, left=Boundary.fixed([xs]))
    exact = xs + (xi - xs) * erf(z / (2 * np.sqrt(D * T_END)))
    assert np.max(np.abs(res.x[-1, :, 0] - exact)) < 5e-4
