"""Non-isothermal anneals along a temperature schedule.

When D does not depend on composition, the diffusion equation dx/dt = D(t) d2x/dz2
becomes the isothermal one in the variable tau = integral of D dt, so every
constant-D similarity solution holds with D t replaced by tau. That gives exact
references for a ramp-hold-cool anneal.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf, erfc

from calphad_kinetics import FixedInterface, SolutionPhase, TemperatureSchedule, solve_moving_boundary, solve_planar, uniform_faces
from calphad_kinetics.schedule import with_time

R = 8.314462618
SCHEDULE = TemperatureSchedule([0.0, 600.0, 3000.0, 3600.0], [700.0, 1200.0, 1200.0, 800.0])
D0, Q = 1e-4, 250e3  # Arrhenius pre-factor (m^2/s) and activation energy (J/mol), invented


def d_of_t(t):
    return D0 * np.exp(-Q / (R * SCHEDULE(t)))


def tau(t_end):
    return quad(d_of_t, 0.0, t_end, points=SCHEDULE.times[1:-1], limit=200)[0]


def test_schedule_interpolates_and_holds():
    assert SCHEDULE(300.0) == pytest.approx(950.0)
    assert SCHEDULE(-5.0) == 700.0 and SCHEDULE(1e5) == 800.0
    assert TemperatureSchedule.isothermal(1000.0)(123.0) == 1000.0
    with pytest.raises(ValueError):
        TemperatureSchedule([0.0, 0.0], [300.0, 400.0])


def test_callbacks_with_and_without_time():
    assert with_time(lambda x: x + 1, 1)(1, 99.0) == 2
    assert with_time(lambda x, t: x + t, 1)(1, 99.0) == 100.0


def test_step_couple_under_an_anneal_is_erf_in_tau():
    faces = uniform_faces(3e-5, 600, origin=-1.5e-5)  # sqrt(tau) = 1.8 um, so 8 diffusion lengths each side
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 0, 0.8, 0.2)[:, None]
    res = solve_planar(faces, x0, lambda x, t: np.broadcast_to(d_of_t(t), x.shape[:-1] + (1, 1)), 3600.0, 1.0)
    exact = 0.5 - 0.3 * erf(z / (2 * np.sqrt(tau(3600.0))))
    assert np.max(np.abs(res.x[-1, :, 0] - exact)) < 1e-4


def test_zener_growth_under_an_anneal_follows_tau():
    """Zener eqs. (10)-(12) with D t replaced by tau: S = alpha sqrt(tau).

    Started inside the hold (t0 = 900 s), where the exact starting profile spans
    many cells, and run through the cooling ramp.
    """
    from test_moving_boundary import N0, N1, NINF, phi1, zener_alpha

    alpha = zener_alpha(N0, N1, NINF)
    t0 = 900.0
    s0 = alpha * np.sqrt(tau(t0))
    length = 40 * np.sqrt(tau(3600.0))
    zr = s0 + (length - s0) * (np.arange(800) + 0.5) / 800
    x_right = NINF + (N1 - NINF) * phi1(zr / np.sqrt(tau(t0))) / phi1(alpha)
    d_right = lambda x, t: np.broadcast_to(d_of_t(t0 + t), x.shape[:-1] + (1, 1))  # solver time starts at t0
    res = solve_moving_boundary(length, s0, np.full(20, N0), x_right, 0.0, d_right, FixedInterface([N0], [N1]),
                                3600.0 - t0, 4.0)
    assert res.interface[-1] == pytest.approx(alpha * np.sqrt(tau(3600.0)), rel=3e-3)


def test_phase_helpers_follow_the_schedule(regular_db):
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    x = np.array([[0.2, 0.3]])
    f = phase.interdiffusion_function(SCHEDULE)
    np.testing.assert_allclose(f(x, 1000.0), phase.interdiffusion_matrix(x, 1200.0), rtol=1e-14)
    np.testing.assert_allclose(f(x, 300.0), phase.interdiffusion_matrix(x, 950.0), rtol=1e-14)
    g = np.array([[1e3, -2e3]])
    np.testing.assert_allclose(phase.lattice_velocity_function(SCHEDULE)(x, g, 0.0),
                               phase.lattice_velocity(x, g, 700.0), rtol=1e-14)
