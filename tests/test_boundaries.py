"""Surfaces held at a chemical potential, and mixed held / zero-flux boundaries."""

import numpy as np
import pytest
from pycalphad import Database
from scipy.special import erfc

from janus_kinetics import Boundary, R, TemperatureSchedule, solve_planar, uniform_faces
from janus_kinetics.interstitial import InterstitialPhase
from conftest import HEADER
from test_interstitial import TERNARY

T = 1200.0

# Ideal interstitial C in an immobile host, with a C mobility that does not depend on
# composition, so D_CC = RT Omega_C [1 + negligible frame term] is constant.
CARBURIZE = HEADER + """
PHASE BCC % 2 1 3 !
CONSTITUENT BCC :A:C,VA: !
PARAMETER G(BCC,A:VA;0) 298.15 0; 6000 N !
PARAMETER G(BCC,A:C;0) 298.15 30000; 6000 N !
PARAMETER MQ(BCC&C,A:*;0) 298.15 -90000+R*T*LN(2E-6); 6000 N !
PARAMETER MQ(BCC&A,A:*;0) 298.15 -400000; 6000 N !
"""


def mu_c(y, temp):
    """Ideal interstitial: mu_C = (G_A:C - G_A:Va)/3 + RT ln(y / (1 - y))."""
    return 30000 / 3 + R * temp * np.log(y / (1 - y))


@pytest.fixture(scope="module")
def carbon():
    return InterstitialPhase(Database(CARBURIZE), "BCC", ["A"], ["C"])


def test_potential_boundary_reaches_the_requested_potential(carbon):
    b = Boundary.potential(carbon, {"C": mu_c(0.02, T)}, T)
    u = b.face(0.0, np.array([0.003]), 1)[0]
    assert u[0] == pytest.approx(3 * 0.02, rel=1e-10)


def test_potential_boundary_follows_a_schedule(carbon):
    schedule = TemperatureSchedule([0.0, 100.0], [1000.0, 1300.0])
    target = mu_c(0.02, 1000.0)
    b = Boundary.potential(carbon, {"C": target}, schedule)
    for t in (0.0, 50.0, 100.0):
        u = b.face(t, np.array([0.003]), 1)[0]
        assert carbon.chemical_potentials(u, schedule(t))[1] == pytest.approx(target, abs=1e-6)


def test_carburizing_at_fixed_potential_is_erfc(carbon):
    """Constant D: u = u0 + (u_s - u0) erfc(z / 2 sqrt(D t)) with u_s set by the potential."""
    u0, u_s = 0.003, 0.06
    d = carbon.interdiffusion_matrix(np.array([u_s]), T)[0, 0]
    assert carbon.interdiffusion_matrix(np.array([u0]), T)[0, 0] == pytest.approx(d, rel=1e-6)
    t_end = 3600.0
    faces = uniform_faces(10 * np.sqrt(d * t_end), 400)
    res = solve_planar(faces, np.full((400, 1), u0), carbon.interdiffusion_function(T), t_end, t_end / 400,
                       left=Boundary.potential(carbon, {"C": mu_c(0.02, T)}, T))
    exact = u0 + (u_s - u0) * erfc(res.z / (2 * np.sqrt(d * t_end)))
    assert np.max(np.abs(res.x[-1, :, 0] - exact)) < 5e-4 * u_s


def test_held_carbon_with_zero_flux_substitutional():
    """C enters at a held potential while B may not cross the surface: B redistributes
    through the coupling but its inventory stays fixed."""
    phase = InterstitialPhase(Database(TERNARY), "BCC", ["A", "B"], ["C"], dependent="A")  # independent B, C
    target = float(phase.chemical_potentials(np.array([0.3, 0.30]), T)[2])
    faces = uniform_faces(2e-3, 200)
    x0 = np.tile([0.3, 0.05], (200, 1))
    res = solve_planar(faces, x0, phase.interdiffusion_function(T), 3600.0, 36.0,
                       left=Boundary.potential(phase, {"C": target}, T))
    inv = res.inventory()
    assert abs(inv[-1, 0] - inv[0, 0]) < 1e-12 * inv[0, 0]  # B conserved
    assert inv[-1, 1] > 1.5 * inv[0, 1]  # C taken up
    assert np.ptp(res.x[-1, :, 0]) > 1e-4  # B moved inside, so its conservation is not trivial
    surface = res.x[-1, 0]
    assert phase.chemical_potentials(surface, T)[2] == pytest.approx(target, abs=0.05 * R * T)


def test_held_subset_of_a_constant_composition():
    """Holding only component 1 of a constant composition leaves component 0 with zero flux."""
    d = np.array([[1e-14, 2e-15], [3e-15, 5e-15]])
    faces = uniform_faces(1e-4, 100)
    res = solve_planar(faces, np.tile([0.2, 0.1], (100, 1)), d, 1e5, 1e3,
                       left=Boundary.fixed([0.2, 0.4], components=[1]))
    inv = res.inventory()
    assert abs(inv[-1, 0] - inv[0, 0]) < 1e-12 * inv[0, 0]
    assert inv[-1, 1] > inv[0, 1]
