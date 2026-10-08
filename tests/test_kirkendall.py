"""Kirkendall marker motion against van Loo, Prog. Solid State Chem. 20 (1990) 47, pp. 52-53.

Eq. (4) relates the Matano and Kirkendall frames, eq. (5) gives the parabolic
marker shift, eqs. (9)-(10) give the marker velocity and Darken's interdiffusion
coefficient from the intrinsic ones, and eq. (11) recovers an intrinsic
diffusivity from a measured profile and marker position. Eq. (11) is applied to
the simulated couple exactly as an experimentalist would apply it to a measured one.
"""

import numpy as np
import pytest
from pycalphad import Database

from janus_kinetics import R, SolutionPhase, solve_planar, uniform_faces
from janus_kinetics.diffusivity import lattice_velocity
from conftest import HEADER

T = 1200.0
D_A, D_B = 4e-15, 1e-15  # intrinsic = tracer diffusivities in the ideal couple below, m^2/s


def ideal_binary(d_a, d_b):
    """Ideal A-B solution with composition-independent mobilities.

    MQ = RT ln(d) makes D* = RT M = exp(MQ/RT) = d at every T, and in an ideal
    solution the intrinsic diffusivity equals D*.
    """
    body = f"""
PHASE SOL % 1 1 !
CONSTITUENT SOL :A,B: !
PARAMETER MQ(SOL&A,A;0) 298.15 R*T*LN({d_a:.12e}); 6000 N !
PARAMETER MQ(SOL&A,B;0) 298.15 R*T*LN({d_a:.12e}); 6000 N !
PARAMETER MQ(SOL&B,A;0) 298.15 R*T*LN({d_b:.12e}); 6000 N !
PARAMETER MQ(SOL&B,B;0) 298.15 R*T*LN({d_b:.12e}); 6000 N !
"""
    return SolutionPhase(Database(HEADER + body), "SOL", ["A", "B"])


def test_ideal_binary_mobilities_give_the_requested_diffusivities():
    phase = ideal_binary(D_A, D_B)
    x = np.array([[0.1], [0.5], [0.9]])
    intr = phase.intrinsic_matrix(x, T)
    np.testing.assert_allclose(intr[:, 0, 0], D_A, rtol=1e-12)
    np.testing.assert_allclose(-intr[:, 1, 0], D_B, rtol=1e-12)


def test_intrinsic_and_interdiffusion_matrices_are_frame_related(regular_db):
    """J_k = J'_k + c_k v with sum_k J_k = 0 (eq. 4) means D~_kj = D^I_kj - x_k sum_i D^I_ij."""
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    x = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.05, 0.9, 0.05]])
    intr = phase.intrinsic_matrix(x, T)
    expected = intr[:, :2, :] - x[:, :2, None] * intr.sum(axis=1)[:, None, :]
    np.testing.assert_allclose(phase.interdiffusion_matrix(x, T), expected, rtol=1e-12)


def test_binary_velocity_and_darken_relation_in_a_nonideal_solution(regular_db):
    """Eqs. (9) and (10) at equal partial volumes: v = (D_A - D_B) dx_A/dz, D~ = x_A D_B + x_B D_A."""
    phase = SolutionPhase(regular_db, "SOL", ["A", "B"])
    x = np.array([[0.15], [0.4], [0.7]])
    intr = phase.intrinsic_matrix(x, T)
    d_a, d_b = intr[:, 0, 0], -intr[:, 1, 0]
    grad = np.array([[-3.0e4], [1.2e5], [-7.0e3]])
    np.testing.assert_allclose(phase.lattice_velocity(x, grad, T), (d_a - d_b) * grad[:, 0], rtol=1e-12)
    xa = x[:, 0]
    np.testing.assert_allclose(phase.interdiffusion_matrix(x, T)[:, 0, 0], xa * d_b + (1 - xa) * d_a, rtol=1e-12)


def test_equal_mobilities_leave_markers_in_place(regular_db):
    """With M_i equal for every element sum_k J'_k vanishes by Gibbs-Duhem, so the lattice does not move."""
    x = np.array([[0.2, 0.3, 0.5], [0.45, 0.45, 0.1]])
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    dmu = phase.dmu_dx(x, T)
    m = np.full_like(x, 3e-19)
    intr = (x * m)[..., :, None] * dmu
    assert np.max(np.abs(lattice_velocity(intr, np.array([[1e4, -2e4], [5e3, 8e3]])))) < 1e-25


@pytest.fixture(scope="module")
def couple():
    """A-rich (x_A = 0.9) left of a weld at z = 0, B-rich (0.1) right, markers on the weld."""
    phase = ideal_binary(D_A, D_B)
    half, n = 4e-4, 800  # half >> 4 sqrt(D_A t_end) = 8e-5 m, so the ends stay untouched
    faces = uniform_faces(2 * half, n, origin=-half)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 0, 0.9, 0.1)[:, None]
    t_end = 4e5
    res = solve_planar(faces, x0, lambda x: phase.interdiffusion_matrix(x, T), t_end, t_end / 2000,
                       save_times=[t_end / 4], markers=[0.0],
                       lattice_velocity=lambda x, g: phase.lattice_velocity(x, g, T))
    return res, z, t_end


def test_markers_move_toward_the_fast_diffuser_side(couple):
    res, _, _ = couple
    assert res.markers[-1, 0] < 0  # A diffuses faster, so the lattice shifts into the A-rich side


def test_marker_shift_is_parabolic(couple):
    """By eq. (5) the Kirkendall plane moves as sqrt(t), so quadrupling t doubles the shift."""
    res, _, _ = couple
    assert res.markers[2, 0] / res.markers[1, 0] == pytest.approx(2.0, rel=2e-3)


def test_marker_velocity_matches_eq9_at_the_marker(couple):
    """v = dx/2t from eq. (5) against (D_A - D_B) dN_A/dz of the computed profile, eq. (9)."""
    res, z, t_end = couple
    zk = res.markers[-1, 0]
    slope = np.interp(zk, 0.5 * (z[1:] + z[:-1]), np.diff(res.x[-1, :, 0]) / np.diff(z))
    assert zk / (2 * t_end) == pytest.approx((D_A - D_B) * slope, rel=5e-3)


def van_loo_eq11(z, c, zk, t):
    """D_i(x_K) from a profile c(z) and the Kirkendall plane zk, van Loo (1990) eq. (11)."""
    c_minus, c_plus = c[0], c[-1]
    left, right = z < zk, z >= zk
    ck = np.interp(zk, z, c)
    zl, cl = np.append(z[left], zk), np.append(c[left], ck)
    zr, cr = np.insert(z[right], 0, zk), np.insert(c[right], 0, ck)
    i_left = np.trapezoid(cl - c_minus, zl)
    i_right = np.trapezoid(c_plus - cr, zr)
    dc_dz = np.interp(zk, 0.5 * (z[1:] + z[:-1]), np.diff(c) / np.diff(z))
    return (1 / (2 * t)) / dc_dz / (c_plus - c_minus) * (c_plus * i_left - c_minus * i_right)


def test_van_loo_eq11_recovers_both_intrinsic_diffusivities(couple):
    res, z, t_end = couple
    zk = res.markers[-1, 0]
    xa = res.x[-1, :, 0]
    assert van_loo_eq11(z, xa, zk, t_end) == pytest.approx(D_A, rel=1e-2)
    assert van_loo_eq11(z, 1 - xa, zk, t_end) == pytest.approx(D_B, rel=1e-2)


def test_tracking_requires_a_velocity():
    with pytest.raises(ValueError, match="lattice_velocity"):
        solve_planar(uniform_faces(1.0, 4), np.zeros((4, 1)), 1.0, 1.0, 0.5, markers=[0.5])


def test_eq_12_reduces_to_eq_11_at_constant_molar_volume(couple):
    from janus_kinetics.couple_analysis import intrinsic_from_profile

    res, z, t_end = couple
    zk = res.markers[-1, 0]
    xa = res.x[-1, :, 0]
    for profile in (xa, 1 - xa):
        assert intrinsic_from_profile(z, profile, t_end, zk) == pytest.approx(van_loo_eq11(z, profile, zk, t_end), rel=1e-9)
