"""Diffusion with unequal partial molar volumes, against van Loo, Prog. Solid State Chem. 20 (1990) 47, pp. 51-53.

The solver runs in concentrations with the volume-fixed interdiffusion matrix.
Eqs. 9 and 10 are checked against intrinsic diffusivities D_i = c_i M_i dmu_i/dc_i
computed here by finite differences, and eqs. 3 and 12 are applied to a simulated
couple as an experimentalist would apply them to a measured one.
"""

import numpy as np
import pytest

from calphad_kinetics import SolutionPhase, solve_planar, uniform_faces
from calphad_kinetics.couple_analysis import interdiffusion_from_profile, intrinsic_from_profile
from calphad_kinetics.diffusivity import to_concentrations, to_mole_fractions
from test_kirkendall import ideal_binary

T = 1200.0
VOL = np.array([1.0e-5, 1.8e-5])  # partial molar volumes of A and B, m^3/mol


def intrinsic_by_differences(phase, x_a, vol=VOL, h=1e-4):
    """(D_A, D_B) with J'_i = -D_i dc_i/dz, from mu along sum V_i c_i = 1 (dc_B = -V_A/V_B dc_A)."""
    c_a = to_concentrations(np.array([x_a, 1 - x_a]), vol)[0]
    dc = h * c_a
    mu = [phase.chemical_potentials(to_mole_fractions(np.array([c_a + s * dc]), vol), T) for s in (1, -1)]
    dmu_a = (mu[0][0] - mu[1][0]) / (2 * dc)
    dmu_b = (mu[0][1] - mu[1][1]) / (2 * dc) / (-vol[0] / vol[1])
    x = np.array([x_a, 1 - x_a])
    c = x / (x @ vol)
    m = phase.mobility(x, T)
    return c[0] * m[0] * dmu_a, c[1] * m[1] * dmu_b


def test_concentration_round_trip():
    x = np.array([[0.2, 0.8], [0.73, 0.27]])
    np.testing.assert_allclose(to_mole_fractions(to_concentrations(x, VOL), VOL), x, rtol=1e-14)


def test_eqs_9_and_10_with_unequal_partial_volumes(regular_db):
    """v = V_A (D_A - D_B) dC_A/dz (eq. 9) and D~ = C_A V_A D_B + C_B V_B D_A (eq. 10)."""
    phase = SolutionPhase(regular_db, "SOL", ["A", "B"])
    for x_a in (0.15, 0.4, 0.8):
        d_a, d_b = intrinsic_by_differences(phase, x_a)
        x = np.array([x_a, 1 - x_a])
        c = x / (x @ VOL)
        d_v = phase.interdiffusion_matrix(x, T, frame="volume", partial_volumes=VOL)[0, 0]
        assert d_v == pytest.approx(c[0] * VOL[0] * d_b + c[1] * VOL[1] * d_a, rel=1e-6)
        grad = np.array([2.5e9])
        v = phase.lattice_velocity(x, grad, T, partial_volumes=VOL)
        assert v == pytest.approx(VOL[0] * (d_a - d_b) * grad[0], rel=1e-6)


@pytest.fixture(scope="module")
def couple():
    phase = ideal_binary(4e-15, 1e-15)
    half, n = 4e-4, 800
    faces = uniform_faces(2 * half, n, origin=-half)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 0, 0.9, 0.1)
    c0 = to_concentrations(np.column_stack([x0, 1 - x0]), VOL)
    t_end = 4e5
    # backward Euler is first order in time; this step holds the marker ratio to 0.3 %
    res = solve_planar(faces, c0, phase.interdiffusion_function(T, "volume", VOL), t_end, t_end / 6000,
                       save_times=[t_end / 4], markers=[0.0], lattice_velocity=phase.lattice_velocity_function(T, VOL))
    x_a = to_mole_fractions(res.x[-1], VOL)[:, 0]
    return phase, res, z, x_a, t_end


def test_moles_are_conserved_while_the_molar_volume_varies(couple):
    _, res, _, x_a, _ = couple
    inv = res.inventory()[:, 0]
    assert np.max(np.abs(inv - inv[0])) < 1e-11 * inv[0]
    vm = np.column_stack([x_a, 1 - x_a]) @ VOL
    assert vm.max() / vm.min() > 1.5  # the couple really spans different molar volumes


def test_eq_3_recovers_the_interdiffusion_coefficient(couple):
    phase, _, z, x_a, t = couple
    vm = np.column_stack([x_a, 1 - x_a]) @ VOL
    d_a = interdiffusion_from_profile(z, x_a, t, vm)
    d_b = interdiffusion_from_profile(z, 1 - x_a, t, vm)
    for target in (0.25, 0.5, 0.75):
        i = np.argmin(np.abs(x_a - target))
        expected = phase.interdiffusion_matrix(np.array([x_a[i], 1 - x_a[i]]), T, "volume", VOL)[0, 0]
        assert d_a[i] == pytest.approx(expected, rel=5e-3)
        assert d_b[i] == pytest.approx(expected, rel=5e-3)  # the same D~ from either element


def test_eq_12_recovers_both_intrinsic_diffusivities_at_the_markers(couple):
    phase, res, z, x_a, t = couple
    zk = res.markers[-1, 0]
    vm = np.column_stack([x_a, 1 - x_a]) @ VOL
    d_a, d_b = intrinsic_by_differences(phase, float(np.interp(zk, z, x_a)))
    assert intrinsic_from_profile(z, x_a, t, zk, vm) == pytest.approx(d_a, rel=1e-2)
    assert intrinsic_from_profile(z, 1 - x_a, t, zk, vm) == pytest.approx(d_b, rel=1e-2)
    # Ignoring the volume change (eq. 11 with one V_m) gets D_A wrong by far more
    assert abs(intrinsic_from_profile(z, x_a, t, zk) / d_a - 1) > 0.1


def test_markers_still_move_parabolically(couple):
    _, res, _, _, _ = couple
    assert res.markers[2, 0] / res.markers[1, 0] == pytest.approx(2.0, rel=3e-3)


def test_analysis_input_checks():
    z = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="equal"):
        interdiffusion_from_profile(z, np.full(5, 0.3), 1.0)
    with pytest.raises(ValueError, match="inside"):
        intrinsic_from_profile(z, np.linspace(0.9, 0.1, 5), 1.0, 2.0)
    with pytest.raises(ValueError, match="increasing"):
        interdiffusion_from_profile(z[::-1], np.linspace(0.9, 0.1, 5), 1.0)
