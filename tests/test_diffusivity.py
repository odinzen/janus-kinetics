"""Interdiffusion matrices in the number-fixed and volume-fixed frames."""

import numpy as np
import pytest

from calphad_kinetics import R, SolutionPhase, UnstableCompositionError
from conftest import REGULAR_L, REGULAR_LABC

T = 1100.0
X = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.05, 0.9, 0.05], [0.4, 0.4, 0.2]])


def test_ideal_reduces_to_darken_form(ideal_db):
    """For an ideal solution D_kj = delta_kj D*_k - x_k (D*_j - D*_n), with D* = RT M.

    Follows from eq. (2) of Jonsson, ISIJ Int. 35 (1995) 1415 with mu_i = mu_i0 + RT ln x_i.
    """
    phase = SolutionPhase(ideal_db, "SOL", ["A", "B", "C"])
    d = phase.interdiffusion_matrix(X, T)
    dstar = R * T * phase.mobility(X, T)
    expected = (np.eye(2)[None] * dstar[:, :2, None]
                - X[:, :2, None] * (dstar[:, None, :2] - dstar[:, None, 2:]))
    np.testing.assert_allclose(d, expected, rtol=1e-11)


def test_ideal_binary_is_darken(ideal_db):
    """For an ideal binary D~ = x_B D*_A + x_A D*_B (van Loo, Prog. Solid State Chem. 20 (1990) 47, eq. 10)."""
    phase = SolutionPhase(ideal_db, "SOL", ["A", "B"])
    x = np.array([[0.1], [0.5], [0.83]])
    d = phase.interdiffusion_matrix(x, T)[:, 0, 0]
    dstar = R * T * phase.mobility(x, T)
    xa = x[:, 0]
    np.testing.assert_allclose(d, (1 - xa) * dstar[:, 0] + xa * dstar[:, 1], rtol=1e-11)


def test_dilute_limit_is_tracer_diffusivity(ideal_db):
    """As x_A -> 0 the Darken form gives D~ = D*_A in pure B, finite despite RT/x_A in G''."""
    phase = SolutionPhase(ideal_db, "SOL", ["A", "B"])
    d = phase.interdiffusion_matrix(np.array([[0.0], [1e-9]]), T)[:, 0, 0]
    dstar_pure_b = phase.mobility_model.tracer_diffusivity(np.array([0.0, 1.0]), T)[0]
    np.testing.assert_allclose(d, dstar_pure_b, rtol=1e-8)


def test_volume_frame_with_equal_volumes_is_number_frame(regular_db):
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    dn = phase.interdiffusion_matrix(X, T)
    dv = phase.interdiffusion_matrix(X, T, frame="volume", partial_volumes=np.full(3, 7.1e-6))
    np.testing.assert_allclose(dv, dn, rtol=1e-11)


def regular_mu(x, T):
    """Closed-form chemical potentials of the REGULAR_TERNARY solution.

    mu_i = G + dG/dx_i - sum_k x_k dG/dx_k, with G extended off the simplex as written.
    """
    g0 = np.array([-1000 - 2 * T, 500.0, 0.0])
    idx = {"A": 0, "B": 1, "C": 2}
    gm = x @ g0 + R * T * (x * np.log(x)).sum()
    grad = g0 + R * T * (np.log(x) + 1)
    for (a, b), (l0, l1) in REGULAR_L.items():
        i, j = idx[a], idx[b]
        bracket = l0(T) + l1 * (x[i] - x[j])
        gm += x[i] * x[j] * bracket
        grad[i] += x[j] * bracket + x[i] * x[j] * l1
        grad[j] += x[i] * bracket - x[i] * x[j] * l1
    gm += REGULAR_LABC * x[0] * x[1] * x[2]
    grad += REGULAR_LABC * np.array([x[1] * x[2], x[0] * x[2], x[0] * x[1]])
    return gm + grad - x @ grad


def test_volume_frame_fluxes_from_first_principles(regular_db):
    """J^V from a smooth profile computed two ways.

    Directly as J'_i = -c_i M_i dmu_i/dz on the lattice, then J^V_k = J'_k - c_k sum_i V_i J'_i,
    and through the matrix as J^V_k = -sum_j D^V_kj dc_j/dz. Gradients are central differences in z.
    """
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    vol = np.array([6.0e-6, 9.0e-6, 12.0e-6])

    def profile(z):
        x1 = 0.30 + 0.10 * np.tanh(z)
        x2 = 0.25 - 0.08 * np.tanh(0.7 * z + 0.2)
        return np.array([x1, x2, 1 - x1 - x2])

    z0, dz = 0.1, 1e-4
    x = profile(z0)
    vm = x @ vol
    c = x / vm
    conc = lambda z: profile(z) / (profile(z) @ vol)
    dc_dz = (conc(z0 + dz) - conc(z0 - dz)) / (2 * dz)
    dmu_dz = (regular_mu(profile(z0 + dz), T) - regular_mu(profile(z0 - dz), T)) / (2 * dz)
    mob = phase.mobility(x, T)
    j_lattice = -c * mob * dmu_dz
    j_direct = j_lattice - c * (vol @ j_lattice)
    dv = phase.interdiffusion_matrix(x, T, frame="volume", partial_volumes=vol)
    j_matrix = -dv @ dc_dz[:2]
    np.testing.assert_allclose(j_matrix, j_direct[:2], rtol=1e-6)
    # The dependent flux closes the volume balance sum_k V_k J_k = 0.
    assert abs(vol @ j_direct) < 1e-12 * np.abs(vol * j_direct).max()


def test_number_frame_fluxes_from_first_principles(regular_db):
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    vm = 7.0e-6

    def profile(z):
        x1 = 0.30 + 0.10 * np.tanh(z)
        x2 = 0.25 - 0.08 * np.tanh(0.7 * z + 0.2)
        return np.array([x1, x2, 1 - x1 - x2])

    z0, dz = -0.3, 1e-4
    x = profile(z0)
    dx_dz = (profile(z0 + dz) - profile(z0 - dz)) / (2 * dz)
    dmu_dz = (regular_mu(profile(z0 + dz), T) - regular_mu(profile(z0 - dz), T)) / (2 * dz)
    j_lattice = -(x / vm) * phase.mobility(x, T) * dmu_dz
    j_number = j_lattice - x * j_lattice.sum()
    d = phase.interdiffusion_matrix(x, T)
    np.testing.assert_allclose(-d @ (dx_dz[:2] / vm), j_number[:2], rtol=1e-6)


def test_unstable_node_raises(gap_db):
    phase = SolutionPhase(gap_db, "SOL", ["A", "B"])
    with pytest.raises(UnstableCompositionError):
        phase.interdiffusion_matrix(np.array([[0.1], [0.5]]), 1000.0)
