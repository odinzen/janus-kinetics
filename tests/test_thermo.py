"""Gibbs energy curvature and thermodynamic factor from pycalphad."""

import numpy as np
import pytest
from pycalphad import calculate

from calphad_kinetics import R, SolutionPhase, UnstableCompositionError
from calphad_kinetics.thermo import dmu_dx, reduced_hessian
from conftest import REGULAR_L, REGULAR_LABC

T = 1000.0
COMPOSITIONS = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.05, 0.9, 0.05], [1 / 3, 1 / 3, 1 / 3]])


def regular_full_hessian(x, T):
    """Hessian of RT sum x ln x + sum_ij x_i x_j (L0 + L1 (x_i - x_j)) + x_A x_B x_C L_ABC."""
    idx = {"A": 0, "B": 1, "C": 2}
    h = np.diag(R * T / x)
    for (a, b), (l0, l1) in REGULAR_L.items():
        i, j = idx[a], idx[b]
        l0 = l0(T)
        h[i, i] += 2 * l1 * x[j]
        h[j, j] += -2 * l1 * x[i]
        h[i, j] += l0 + 2 * l1 * (x[i] - x[j])
        h[j, i] = h[i, j]
    for i, j, k in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
        h[i, j] += REGULAR_LABC * x[k]
        h[j, i] = h[i, j]
    return h


def test_ideal_curvature_and_factor(ideal_db):
    """For an ideal solution G''_kj = RT (delta_kj / x_k + 1 / x_n) and the factor is identity-like."""
    phase = SolutionPhase(ideal_db, "SOL", ["A", "B", "C"])
    x = COMPOSITIONS
    g2 = phase.reduced_hessian(x, T)
    expected = R * T * (np.eye(2)[None] / x[:, :2, None] + 1.0 / x[:, 2, None, None])
    np.testing.assert_allclose(g2, expected, rtol=1e-12)

    dmu = phase.dmu_dx(x, T)
    expected_dmu = np.zeros((len(x), 3, 2))
    expected_dmu[:, :2, :] = R * T * np.eye(2)[None] / x[:, :2, None]
    expected_dmu[:, 2, :] = -R * T / x[:, 2, None]
    np.testing.assert_allclose(dmu, expected_dmu, rtol=1e-12, atol=1e-6)

    gamma = phase.thermodynamic_factor(x, T)
    identity_like = np.vstack([np.eye(2), -np.ones((1, 2))])
    np.testing.assert_allclose(gamma, np.broadcast_to(identity_like, gamma.shape), rtol=1e-12, atol=1e-14)


def test_regular_curvature_matches_closed_form(regular_db):
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    for x in COMPOSITIONS:
        h = regular_full_hessian(x, T)
        expected = h[:2, :2] - h[:2, 2:] - h[2:, :2] + h[2:, 2:]
        np.testing.assert_allclose(phase.reduced_hessian(x, T), expected, rtol=1e-10)


def test_curvature_matches_finite_difference_of_pycalphad_calculate(regular_db):
    """Independent path through central differences of GM from pycalphad's compiled ``calculate``."""
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    h = 1e-4  # truncation ~ h^2 RT/x^3, round-off ~ 1e-16 GM / h^2; both < 1e-5 relative here

    def gm(x1, x2):
        pts = np.array([[x1, x2, 1 - x1 - x2, 1.0]])  # site fraction order A, B, C, VA
        res = calculate(regular_db, ["A", "B", "C", "VA"], "SOL", T=T, P=101325, N=1, points=pts, output="GM")
        return float(np.squeeze(res.GM.values))

    for x in COMPOSITIONS[:3]:
        x1, x2 = x[0], x[1]
        g0 = gm(x1, x2)
        d11 = (gm(x1 + h, x2) - 2 * g0 + gm(x1 - h, x2)) / h**2
        d22 = (gm(x1, x2 + h) - 2 * g0 + gm(x1, x2 - h)) / h**2
        d12 = (gm(x1 + h, x2 + h) - gm(x1 + h, x2 - h) - gm(x1 - h, x2 + h) + gm(x1 - h, x2 - h)) / (4 * h**2)
        fd = np.array([[d11, d12], [d12, d22]])
        np.testing.assert_allclose(phase.reduced_hessian(x, T), fd, rtol=2e-5)


def test_gibbs_energy_value_matches_pycalphad(regular_db):
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    x = COMPOSITIONS[0]
    pts = np.array([[*x, 1.0]])
    res = calculate(regular_db, ["A", "B", "C", "VA"], "SOL", T=T, P=101325, N=1, points=pts, output="GM")
    assert phase.gibbs.gm(x, T) == pytest.approx(float(np.squeeze(res.GM.values)), rel=1e-12)


def test_dmu_dx_is_derivative_of_chemical_potential(regular_db):
    """dmu_i/dx_j against differences of mu_i = G + sum_k (delta_ik - x_k) dG/dx_k, built from GM alone."""
    phase = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    h = 1e-6

    def grad_reduced(x1, x2):
        gm = lambda a, b: phase.gibbs.gm(np.array([a, b, 1 - a - b]), T)
        return np.array([(gm(x1 + h, x2) - gm(x1 - h, x2)) / (2 * h), (gm(x1, x2 + h) - gm(x1, x2 - h)) / (2 * h)])

    def mu(x1, x2):
        x = np.array([x1, x2])
        g = phase.gibbs.gm(np.array([x1, x2, 1 - x1 - x2]), T)
        dg = grad_reduced(x1, x2)
        upper = g + dg - x @ dg
        return np.append(upper, g - x @ dg)

    x = COMPOSITIONS[0]
    step = 1e-4
    fd = np.column_stack([
        (mu(x[0] + step, x[1]) - mu(x[0] - step, x[1])) / (2 * step),
        (mu(x[0], x[1] + step) - mu(x[0], x[1] - step)) / (2 * step),
    ])
    np.testing.assert_allclose(phase.dmu_dx(x, T), fd, rtol=1e-4, atol=1.0)


def test_spinodal_raises_named_error(gap_db):
    phase = SolutionPhase(gap_db, "SOL", ["A", "B"])
    with pytest.raises(UnstableCompositionError, match="miscibility gap or two-phase field") as err:
        phase.dmu_dx(np.array([[0.5]]), T)
    assert err.value.eigenvalues.min() < 0
    # x = 0.05: G'' = RT/(x(1-x)) - 2 L0 = 175 kJ - 60 kJ > 0, outside the spinodal.
    phase.dmu_dx(np.array([[0.05]]), T)


def test_reduced_hessian_helpers_shape():
    h = np.arange(16.0).reshape(4, 4)
    h = h + h.T
    g = reduced_hessian(h)
    assert g.shape == (3, 3)
    assert g[0, 1] == h[0, 1] - h[0, 3] - h[3, 1] + h[3, 3]
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert dmu_dx(x, g).shape == (4, 3)
