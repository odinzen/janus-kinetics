"""Diffusion in a long cylinder and a sphere, against series solutions derived here.

Constant D, uniform start c0, surface r = a held at cs. For the sphere u = r (c - cs)
obeys u_t = D u_rr with u(0) = u(a) = 0, so by separation of variables

    c(r, t) - cs = (c0 - cs) (2 a / (pi r)) sum_n (-1)^(n+1) / n sin(n pi r / a) exp(-n^2 pi^2 D t / a^2),

which is 2 (c0 - cs) sum_n (-1)^(n+1) exp(...) at the centre, and the fraction of the
change taken up, M_t / M_inf = 1 - (6 / pi^2) sum_n exp(-n^2 pi^2 D t / a^2) / n^2. For the
cylinder the radial eigenfunctions are J0(alpha_n r) with J0(alpha_n a) = 0; projecting the
uniform start onto them (int_0^a r J0(alpha r) dr = a J1(alpha a) / alpha, norm a^2 J1^2 / 2) gives

    c(r, t) - cs = (c0 - cs) (2 / a) sum_n J0(alpha_n r) / (alpha_n J1(alpha_n a)) exp(-alpha_n^2 D t),

with M_t / M_inf = 1 - sum_n 4 / (alpha_n a)^2 exp(-alpha_n^2 D t).
"""

import numpy as np
import pytest
from scipy.special import j0, j1, jn_zeros

from calphad_kinetics import Boundary, solve_planar, uniform_faces
from calphad_kinetics.solver import cell_measures

A, D, C0, CS = 1e-4, 1e-12, 0.10, 0.02
TIMES = np.array([200.0, 1000.0, 3000.0])


def _run(geometry, n=400, dt=0.25):
    faces = uniform_faces(A, n)
    return solve_planar(faces, np.full(n, C0), np.array([[D]]), TIMES[-1], dt, right=Boundary.fixed([CS]),
                        save_times=TIMES, geometry=geometry)


def sphere_exact(r, t, terms=400):
    n = np.arange(1, terms + 1)[:, None]
    arg = n * np.pi * np.asarray(r)[None] / A
    shape = np.where(arg > 0, np.sin(arg) / np.where(arg > 0, arg, 1), 1.0)  # sin(x)/x
    return CS + (C0 - CS) * 2 * ((-1.0) ** (n + 1) * shape * np.exp(-(n * np.pi) ** 2 * D * t / A**2)).sum(axis=0)


def cylinder_exact(r, t, terms=400):
    alpha = jn_zeros(0, terms)[:, None] / A
    series = j0(alpha * np.asarray(r)[None]) / (alpha * j1(alpha * A)) * np.exp(-alpha**2 * D * t)
    return CS + (C0 - CS) * 2 / A * series.sum(axis=0)


@pytest.mark.parametrize("geometry, exact", [("spherical", sphere_exact), ("cylindrical", cylinder_exact)])
def test_profiles_match_the_series(geometry, exact):
    res = _run(geometry)
    for k, t in enumerate(TIMES, start=1):
        np.testing.assert_allclose(res.x[k, :, 0], exact(res.z, t), atol=3e-4 * (C0 - CS))


@pytest.mark.parametrize("geometry, exact", [("spherical", sphere_exact), ("cylindrical", cylinder_exact)])
def test_error_falls_with_the_step(geometry, exact):
    """Backward Euler is first order in time: eight times smaller steps (and twice the cells) cut the
    largest error at 200 s about eightfold, from 0.16 % to 0.02 % of the change."""
    coarse, fine = _run(geometry, 200, 2.0), _run(geometry, 400, 0.25)
    err = [np.abs(r.x[1, :, 0] - exact(r.z, TIMES[0])).max() for r in (coarse, fine)]
    assert 6 < err[0] / err[1] < 10


def test_uptake_matches_the_series():
    sph, cyl = _run("spherical"), _run("cylindrical")
    n = np.arange(1, 400)
    alpha_a = jn_zeros(0, 400)
    for k, t in enumerate(TIMES, start=1):
        f_sph = 1 - 6 / np.pi**2 * (np.exp(-(n * np.pi) ** 2 * D * t / A**2) / n**2).sum()
        f_cyl = 1 - (4 / alpha_a**2 * np.exp(-alpha_a**2 * D * t / A**2)).sum()
        for res, f, p in ((sph, f_sph, 2), (cyl, f_cyl, 1)):
            total = A ** (p + 1) / (p + 1)
            taken = (C0 * total - res.inventory()[k, 0]) / ((C0 - CS) * total)
            assert taken == pytest.approx(f, rel=2e-3)


def test_centre_concentration_of_the_sphere():
    """At the centre the series is 2 (c0 - cs) sum (-1)^(n+1) exp(-n^2 pi^2 D t / a^2) + cs; the first
    cell centre sits half a cell out, where the profile is flat to second order."""
    res = _run("spherical")
    for k, t in enumerate(TIMES, start=1):
        assert res.x[k, 0, 0] == pytest.approx(sphere_exact([0.0], t)[0], abs=3e-4 * (C0 - CS))


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_closed_curved_domains_conserve_solute(geometry):
    """No flux at either end and D rising tenfold with composition: the inventory sum_i V_i x_i stays put."""
    faces = uniform_faces(A, 80)
    x0 = np.where(0.5 * (faces[1:] + faces[:-1]) < A / 2, 0.3, 0.05)
    res = solve_planar(faces, x0, lambda x: (1e-13 + 1e-12 * x)[..., None], 3e4, 100.0, geometry=geometry)
    inv = res.inventory()[:, 0]
    assert abs(inv[-1] - inv[0]) < 1e-12 * inv[0]
    assert res.x[-1, :, 0].std() < 0.2 * res.x[0, :, 0].std()


def test_cell_measures_add_up_to_the_volume():
    faces = np.array([0.0, 1.0, 3.0, 4.0])
    assert cell_measures(faces, "planar").sum() == pytest.approx(4.0)
    assert cell_measures(faces, "cylindrical").sum() == pytest.approx(4.0**2 / 2)
    assert cell_measures(faces, "spherical").sum() == pytest.approx(4.0**3 / 3)


def test_geometry_inputs_are_checked():
    faces = uniform_faces(A, 10)
    with pytest.raises(ValueError, match="geometry"):
        solve_planar(faces, np.full(10, C0), np.array([[D]]), 1.0, 1.0, geometry="conical")
    with pytest.raises(ValueError, match="r = 0"):
        solve_planar(faces, np.full(10, C0), np.array([[D]]), 1.0, 1.0, left=Boundary.fixed([0.1]), geometry="spherical")
    with pytest.raises(ValueError, match="planar geometry only"):
        solve_planar(faces, np.full(10, C0), np.array([[D]]), 1.0, 1.0, geometry="cylindrical", markers=[1e-5],
                     lattice_velocity=lambda x, g: 0 * g[..., 0])


# --- growth of a cylinder and a sphere, Zener (1949) eqs. (10)-(12) ----------------------------------------
from scipy.integrate import quad  # noqa: E402
from scipy.special import erfc, exp1  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from calphad_kinetics.moving_boundary import FixedInterface, solve_moving_boundary  # noqa: E402

N0, N1, NINF, DM = 0.95, 0.05, 0.20, 1e-14


def phi(x, lam):
    """Zener eq. (11), the integral of exp(-z^2/4) z^(1-lambda) from x to infinity, in closed form:
    E1(x^2/4) / 2 for lambda = 2 (substitute u = z^2/4), and exp(-x^2/4)/x - (sqrt(pi)/2) erfc(x/2) for
    lambda = 3 (integrate by parts); checked against quadrature below."""
    if lam == 2:
        return 0.5 * exp1(x * x / 4)
    return np.exp(-x * x / 4) / x - np.sqrt(np.pi) / 2 * erfc(x / 2)


def test_closed_forms_match_quadrature():
    for lam in (2, 3):
        for x in (0.05, 0.5, 2.0):
            ref = quad(lambda z: np.exp(-z * z / 4) * z ** (1 - lam), x, np.inf, limit=200)[0]
            assert phi(x, lam) == pytest.approx(ref, rel=1e-8)


def zener_alpha(lam, xi):
    """Eq. (12): xi^-1 alpha^lambda = 2 exp(-alpha^2/4) / phi_lambda(alpha)."""
    return brentq(lambda a: a**lam / xi - 2 * np.exp(-a * a / 4) / phi(a, lam), 1e-6, 20)


def test_zener_growth_coefficients_lie_in_the_stated_ranges():
    """Eq. (4): K3 = alpha3 (n0 - n_inf)^1/2 / (n_inf - n1)^1/2 lies between 1.4 and 2.4 as n_inf runs from
    n1 to n0, which checks the transcription of eqs. (11)-(12) for lambda = 3."""
    for ninf in (0.051, 0.2, 0.5, 0.9):
        a = zener_alpha(3, (ninf - N1) / (N0 - N1))
        k3 = a * np.sqrt(N0 - ninf) / np.sqrt(ninf - N1)
        assert 1.4 - 0.05 < k3 < 2.4 + 0.05


def zener_curved(geometry, n_right=1200, steps=1600):
    lam = {"cylindrical": 2, "spherical": 3}[geometry]
    alpha = zener_alpha(lam, (NINF - N1) / (N0 - N1))
    t0, t1 = 1e3, 1e5
    s0 = alpha * np.sqrt(DM * t0)
    length = 30 * np.sqrt(DM * t1)
    zr = s0 + (length - s0) * (np.arange(n_right) + 0.5) / n_right
    x_right = NINF + (N1 - NINF) * np.array([phi(z / np.sqrt(DM * t0), lam) for z in zr]) / phi(alpha, lam)  # eq. (10)
    res = solve_moving_boundary(length, s0, np.full(20, N0), x_right, 1e-30, DM, FixedInterface([N0], [N1]),
                                t1 - t0, (t1 - t0) / steps, geometry=geometry)
    return res, alpha, lam, t1


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_zener_curved_growth(geometry):
    """Started from Zener's similarity profile at 1e3 s, the particle radius at 1e5 s is alpha sqrt(D t)
    within 0.5 % (0.24 % for the cylinder, 0.35 % for the sphere), the matrix profile follows eq. (10), and
    the solute is conserved."""
    res, alpha, lam, t1 = zener_curved(geometry)
    assert res.interface[-1] == pytest.approx(alpha * np.sqrt(DM * t1), rel=5e-3)
    z = 0.5 * (res.z_right[-1, 1:] + res.z_right[-1, :-1])
    exact = NINF + (N1 - NINF) * np.array([phi(r / np.sqrt(DM * t1), lam) for r in z]) / phi(alpha, lam)
    assert np.max(np.abs(res.x_right[-1, :, 0] - exact)) < 5e-3
    inv = res.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-10 * inv[0, 0]
    # the stretching core grid keeps a uniform composition uniform only if each face sweeps the right volume
    np.testing.assert_allclose(res.x_left[-1, :, 0], N0, atol=1e-12)


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_zener_curved_error_falls_with_the_grid(geometry):
    """Doubling the matrix cells cuts the radius error to about 0.6 of its value, first order in the grid as
    for the planar interface (its one-sided interface gradient)."""
    errors = []
    for n in (1200, 2400):
        res, alpha, _, t1 = zener_curved(geometry, n, 1600)
        errors.append(abs(res.interface[-1] / (alpha * np.sqrt(DM * t1)) - 1))
    assert errors[1] < 0.75 * errors[0]


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_two_diffusing_shells_conserve_solute(geometry):
    """Core and shell both diffusing (the core 3 times slower), interface compositions held: every element is
    conserved to round-off, which needs the end fluxes of both shells on the same per-area footing."""
    nl, nr = 60, 120
    res = solve_moving_boundary(3e-6, 1e-6, np.linspace(0.90, 0.80, nl), np.linspace(0.30, 0.10, nr), DM / 3, DM,
                                FixedInterface([0.85], [0.25]), 200.0, 1.0, geometry=geometry)
    inv = res.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-11 * inv[0, 0]
    assert abs(res.interface[-1] - res.interface[0]) > 1e-8  # the interface did move
