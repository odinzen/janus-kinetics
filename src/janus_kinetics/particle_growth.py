"""Growth of a spherical particle in a multicomponent matrix with local equilibrium at its interface, after Chen,
Jeppsson and Agren, Acta Mater. 56 (2008) 1890.

For a binary the exact growth of a sphere from zero size is v = 2 D lambda^2 / R with

    2 lambda^2 - 2 lambda^3 sqrt(pi) exp(lambda^2) erfc(lambda) = Omega,   Omega = (c^M - c^I) / (c^P - c^I)    (eqs. 1-2)

which they write as the steady-state flux over an effective distance n R, v (c^P - c^I) = D (c^M - c^I) / (n R) with
n = Omega / (2 lambda^2) (eqs. 4-6): n -> 1 at small supersaturation, n -> Omega (1 - Omega) / 3 -> 0 near Omega = 1.
In a multicomponent matrix each element j keeps its own factor n_j from its own Omega_j:

    v (c^P_i - c^I_i) = sum_j D_ij (c^M_j - c^I_j) / (n_j R)              (eq. 10, "indirect")
    v (c^P_i - c^I_i) = c^I_i M_i (mu^M_i - mu^I_i) / (n_i R)             (eq. 13, "direct")

closed by local equilibrium, mu^P_i + 2 sigma V_m / R = mu^I_i for every element (eq. 9 with the Gibbs-Thomson
pressure on the particle). Concentrations are mole fractions over one molar volume, so the volume cancels. With a
diagonal D, no capillarity and fixed interface compositions every element's field is the exact binary one, so eq. (10)
is then exact for parabolic growth from zero size. The particle may be a solution phase whose interior does not
diffuse, or nearly stoichiometric.

Where an element's Omega_j is not in (0, 1) (it diffuses against its partitioning through the others), eq. (2) has
no root; this package takes n_j = 1 there, the steady-state value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq, root
from scipy.special import erfcx

from .phase import SolutionPhase
from .thermo import R


def _f(lam):
    return 2 * lam**2 * (1 - lam * np.sqrt(np.pi) * erfcx(lam))


def _one_minus_f(lam):
    # 1 - f cancels in floating point at large lambda; there it is the asymptotic series of erfcx
    if lam < 30:
        return 1 - _f(lam)
    y = 1 / lam**2
    return y * (1.5 - y * (3.75 - y * (13.125 - y * 59.0625)))


def growth_parameter(omega: float) -> float:
    """lambda of eq. (2) for a supersaturation 0 < Omega < 1."""
    if not 0 < omega < 1:
        raise ValueError("eq. (2) has a root only for 0 < Omega < 1")
    hi = 1.0
    while _f(hi) < omega and hi < 1e9:
        hi *= 4
    if omega < 0.9:
        return brentq(lambda lam: _f(lam) - omega, 0.0, hi, xtol=1e-14 * hi, rtol=1e-14)
    rest = 1 - omega
    return brentq(lambda lam: _one_minus_f(lam) - rest, 0.5, max(hi, 2 / np.sqrt(rest)), rtol=1e-14)


def _log_f_and_slope(t, upper):
    """ln f (or ln(1 - f) where ``upper``) at lambda = e^t and its derivative in t."""
    t = np.asarray(t, dtype=float)
    upper = np.broadcast_to(upper, t.shape)
    lam = np.exp(t)
    value, slope = np.empty_like(lam), np.empty_like(lam)
    far = lam >= 30  # only reached on the 1 - f branch, by the series
    near = ~far
    ln = lam[near]
    e = erfcx(ln)
    f = 2 * ln**2 * (1 - ln * np.sqrt(np.pi) * e)
    df = 4 * ln * (1 - ln * np.sqrt(np.pi) * e) - 2 * ln**2 * np.sqrt(np.pi) * (e + ln * (2 * ln * e - 2 / np.sqrt(np.pi)))
    up = upper[near]
    value[near] = np.where(up, np.log(np.maximum(1 - f, 1e-300)), np.log(np.maximum(f, 1e-300)))
    slope[near] = np.where(up, -df / np.maximum(1 - f, 1e-300), df / np.maximum(f, 1e-300)) * ln
    if far.any():
        lf = lam[far]
        y = 1 / lf**2
        series = y * (1.5 - y * (3.75 - y * (13.125 - y * 59.0625)))
        dseries = (1.5 - y * (7.5 - y * (39.375 - y * 236.25))) * (-2 * y / lf)
        uf = upper[far]
        value[far] = np.where(uf, np.log(series), np.log1p(-series))
        slope[far] = np.where(uf, dseries / series, -dseries / (1 - series)) * lf
    return value, slope


# each branch tabulated only around where it is used (Omega = 1/2 has lambda = 1.04), so that
# neither table reaches the end where its value is rounding noise and stops being monotonic
_T_LOW = np.linspace(np.log(1e-7), np.log(3.0), 20001)
_T_HIGH = np.linspace(np.log(0.3), np.log(1e7), 20001)
_LOW = _log_f_and_slope(_T_LOW, False)
_HIGH = _log_f_and_slope(_T_HIGH, True)


def _inverse(target, table, t):
    """t at ``target`` by cubic Hermite interpolation of the inverse of the tabulated (value, slope)."""
    v, slope = table
    if v[-1] < v[0]:
        v, slope, t = v[::-1], slope[::-1], t[::-1]
    i = np.clip(np.searchsorted(v, target) - 1, 0, v.size - 2)
    h = v[i + 1] - v[i]
    s = (target - v[i]) / h
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * t[i] + (s3 - 2 * s2 + s) * h / slope[i]
            + (-2 * s3 + 3 * s2) * t[i + 1] + (s3 - s2) * h / slope[i + 1])


def diffusion_distance_factor(omega) -> np.ndarray:
    """n = Omega / (2 lambda^2) of eq. (5); 1 where Omega <= 0, below 1e-12, or at or above 1 - 1e-12 taken there.
    Vectorized: ln lambda by cubic Hermite interpolation of the inverse of ln f below Omega = 1/2 and of ln(1 - f)
    above (which keeps both ends free of cancellation), tabulated with their slopes."""
    om = np.asarray(omega, dtype=float)
    o = np.clip(om, 1e-12, 1 - 1e-12)
    upper = o >= 0.5
    t = np.where(upper, _inverse(np.log(1 - o), _HIGH, _T_HIGH), _inverse(np.log(o), _LOW, _T_LOW))
    lam = np.exp(t)
    n = np.where(om > 1e-12, o / (2 * lam**2), 1.0)
    return n if np.ndim(omega) else float(n)


@dataclass(frozen=True)
class GrowthState:
    """Growth rate ``velocity`` (m/s) of a particle of ``radius``, with all mole fractions of the matrix
    (``matrix_interface``) and particle (``particle_interface``) at its interface, and each independent element's
    supersaturation ``omega`` and distance factor ``factor``."""

    radius: float
    velocity: float
    matrix_interface: np.ndarray
    particle_interface: np.ndarray
    omega: np.ndarray
    factor: np.ndarray


def particle_growth_rate(matrix: SolutionPhase, particle: SolutionPhase, x_matrix: Sequence[float], radius: float,
                         T: float, start: tuple[Sequence[float], Sequence[float]], diffusivity=None,
                         gamma: float = 0.0, molar_volume: float = 1e-5,
                         form: Literal["diffusivity", "mobility"] = "diffusivity") -> GrowthState:
    """Growth rate of a sphere of ``radius`` in a matrix of far-field composition ``x_matrix`` (independent mole
    fractions, the phases' dependent element last), solving eq. (9) with eq. (10) (``form="diffusivity"``, with the
    matrix interdiffusion ``diffusivity``, an (n-1) x (n-1) array in m^2/s) or eq. (13) (``form="mobility"``, with the
    matrix's own mobilities at the interface composition). ``start`` gives the interface compositions (matrix,
    particle) to begin from, for instance a tie-line from pycalphad; where more than one operating tie-line exists
    (partitioning and non-partitioning growth) it decides which is found. ``gamma`` (J/m^2) and ``molar_volume``
    (m^3/mol) add the Gibbs-Thomson pressure on the particle."""
    if matrix.elements != particle.elements:
        raise ValueError(f"element order differs: {matrix.elements} vs {particle.elements}")
    if radius <= 0 or T <= 0 or gamma < 0 or molar_volume <= 0:
        raise ValueError("radius, temperature and molar volume must be positive and gamma not negative")
    xm = np.atleast_1d(np.asarray(x_matrix, dtype=float))
    m = xm.size
    if m != len(matrix.elements) - 1:
        raise ValueError("x_matrix holds the independent mole fractions, one fewer than the elements")
    if form == "diffusivity":
        if diffusivity is None:
            raise ValueError("the diffusivity form needs the matrix interdiffusion matrix")
        d = np.atleast_2d(np.asarray(diffusivity, dtype=float))
        if d.shape != (m, m):
            raise ValueError(f"diffusivity must be {m} x {m}")
        scale = np.abs(np.diag(d))
    elif form == "mobility":
        mu_far = matrix.chemical_potentials(xm, T)
        scale = R * T * matrix.mobility(xm, T)[:m]
    else:
        raise ValueError("form is 'diffusivity' or 'mobility'")
    if np.any(scale <= 0):
        raise ValueError("diffusivities and mobilities must be positive")
    pressure = 2 * gamma * molar_volume / radius
    rt = R * T

    # each phase's mole fractions are carried as x_j = exp(y_j) / (1 + sum_k exp(y_k)), the dependent element
    # taking 1 / (1 + sum), so that no step can leave the composition simplex
    def to_x(y):
        e = np.exp(np.clip(y, -700, 700))
        return e / (1 + e.sum())

    def to_y(x):
        return np.log(x) - np.log(1 - x.sum())

    def parts(u):
        xi, xp, w = to_x(u[:m]), to_x(u[m:2 * m]), u[-1]
        omega = (xm - xi) / np.where(np.abs(xp - xi) > 1e-300, xp - xi, 1e-300)
        return xi, xp, w, omega, diffusion_distance_factor(omega)

    def residual(u):
        xi, xp, w, omega, n = parts(u)
        mu_i = matrix.chemical_potentials(xi, T)
        mu_p = particle.chemical_potentials(xp, T)
        equilibrium = (mu_i - mu_p - pressure) / rt
        # w = v R / D_ref, with D_ref the slowest element's diffusivity, keeps the flux rows of one size
        if form == "diffusivity":
            flux = (w * scale.min() * (xp - xi) - d @ ((xm - xi) / n)) / scale
        else:
            mob = matrix.mobility(xi, T)[:m]
            flux = (w * scale.min() * (xp - xi) - xi * mob * (mu_far[:m] - mu_i[:m]) / n) / scale
        return np.concatenate([equilibrium, flux])

    xi0 = np.atleast_1d(np.asarray(start[0], dtype=float))
    xp0 = np.atleast_1d(np.asarray(start[1], dtype=float))
    if np.any(xi0 <= 0) or np.any(xp0 <= 0) or xi0.sum() >= 1 or xp0.sum() >= 1:
        raise ValueError("the start compositions must lie inside the composition simplex")
    sol = root(residual, np.concatenate([to_y(xi0), to_y(xp0), [0.0]]), method="hybr", options={"xtol": 1e-13})
    if not sol.success or np.max(np.abs(residual(sol.x))) > 1e-8:
        raise RuntimeError(f"no operating tie-line found from the start given: {sol.message}")
    xi, xp, w, omega, n = parts(sol.x)
    full = lambda x: np.append(x, 1 - x.sum())  # noqa: E731
    return GrowthState(radius, float(w * scale.min() / radius), full(xi), full(xp), omega, n)


def _fixed_composition_batch(matrix: SolutionPhase, composition, gibbs, radius, x_far, T: float, gamma: float,
                             molar_volume: float, form: str, diffusivity=None, misfit: float = 0.0, start=None,
                             tol: float = 1e-10, max_iter: int = 60, strict: bool = True):
    """Growth of k particles of fixed mole fractions ``composition`` (k, n) and molar Gibbs energies ``gibbs`` (k,):
    one equilibrium condition each, sum_i x^P_i mu^I_i = G^P + V_m (2 gamma / R + misfit), with the flux balances,
    solved by Newton for all particles at once. Returns velocity (k,), matrix interface (k, n), omega and factor
    (k, n-1) and the solver state (k, n) to start the next call from; with ``strict=False`` also a mask of the
    particles solved, the others (too small for any matrix composition to hold them) left NaN."""
    xp = np.atleast_2d(np.asarray(composition, dtype=float))
    k, n_el = xp.shape
    m = n_el - 1
    g = np.asarray(gibbs, dtype=float).reshape(k)
    rad = np.asarray(radius, dtype=float).reshape(k)
    x0 = np.asarray(x_far, dtype=float)
    rt = R * T
    mu_far = matrix.chemical_potentials(x0[:m], T)
    if form == "diffusivity":
        d = np.atleast_2d(np.asarray(diffusivity, dtype=float))
        scale = np.abs(np.diag(d))
    else:
        scale = rt * matrix.mobility(x0[:m], T)[:m]
    d_ref = scale.min()
    pressure = molar_volume * (2 * gamma / rad + misfit)

    def to_x(y):
        e = np.exp(np.clip(y, -700, 700))
        return e / (1 + e.sum(axis=-1, keepdims=True))

    def parts(u, rows):
        xi = to_x(u[:, :m])
        diff = xp[rows, :m] - xi
        omega = (x0[:m] - xi) / np.where(np.abs(diff) > 1e-14, diff, 1e-14)
        return xi, omega, diffusion_distance_factor(omega)

    def residual(u, rows):
        xi, omega, nf = parts(u, rows)
        mu = matrix.chemical_potentials(xi, T)
        r = np.empty((len(rows), m + 1))
        r[:, 0] = ((xp[rows] * mu).sum(axis=1) - g[rows] - pressure[rows]) / rt
        if form == "diffusivity":
            flux = ((x0[:m] - xi) / nf) @ d.T
        else:
            flux = xi * matrix.mobility(xi, T)[..., :m] * (mu_far[:m] - mu[:, :m]) / nf
        r[:, 1:] = (u[:, -1:] * d_ref * (xp[rows, :m] - xi) - flux) / scale
        return r

    if start is None:
        xi0 = np.clip(np.broadcast_to(x0[:m], (k, m)), 1e-12, None)
        u = np.concatenate([np.log(xi0) - np.log(1 - xi0.sum(axis=1, keepdims=True)), np.zeros((k, 1))], axis=1)
    else:
        u = np.array(start, dtype=float).reshape(k, m + 1)
    # only the particles not yet converged are iterated, so a few new ones do not cost a pass over all
    done = ~np.all(np.isfinite(u), axis=1)
    bad = done.copy()  # a non-finite start cannot be iterated; it is reported unsolved
    for _ in range(max_iter):
        rows = np.flatnonzero(~done)
        if rows.size == 0:
            break
        ua = u[rows]
        r = residual(ua, rows)
        ok = np.all(np.abs(r) < tol, axis=1)
        done[rows[ok]] = True
        rows, ua, r = rows[~ok], ua[~ok], r[~ok]
        if rows.size == 0:
            break
        jac = np.zeros((rows.size, m + 1, m + 1))
        for c in range(m):
            h = 1e-7 * (1 + np.abs(ua[:, c]))
            up = ua.copy()
            up[:, c] += h
            jac[:, :, c] = (residual(up, rows) - r) / h[:, None]
        # the residual is linear in w: no equilibrium term, d_ref (x^P - x^I) / scale in each flux row
        jac[:, 1:, m] = d_ref * (xp[rows, :m] - parts(ua, rows)[0]) / scale
        r = np.nan_to_num(r)
        try:
            step = -np.linalg.solve(jac, r[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # a particle whose rows lost rank (one far outside any solution) must not stall the rest
            step = -(np.linalg.pinv(np.nan_to_num(jac)) @ r[..., None])[..., 0]
        # at most one e-fold of any interface mole fraction per iteration
        big = np.abs(step[:, :m]).max(axis=1, initial=0.0)
        step *= np.minimum(1.0, 1.0 / np.maximum(big, 1e-300))[:, None]
        u[rows] = ua + np.nan_to_num(step)
    done &= ~bad
    if not done.all() and strict:
        raise RuntimeError(f"growth rates did not converge for {int((~done).sum())} of {k} particles")
    xi, omega, nf = parts(np.nan_to_num(u), np.arange(k))
    full = np.concatenate([xi, 1 - xi.sum(axis=1, keepdims=True)], axis=1)
    velocity = u[:, -1] * d_ref / rad
    if strict:
        return velocity, full, omega, nf, u
    u = np.where(done[:, None], u, np.nan)
    return np.where(done, velocity, np.nan), full, omega, nf, u, done


def fixed_composition_growth_rate(matrix: SolutionPhase, composition: Sequence[float], gibbs: float,
                                  x_matrix: Sequence[float], radius: float, T: float, diffusivity=None,
                                  gamma: float = 0.0, molar_volume: float = 1e-5, misfit: float = 0.0,
                                  form: Literal["diffusivity", "mobility"] = "diffusivity") -> GrowthState:
    """Growth rate of a sphere of fixed mole fractions ``composition`` (all n) and molar Gibbs energy ``gibbs``
    (J/mol), a stoichiometric phase or one whose composition is held, as the paper's model allows (p. 1891).
    Local equilibrium is then the single condition sum_i x^P_i mu^I_i = G^P + V_m (2 sigma / R + misfit), with
    ``misfit`` an elastic energy per volume; the flux balances are those of :func:`particle_growth_rate`, whose
    other arguments these are. The matrix interface starts from the far-field composition."""
    xp = np.asarray(composition, dtype=float)
    if xp.shape != (len(matrix.elements),) or abs(xp.sum() - 1) > 1e-9 or np.any(xp < 0):
        raise ValueError("composition needs one mole fraction per element, summing to 1")
    if radius <= 0 or T <= 0 or gamma < 0 or molar_volume <= 0:
        raise ValueError("radius, temperature and molar volume must be positive and gamma not negative")
    if form not in ("diffusivity", "mobility"):
        raise ValueError("form is 'diffusivity' or 'mobility'")
    if form == "diffusivity" and diffusivity is None:
        raise ValueError("the diffusivity form needs the matrix interdiffusion matrix")
    x0 = np.atleast_1d(np.asarray(x_matrix, dtype=float))
    x0 = np.append(x0, 1 - x0.sum()) if x0.size == xp.size - 1 else x0
    v, xi, omega, nf, _ = _fixed_composition_batch(matrix, xp[None], [gibbs], [radius], x0, T, gamma, molar_volume,
                                                   form, diffusivity, misfit)
    return GrowthState(radius, float(v[0]), xi[0], xp.copy(), omega[0], nf[0])
