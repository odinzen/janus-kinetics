"""Late-stage coarsening of a multicomponent alloy, after Philippe and Voorhees, Acta Mater. 61 (2013) 4237.

A spherical particle of radius R in a matrix whose far-field composition is C^inf grows as

    dR/dt = [dC^T G dC^inf - 2 sigma V_m / R] / (R dC^T M^-1 dC)          (eq. 28)

with dC = C^beta - C^alpha the equilibrium tie-line, dC^inf = C^inf - C^alpha, G the Hessian of the matrix molar
Gibbs energy and M the mobility matrix (D = M G), all over the independent components 2..N. Once the distribution
has forgotten its start, the mean radius, the matrix supersaturation and the number density follow

    <R>^3 = K t,   K = 8 sigma V_m / (9 dC^T M^-1 dC)                        (eqs. 41-42)
    C^inf - C^alpha = (3 sigma V_m)^(2/3) dC (dC^T M^-1 dC)^(1/3) / (dC^T G dC) t^(-1/3)   (eq. 43)
    N_v = 0.21 phi_eq / (K t)                                                (eq. 44)

and the precipitate composition drifts by eq. (45). The supersaturation lies along the tie-line, the mobilities set
only its amplitude. The time origin is the start of a distribution of zero mean radius; for data that start
elsewhere compare rates (d<R>^3/dt = K) or use the critical radius, which equals <R> at long times.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# 3 / (4 pi <u^3>) for the LSW distribution in u = R / <R>, whose third moment is 1.12960; the paper rounds it to 0.21
NUMBER_DENSITY_FACTOR = 0.2113424


def _vectors(dc, mobility):
    dc = np.atleast_1d(np.asarray(dc, dtype=float))
    m = np.atleast_2d(np.asarray(mobility, dtype=float))
    if m.shape != (dc.size, dc.size):
        raise ValueError("the mobility matrix must be square over the same independent components as dC")
    q = float(dc @ np.linalg.solve(m, dc))
    if not q > 0:
        raise ValueError("dC^T M^-1 dC must be positive; the mobility matrix is not positive definite along dC")
    return dc, q


def _check(molar_volume, gamma):
    if molar_volume <= 0 or gamma <= 0:
        raise ValueError("the molar volume and interfacial energy must be positive")


def coarsening_rate_constant(dc: Sequence[float], mobility, molar_volume: float, gamma: float) -> float:
    """K in <R>^3 = K t (m^3/s), eq. (42). ``dc`` is C^beta - C^alpha over the independent components, ``mobility``
    the matrix M with D = M G (m^2 mol / (J s)), ``molar_volume`` V_m of the precipitate (m^3/mol) and ``gamma`` the
    interfacial energy (J/m^2). For a diagonal M this is eq. (47)."""
    _check(molar_volume, gamma)
    _, q = _vectors(dc, mobility)
    return 8 * gamma * molar_volume / (9 * q)


def critical_radius(dc: Sequence[float], hessian, supersaturation: Sequence[float], molar_volume: float,
                    gamma: float) -> float:
    """R_c = 2 sigma V_m / (dC^T G dC^inf), eq. (29): the radius that neither grows nor shrinks."""
    _check(molar_volume, gamma)
    dc = np.atleast_1d(np.asarray(dc, dtype=float))
    force = float(dc @ np.atleast_2d(hessian) @ np.atleast_1d(np.asarray(supersaturation, dtype=float)))
    return 2 * gamma * molar_volume / force if force > 0 else np.inf


def matrix_supersaturation(t, dc: Sequence[float], mobility, hessian, molar_volume: float, gamma: float) -> np.ndarray:
    """C^inf - C^alpha at times ``t`` (s), eq. (43); one row per time."""
    _check(molar_volume, gamma)
    dc, q = _vectors(dc, mobility)
    curvature = float(dc @ np.atleast_2d(hessian) @ dc)
    if not curvature > 0:
        raise ValueError("dC^T G dC must be positive; the matrix Hessian is not stable along the tie-line")
    t = np.atleast_1d(np.asarray(t, dtype=float))
    amplitude = (3 * gamma * molar_volume) ** (2 / 3) * q ** (1 / 3) / curvature
    return amplitude * np.outer(t ** (-1 / 3), dc)


def number_density(t, rate_constant: float, volume_fraction: float):
    """Particles per volume at time ``t``, eq. (44), with the LSW factor 0.2113 the paper rounds to 0.21."""
    return NUMBER_DENSITY_FACTOR * volume_fraction / (rate_constant * np.asarray(t, dtype=float))


def precipitate_composition_shift(t, dc: Sequence[float], mobility, hessian, precipitate_hessian, molar_volume: float,
                                  gamma: float, partial_volume_difference: Sequence[float] | None = None) -> np.ndarray:
    """<C^beta> - C^beta_eq of the average particle at times ``t``, eq. (45), one row per time. The term in
    ``precipitate_hessian`` turns the shift off the tie-line unless the two Hessians agree;
    ``partial_volume_difference`` is V_i^beta - V_1^beta for i = 2..N (zero when omitted)."""
    _check(molar_volume, gamma)
    dc, q = _vectors(dc, mobility)
    g_a = np.atleast_2d(np.asarray(hessian, dtype=float))
    g_b = np.atleast_2d(np.asarray(precipitate_hessian, dtype=float))
    dv = np.zeros_like(dc) if partial_volume_difference is None else np.asarray(partial_volume_difference, dtype=float)
    direction = np.linalg.solve(g_b, g_a @ dc) / float(dc @ g_a @ dc) - np.linalg.solve(g_b, dv) / molar_volume
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return (3 * gamma * molar_volume) ** (2 / 3) * q ** (1 / 3) * np.outer(t ** (-1 / 3), direction)
