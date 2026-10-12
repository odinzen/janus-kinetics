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


def diffusion_distance_factor(omega) -> np.ndarray:
    """n = Omega / (2 lambda^2) of eq. (5); 1 where Omega <= 0, below 1e-12, or at or above 1 - 1e-12 taken there."""
    om = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.ones_like(om)
    for i, o in enumerate(om):
        if o > 1e-12:
            o = min(o, 1 - 1e-12)
            out[i] = o / (2 * growth_parameter(o) ** 2)
    return out if np.ndim(omega) else out[0]


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
