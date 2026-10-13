"""The operating point of a dendrite tip in a multicomponent alloy, as Kattner, Boettinger and Coriell,
Z. Metallkd. 87 (1996) 522, set it out (eqs. 2-5): the Ivantsov solution for each solute's diffusion field around
a paraboloid tip, marginal stability for its radius, and the liquidus at the tip's liquid composition for its
temperature, with the partition ratios and liquidus slopes from the phase equilibrium there.

For solute i with Peclet number P_i = V R / (2 D_i) the liquid at the tip holds
C*_i = C0_i / [1 - (1 - k_i) Iv(P_i)], Iv(P) = P exp(P) E1(P) (eq. 2), and the radius is
R = 2 pi [Gamma / (sum_i m_i G_ci xi_i - G)]^(1/2) (eq. 3), Gamma = T_M sigma / Delta H_v the Gibbs-Thomson
coefficient (K m), G the temperature gradient in the liquid, m_i the liquidus slopes and G_ci the solute gradients
at the tip. The tip runs at T* = T_L(C*) minus its curvature undercooling (eq. 5). The paper takes xi_i = 1, right
at the small Peclet numbers of castings, and so does this module.

Two places where the printed equations are corrected here. Eq. (4) prints G_ci = 2 P_i (1 - k_i) C*_i / R; the
liquid concentration falls away from a tip that rejects solute, so the gradient is minus that,
-V (1 - k_i) C*_i / D_i, and only with that sign is m_i G_ci positive for every solute (m_i < 0 when k_i < 1,
m_i > 0 when k_i > 1) and eq. (3) real. Eq. (5) prints the curvature term as T_M Gamma / (2 R); the tip of a
paraboloid of revolution has both principal radii R, so the Gibbs-Thomson undercooling is 2 Gamma / R, four times
the printed term.

The paper's own numbers (Figs. 4-8, IN718 and Rene N4) come from a preliminary ten-component database that was not
published, so they cannot be reproduced; the tests check the equations, their limits, and the paper's statement
(p. 527) that with equal liquid diffusivities and no curvature undercooling the tip liquid is the liquid of a lever
rule calculation at the tip temperature.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import exp1, hyperu

from .errors import ConvergenceError
from .phase import SolutionPhase
from .thermo import R as GAS_CONSTANT


def ivantsov(peclet):
    """Iv(P) = P exp(P) E1(P), the supersaturation a paraboloid needle growing at Peclet number P needs (eq. 2)."""
    p = np.asarray(peclet, dtype=float)
    if np.any(p <= 0):
        raise ValueError("the Peclet number must be positive")
    # exp(P) E1(P) overflows past P ~ 700; U(1, 1, P) is the same function without the overflow
    small = np.minimum(p, 300.0)
    return np.where(p <= 300.0, small * np.exp(small) * exp1(small), p * hyperu(1.0, 1.0, np.maximum(p, 300.0)))


@dataclass(frozen=True)
class Liquidus:
    """The liquidus at one liquid composition: its temperature (K), the solid on it (independent mole fractions),
    the partition ratios k_i = x_S,i / x_L,i and the slopes m_i = dT_L / dx_L,i (K per mole fraction, the other
    independent fractions held)."""

    temperature: float
    solid: np.ndarray
    partition: np.ndarray
    slopes: np.ndarray


def _mu_difference(solid, liquid, xs, xl, T):
    return (solid.chemical_potentials(xs, T) - liquid.chemical_potentials(xl, T)) / (GAS_CONSTANT * T)


def liquidus(solid: SolutionPhase, liquid: SolutionPhase, x_liquid: Sequence[float], temperature: float,
             x_solid: Sequence[float] | None = None, tol: float = 1e-12, max_iter: int = 100) -> Liquidus:
    """The temperature at which ``liquid`` of composition ``x_liquid`` (independent mole fractions, the phases'
    element order) is in equilibrium with ``solid``: every element's chemical potential equal in both, solved by
    Newton from ``temperature`` and ``x_solid`` (the liquid composition by default). The slopes follow from the
    same Jacobian by implicit differentiation."""
    xl = np.asarray(x_liquid, dtype=float)
    xs = xl.copy() if x_solid is None else np.asarray(x_solid, dtype=float).copy()
    T = float(temperature)
    h = 1e-3  # K, for the temperature column; the chemical potentials are smooth in T
    for _ in range(max_iter):
        f = _mu_difference(solid, liquid, xs, xl, T)
        jac = np.column_stack([solid.dmu_dx(xs, T) / (GAS_CONSTANT * T),
                               (_mu_difference(solid, liquid, xs, xl, T + h) - _mu_difference(solid, liquid, xs, xl, T - h)) / (2 * h)])
        step = -np.linalg.solve(jac, f)
        dx, dT = step[:-1], step[-1]
        # keep the solid inside the simplex, going at most halfway to an edge
        full, dfull = np.append(xs, 1 - xs.sum()), np.append(dx, -dx.sum())
        shrink = dfull < 0
        scale = min(1.0, 0.5 * np.min(-full[shrink] / dfull[shrink])) if shrink.any() else 1.0
        xs, T = xs + scale * dx, T + scale * dT
        if np.max(np.abs(f)) < tol and scale == 1.0:
            break
    else:
        raise ConvergenceError(f"liquidus not found from {temperature:g} K at x_L = {xl}")
    # F(xs, T; xl) = 0, so J d(xs, T)/dxl = dmu_L/dxl / RT
    sens = np.linalg.solve(jac, liquid.dmu_dx(xl, T) / (GAS_CONSTANT * T))
    return Liquidus(T, xs, xs / xl, sens[-1])


@dataclass(frozen=True)
class DendriteTip:
    """A dendrite tip at ``velocity`` (m/s) in a liquid gradient ``gradient`` (K/m): its radius (m), the solutes'
    Peclet numbers, the liquid and solid at the tip (independent mole fractions), the partition ratios and liquidus
    slopes used, the liquidus at the tip liquid T_L(C*), the curvature undercooling 2 Gamma / R and the tip
    temperature T* = T_L(C*) - 2 Gamma / R (K)."""

    velocity: float
    gradient: float
    radius: float
    peclet: np.ndarray
    liquid: np.ndarray
    solid: np.ndarray
    partition: np.ndarray
    slopes: np.ndarray
    liquidus_temperature: float
    curvature_undercooling: float
    temperature: float


def _tip_radius(velocity, gradient, alloy, k, m, d, gibbs_thomson):
    """Eqs. (2)-(4) for fixed k and m: R with R^2 (sum m_i G_ci - G) = 4 pi^2 Gamma, G_ci = -V (1 - k_i) C*_i / D_i."""
    if np.any(m * (k - 1) <= 0):
        raise ValueError("every solute must lower the liquidus as it is rejected (m_i (k_i - 1) > 0)")

    def residual(log_r):
        r = np.exp(log_r)
        c = alloy / (1 - (1 - k) * ivantsov(velocity * r / (2 * d)))
        return r**2 * (np.sum(m * (k - 1) * velocity * c / d) - gradient) - 4 * np.pi**2 * gibbs_thomson

    # far from the tip the solute gradients settle at V (1 - k) C0 / (k D); below that bound in sum the gradient wins
    # everywhere and the front is planar
    if np.sum(m * (k - 1) * velocity * alloy / (k * d)) <= gradient:
        raise ValueError("the velocity is below the limit of constitutional undercooling: the front stays planar")
    lo, hi = np.log(1e-12), np.log(1e-12)
    while residual(hi) < 0:
        lo, hi = hi, hi + np.log(10.0)
        if hi > np.log(1e6):
            raise ConvergenceError("no tip radius below 1000 km")
    return float(np.exp(brentq(residual, lo, hi, xtol=1e-14, rtol=1e-13)))


def dendrite_tip(velocity: float, gradient: float, alloy: Sequence[float], partition: Sequence[float],
                 slopes: Sequence[float], diffusivities, gibbs_thomson: float, alloy_liquidus: float) -> DendriteTip:
    """The tip of eqs. (2)-(5) for a linear phase diagram: constant partition ratios and liquidus slopes, so that
    T_L(C*) = T_L(C0) + sum_i m_i (C*_i - C0_i) with ``alloy_liquidus`` = T_L(C0). ``diffusivities`` are the
    solutes' liquid diffusivities (m^2/s, a scalar for all), ``gibbs_thomson`` Gamma in K m."""
    c0, k, m = (np.asarray(a, dtype=float) for a in (alloy, partition, slopes))
    d = np.broadcast_to(np.asarray(diffusivities, dtype=float), c0.shape)
    r = _tip_radius(velocity, gradient, c0, k, m, d, gibbs_thomson)
    p = velocity * r / (2 * d)
    c = c0 / (1 - (1 - k) * ivantsov(p))
    tl = alloy_liquidus + float(np.sum(m * (c - c0)))
    curv = 2 * gibbs_thomson / r
    return DendriteTip(velocity, gradient, r, p, c, k * c, k, m, tl, curv, tl - curv)


def dendrite_tip_calphad(solid: SolutionPhase, liquid: SolutionPhase, velocity: float, gradient: float,
                         alloy: Sequence[float], diffusivities, gibbs_thomson: float, temperature: float,
                         tol: float = 1e-10, max_iter: int = 200) -> DendriteTip:
    """The tip of eqs. (2)-(5) with k_i and m_i from the solid-liquid equilibrium at T_L(C*), iterated with the
    tip equations until C* settles, as the paper does (p. 526). ``temperature`` is a guess of the alloy's liquidus;
    compositions are independent mole fractions in the phases' element order."""
    c0 = np.asarray(alloy, dtype=float)
    d = np.broadcast_to(np.asarray(diffusivities, dtype=float), c0.shape)
    eq = liquidus(solid, liquid, c0, temperature)
    c = c0
    for _ in range(max_iter):
        r = _tip_radius(velocity, gradient, c0, eq.partition, eq.slopes, d, gibbs_thomson)
        p = velocity * r / (2 * d)
        new = c0 / (1 - (1 - eq.partition) * ivantsov(p))
        eq = liquidus(solid, liquid, new, eq.temperature, eq.solid)
        if np.max(np.abs(new - c) / new) < tol:
            break
        c = new
    else:
        raise ConvergenceError("the tip composition and the liquidus at it did not settle")
    curv = 2 * gibbs_thomson / r
    return DendriteTip(velocity, gradient, r, p, new, eq.solid, eq.partition, eq.slopes, eq.temperature, curv,
                       eq.temperature - curv)
