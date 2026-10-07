"""Conversion of plates, wires and particles through a growing reaction layer.

A plate of thickness 2 r0, or a cylinder or sphere of radius r0, of A reacts with a
matrix B into a compound layer whose growth is limited by diffusion through it
(van Loo, Prog. Solid State Chem. 20 (1990) 47, pp. 74-76, after Carter). The
conversion alpha is the fraction of A used up, z the volume of product formed per
volume of A used up, and k_p the parabolic constant of the same layer in a planar
couple, d^2 = 2 k_p t. Van Loo's eq. 26 relates them:

    plate      (z alpha)^2                                               = 2 k_p t / r0^2
    cylinder   z/2 (1 + (z-1) alpha) ln(1 + (z-1) alpha)/(z-1)
                 + z/2 (1 - alpha) ln(1 - alpha)                         = 2 k_p t / r0^2
    sphere     z^2/(z-1) - z/(z-1) (1 + (z-1) alpha)^(2/3)
                 - z (1 - alpha)^(2/3)                                   = 2 k_p t / r0^2

and eq. 25 gives the layer thickness. The left-hand side follows from steady
diffusion across the curved shell, so it also holds when k_p changes with
temperature during an anneal, with k_p t replaced by the integral of k_p dt.
Core and matrix are taken as rigid: stresses and Kirkendall voiding, which van
Loo warns can dominate in curved geometries (p. 76), are not modelled.

For a layer between line compounds, k_p comes from
:func:`calphad_kinetics.parabolic_growth` and z = (V_layer / x_A,layer) / (V_core / x_A,core).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import xlogy

from .schedule import Temperature, TemperatureSchedule, temperature_at

Geometry = Literal["plate", "cylinder", "sphere"]


def _check(z: float, geometry: str):
    if z <= 0:
        raise ValueError("the volume ratio z must be positive")
    if geometry not in ("plate", "cylinder", "sphere"):
        raise ValueError("geometry must be 'plate', 'cylinder' or 'sphere'")


def conversion_integral(alpha, z: float, geometry: Geometry):
    """Left-hand side of van Loo's eq. 26, equal to 2 k_p t / r0^2 at conversion alpha."""
    _check(z, geometry)
    a = np.asarray(alpha, dtype=float)
    if np.any((a < 0) | (a > 1)):
        raise ValueError("conversion must lie in [0, 1]")
    e = z - 1.0
    if geometry == "plate":
        return (z * a) ** 2
    if geometry == "cylinder":
        # (1 + e a) ln(1 + e a) / e, written to stay accurate as z -> 1, where it tends to a
        grow = (1 + e * a) * (np.log1p(e * a) / e if e != 0 else a)
        return z / 2 * grow + z / 2 * xlogy(1 - a, 1 - a)
    # z^2/e - z/e (1 + e a)^(2/3) = z [1 - expm1(2/3 ln(1 + e a)) / e], which tends to z (1 - 2a/3)
    shell = 1 - np.expm1(2 / 3 * np.log1p(e * a)) / e if e != 0 else 1 - 2 * a / 3
    return z * shell - z * (1 - a) ** (2 / 3)


def layer_thickness(alpha, z: float, geometry: Geometry):
    """Van Loo's eq. 25: layer thickness d / r0 at conversion alpha."""
    _check(z, geometry)
    a = np.asarray(alpha, dtype=float)
    if geometry == "plate":
        return z * a
    n = 2 if geometry == "cylinder" else 3
    return (1 + (z - 1) * a) ** (1 / n) - (1 - a) ** (1 / n)


def _exposure(t, growth_constant, temperature) -> np.ndarray:
    """Integral of k_p dt from 0 to each t, in m^2."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t < 0):
        raise ValueError("times must be non-negative")
    if not callable(growth_constant):
        if temperature is not None:
            raise ValueError("a constant growth constant needs no temperature")
        return growth_constant * t
    if temperature is None:
        raise ValueError("a growth constant that depends on T needs a temperature or schedule")
    kp = lambda s: growth_constant(temperature_at(temperature, s))
    corners = list(temperature.times) if isinstance(temperature, TemperatureSchedule) else []
    out = np.empty_like(t)
    for i, ti in enumerate(t):
        pts = [c for c in corners if 0 < c < ti]
        out[i] = quad(kp, 0.0, ti, points=pts or None, epsabs=0.0, epsrel=1e-12, limit=200)[0] if ti > 0 else 0.0
    return out


def conversion(t, growth_constant: float | Callable[[float], float], radius: float, z: float,
               geometry: Geometry, temperature: Temperature | None = None) -> np.ndarray:
    """Fraction of the plate, cylinder or sphere converted after time t (s).

    ``growth_constant`` is the planar k_p (m^2/s), a number or a function of T
    evaluated along ``temperature`` (a constant or a schedule). ``radius`` is r0,
    the half-thickness of a plate. Eq. 26 is inverted for alpha, which is 1 once
    the core is used up.
    """
    _check(z, geometry)
    if radius <= 0:
        raise ValueError("radius must be positive")
    target = 2 * _exposure(t, growth_constant, temperature) / radius**2
    full = float(conversion_integral(1.0, z, geometry))
    alpha = np.ones_like(target)
    for i, f in enumerate(target):
        if f < full:
            alpha[i] = brentq(lambda a: conversion_integral(a, z, geometry) - f, 0.0, 1.0, xtol=1e-15, rtol=1e-14)
    return alpha


def growth_constant_from_conversion(alpha, t, radius: float, z: float, geometry: Geometry):
    """Planar k_p (m^2/s) from a measured conversion alpha at time t, by eq. 26.

    This is van Loo's use of the curves: powder or wire data give the parabolic
    constant of a planar couple.
    """
    if np.any(np.asarray(t) <= 0):
        raise ValueError("times must be positive")
    return conversion_integral(alpha, z, geometry) * radius**2 / (2 * np.asarray(t, dtype=float))
