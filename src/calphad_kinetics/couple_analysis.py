"""Diffusivities read from a measured binary diffusion-couple profile.

The inverse of the solvers: given a penetration curve N_i(z) after an anneal of
time t between two semi-infinite end members, these return the interdiffusion
coefficient (van Loo, Prog. Solid State Chem. 20 (1990) 47, eq. 3, which needs no
Matano plane and allows a molar volume that varies with composition) and, from
the position of inert markers, the intrinsic diffusivities at the Kirkendall
plane (eq. 12). With constant molar volume they reduce to eqs. 2 and 11.

Profiles must reach the undisturbed end compositions at both ends of ``z``; the
first and last points are taken as N_i^- and N_i^+. Integrals use the trapezoid
rule on the given points, so the accuracy follows the sampling of the profile.
"""

from __future__ import annotations

import numpy as np


def _profile(z, x, molar_volume):
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    if z.ndim != 1 or z.shape != x.shape or z.size < 3:
        raise ValueError("z and the profile need the same 1D shape, at least three points")
    if np.any(np.diff(z) <= 0):
        raise ValueError("z must be strictly increasing")
    if x[-1] == x[0]:
        raise ValueError("the end compositions are equal; there is no couple to analyse")
    vm = np.broadcast_to(np.asarray(1.0 if molar_volume is None else molar_volume, dtype=float), z.shape)
    if np.any(vm <= 0):
        raise ValueError("molar volumes must be positive")
    return z, x, np.array(vm)


def _cumulative(f, z):
    return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(z))])


def interdiffusion_from_profile(z, x, t: float, molar_volume=None) -> np.ndarray:
    """D~ (m^2/s) at every point of the profile ``x`` (mole fraction of one element), eq. 3.

        D~(Y*) = V_m/(2t) (dz/dY)* [ (1 - Y*) int_{-inf}^{z*} Y/V_m dz + Y* int_{z*}^{+inf} (1 - Y)/V_m dz ]

    with Y = (N - N^-)/(N^+ - N^-). ``molar_volume`` (m^3/mol) is a constant, or
    an array along z; it cancels when constant (eq. 2). Values where the profile
    is flat (the undisturbed ends) are meaningless, so read D~ inside the zone.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    z, x, vm = _profile(z, x, molar_volume)
    y = (x - x[0]) / (x[-1] - x[0])
    left = _cumulative(y / vm, z)
    right = _cumulative((1 - y) / vm, z)
    right = right[-1] - right
    dy_dz = np.gradient(y, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        return vm / (2 * t) / dy_dz * ((1 - y) * left + y * right)


def intrinsic_from_profile(z, x, t: float, z_marker: float, molar_volume=None) -> float:
    """Intrinsic diffusivity D_i (m^2/s) of the element whose mole fraction is ``x``, at the Kirkendall plane, eq. 12.

        D_i(z_K) = 1/(2t) (dz/dC_i)_K [ N_i^+ int_{-inf}^{z_K} Y/V_m dz - N_i^- int_{z_K}^{+inf} (1 - Y)/V_m dz ]

    with C_i = N_i / V_m. Call it with N_A and with 1 - N_A for the two elements of
    a binary. ``z_marker`` is the measured marker position in the same frame as z.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    z, x, vm = _profile(z, x, molar_volume)
    if not z[0] < z_marker < z[-1]:
        raise ValueError("the marker must lie inside the profile")
    y = (x - x[0]) / (x[-1] - x[0])
    vk, yk = np.interp(z_marker, z, vm), np.interp(z_marker, z, y)
    left = z < z_marker
    zl, fl = np.append(z[left], z_marker), np.append((y / vm)[left], yk / vk)
    zr, fr = np.insert(z[~left], 0, z_marker), np.insert(((1 - y) / vm)[~left], 0, (1 - yk) / vk)
    i_left, i_right = np.trapezoid(fl, zl), np.trapezoid(fr, zr)
    c = x / vm
    dc_dz = np.interp(z_marker, 0.5 * (z[1:] + z[:-1]), np.diff(c) / np.diff(z))
    return float((x[-1] * i_left - x[0] * i_right) / (2 * t * dc_dz))
