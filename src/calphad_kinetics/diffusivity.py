"""Reduced interdiffusion matrices D = P Lambda Phi.

All composition arrays carry the dependent element last. ``dmu`` is
dmu_i/dx_j with x_n dependent, shape (..., n, n-1), as returned by
:func:`calphad_kinetics.thermo.dmu_dx`.
"""

from __future__ import annotations

import numpy as np


def intrinsic_matrix(x: np.ndarray, mobility: np.ndarray, dmu: np.ndarray) -> np.ndarray:
    """Intrinsic (lattice-frame) diffusivities D^I_kj = x_k M_k dmu_k/dx_j, shape (..., n, n-1), m^2/s.

    The lattice-fixed flux of the vacancy mechanism is J'_k = -c_k M_k grad mu_k
    (Andersson and Agren, J. Appl. Phys. 72 (1992) 1350, eqs. (58) and (62)). At
    constant molar volume this is J'_k = -sum_j D^I_kj dc_j/dz. In a binary the two
    rows are Darken's intrinsic diffusivities D_A and -D_B (van Loo, Prog. Solid
    State Chem. 20 (1990) 47, eqs. (9)-(10)).
    """
    x = np.asarray(x, dtype=float)
    return (x * mobility)[..., :, None] * dmu


def number_fixed_matrix(x: np.ndarray, mobility: np.ndarray, dmu: np.ndarray) -> np.ndarray:
    """Interdiffusion matrix in the number-fixed frame, shape (..., n-1, n-1), m^2/s.

    D_kj = sum_i (delta_ik - x_k) x_i M_i dmu_i/dx_j, the vacancy-mechanism result
    of Andersson and Agren, J. Appl. Phys. 72 (1992) 1350, eqs. (59)-(60), quoted
    as eq. (2) of Jonsson, ISIJ Int. 35 (1995) 1415. Fluxes follow J_k = -sum_j D_kj dc_j/dz
    with c_j = x_j / V_m and a constant molar volume.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    p = np.eye(n)[:-1, :] - x[..., :-1, None]
    return p @ intrinsic_matrix(x, mobility, dmu)


def lattice_velocity(intrinsic: np.ndarray, dx_dz: np.ndarray) -> np.ndarray:
    """Velocity of the lattice (inert markers) relative to the number-fixed frame, m/s.

    The two frames are related by J_k = J'_k + c_k v (van Loo, Prog. Solid State
    Chem. 20 (1990) 47, eq. (4)); summing over k with sum_k J_k = 0 gives
    v = -V_m sum_k J'_k = sum_k sum_j D^I_kj dx_j/dz. For a binary this is
    v = (D_A - D_B) dx_A/dz, van Loo's eq. (9) at equal partial volumes.
    ``dx_dz`` holds the independent gradients, shape (..., n-1).
    """
    return np.einsum("...kj,...j->...", intrinsic, dx_dz)


def volume_fixed_matrix(
    x: np.ndarray, mobility: np.ndarray, dmu: np.ndarray, partial_volumes: np.ndarray
) -> np.ndarray:
    """Interdiffusion matrix in the volume-fixed frame, shape (..., n-1, n-1), m^2/s.

    Andersson and Agren, J. Appl. Phys. 72 (1992) 1350, eqs. (28)-(29), (40),
    (50), (54) and (58), kept with general partial volumes rather than their later
    equal-volume approximation. With constant partial molar volumes V_i, c_i = x_i / V_m and
    sum_i V_i c_i = 1, so c_n is dependent:

        D^V_kj = sum_i (delta_ik - c_k V_i) c_i M_i dmu_i/dc_j

    where dmu_i/dc_j follows from dx_k/dc_j = V_m [delta_kj - x_k (1 - V_j/V_n)].
    The fluxes then satisfy sum_k V_k J_k = 0 and J_k = -sum_j D^V_kj dc_j/dz.
    Equal partial volumes recover :func:`number_fixed_matrix`.
    """
    x = np.asarray(x, dtype=float)
    vol = np.asarray(partial_volumes, dtype=float)
    n = x.shape[-1]
    vm = (x * vol).sum(axis=-1)[..., None]
    c = x / vm
    p = np.eye(n)[:-1, :] - c[..., :-1, None] * vol
    lam = c * mobility
    dxdc = vm[..., None] * (np.eye(n - 1) - x[..., :-1, None] * (1.0 - vol[:-1] / vol[-1]))
    return p @ (lam[..., :, None] * (dmu @ dxdc))
