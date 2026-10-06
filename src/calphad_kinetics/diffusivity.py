"""Reduced interdiffusion matrices D = P Lambda Phi.

All composition arrays carry the dependent element last. ``dmu`` is
dmu_i/dx_j with x_n dependent, shape (..., n, n-1), as returned by
:func:`calphad_kinetics.thermo.dmu_dx`.
"""

from __future__ import annotations

import numpy as np


def number_fixed_matrix(x: np.ndarray, mobility: np.ndarray, dmu: np.ndarray) -> np.ndarray:
    """Interdiffusion matrix in the number-fixed frame, shape (..., n-1, n-1), m^2/s.

    D_kj = sum_i (delta_ik - x_k) x_i M_i dmu_i/dx_j, the vacancy-mechanism result
    of Andersson and Agren, J. Appl. Phys. 72 (1992) 1350, quoted as eq. (2) of
    Jonsson, ISIJ Int. 35 (1995) 1415. Fluxes follow J_k = -sum_j D_kj dc_j/dz
    with c_j = x_j / V_m and a constant molar volume.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    p = np.eye(n)[:-1, :] - x[..., :-1, None]
    return p @ ((x * mobility)[..., :, None] * dmu)


def volume_fixed_matrix(
    x: np.ndarray, mobility: np.ndarray, dmu: np.ndarray, partial_volumes: np.ndarray
) -> np.ndarray:
    """Interdiffusion matrix in the volume-fixed frame, shape (..., n-1, n-1), m^2/s.

    Andersson and Agren, J. Appl. Phys. 72 (1992) 1350. With constant partial
    molar volumes V_i, c_i = x_i / V_m and sum_i V_i c_i = 1, so c_n is dependent:

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
