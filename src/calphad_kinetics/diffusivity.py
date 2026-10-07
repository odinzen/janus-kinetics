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
    c = x / (x * vol).sum(axis=-1)[..., None]
    p = np.eye(n)[:-1, :] - c[..., :-1, None] * vol
    lam = c * mobility
    return p @ (lam[..., :, None] * (dmu @ _dx_dc(x, vol)))


def _dx_dc(x: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """dx_k/dc_j = V_m [delta_kj - x_k (1 - V_j/V_n)] for the independent k, j."""
    n = x.shape[-1]
    vm = (x * vol).sum(axis=-1)[..., None, None]
    return vm * (np.eye(n - 1) - x[..., :-1, None] * (1.0 - vol[:-1] / vol[-1]))


def to_concentrations(x: np.ndarray, partial_volumes: np.ndarray) -> np.ndarray:
    """Independent concentrations c_k = x_k / V_m (mol/m^3) from all n mole fractions, V_m = sum_i x_i V_i."""
    x = np.asarray(x, dtype=float)
    vol = np.asarray(partial_volumes, dtype=float)
    return x[..., :-1] / (x * vol).sum(axis=-1)[..., None]


def to_mole_fractions(c: np.ndarray, partial_volumes: np.ndarray) -> np.ndarray:
    """All n mole fractions from the independent concentrations, closing sum_i V_i c_i = 1."""
    c = np.asarray(c, dtype=float)
    vol = np.asarray(partial_volumes, dtype=float)
    c_last = (1.0 - (c * vol[:-1]).sum(axis=-1)) / vol[-1]
    full = np.concatenate([c, c_last[..., None]], axis=-1)
    return full / full.sum(axis=-1)[..., None]


def volume_fixed_lattice_velocity(
    x: np.ndarray, mobility: np.ndarray, dmu: np.ndarray, partial_volumes: np.ndarray, dc_dz: np.ndarray
) -> np.ndarray:
    """Lattice (marker) velocity relative to the volume-fixed frame, m/s.

    With sum_k V_k J_k = 0 in that frame and J_k = J'_k + c_k v (van Loo, Prog.
    Solid State Chem. 20 (1990) 47, eqs. 4 and 6), v = -sum_k V_k J'_k, where the
    lattice flux is J'_k = -c_k M_k dmu_k/dz. In a binary this is van Loo's eq. 9,
    v = V_A (D_A - D_B) dc_A/dz. ``dc_dz`` holds the independent concentration
    gradients (mol/m^4), shape (..., n-1).
    """
    x = np.asarray(x, dtype=float)
    vol = np.asarray(partial_volumes, dtype=float)
    c = x / (x * vol).sum(axis=-1)[..., None]
    dmu_dz = np.einsum("...kj,...j->...k", dmu @ _dx_dc(x, vol), np.asarray(dc_dz, dtype=float))
    return (vol * c * mobility * dmu_dz).sum(axis=-1)
