"""Gibbs energy curvature and chemical potential derivatives from pycalphad.

The molar Gibbs energy of the phase is taken symbolically from
``pycalphad.Model.GM`` and differentiated exactly with SymEngine, so no
finite-difference step enters the thermodynamic factor.
"""

from __future__ import annotations

import numpy as np
import symengine as se
from pycalphad import Database, Model, variables as v

from ._lattice import SubstitutionalLattice
from .errors import UnstableCompositionError

R = float(v.R)  # same gas constant pycalphad uses, so TDB expressions stay consistent

ONE_ATMOSPHERE = 101325.0


class GibbsEnergy:
    """Compiled molar Gibbs energy (J per mole of atoms) and its exact Hessian."""

    def __init__(self, db: Database, lattice: SubstitutionalLattice, pressure: float = ONE_ATMOSPHERE):
        self.lattice = lattice
        model = Model(db, lattice.pycalphad_components, lattice.phase_name)
        gm = model.GM.xreplace(lattice.site_fraction_map()).xreplace({v.P: se.Float(pressure)})
        leftover = [s for s in gm.free_symbols if s not in set(lattice.x_symbols) and s != v.T]
        if leftover:
            raise ValueError(f"unresolved symbols in GM of {lattice.phase_name}: {leftover}")
        xs = list(lattice.x_symbols)
        n = len(xs)
        hess = [gm.diff(xs[i]).diff(xs[j]) for i in range(n) for j in range(i, n)]
        args = xs + [v.T]
        self._gm = se.Lambdify(args, [gm], cse=True)
        self._hess = se.Lambdify(args, hess, cse=True)
        self._upper = np.triu_indices(n)

    def _args(self, x: np.ndarray, T) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        T = np.broadcast_to(np.asarray(T, dtype=float), x.shape[:-1])
        return np.concatenate([x, T[..., None]], axis=-1)

    def gm(self, x: np.ndarray, T) -> np.ndarray:
        """Molar Gibbs energy at full mole fractions ``x`` (..., n)."""
        return self._gm(self._args(x, T))[..., 0]

    def hessian(self, x: np.ndarray, T) -> np.ndarray:
        """Unconstrained second derivatives d2GM/dx_i dx_j, shape (..., n, n).

        Only the projection onto the composition simplex has physical meaning;
        use :func:`reduced_hessian` for that.
        """
        x = np.asarray(x, dtype=float)
        n = x.shape[-1]
        upper = self._hess(self._args(x, T))
        out = np.empty(x.shape[:-1] + (n, n))
        out[..., self._upper[0], self._upper[1]] = upper
        out[..., self._upper[1], self._upper[0]] = upper
        return out


def reduced_hessian(hessian: np.ndarray) -> np.ndarray:
    """Curvature of G with x_n eliminated, G''_kj = H_kj - H_kn - H_nj + H_nn, shape (..., n-1, n-1)."""
    h = hessian
    return h[..., :-1, :-1] - h[..., :-1, -1:] - h[..., -1:, :-1] + h[..., -1:, -1:]


def check_stability(x: np.ndarray, g_reduced: np.ndarray, elements: tuple[str, ...]) -> None:
    """Raise :class:`UnstableCompositionError` where the reduced curvature is not positive definite."""
    eig = np.linalg.eigvalsh(g_reduced)
    bad = eig.min(axis=-1) <= 0.0
    if np.any(bad):
        raise UnstableCompositionError(np.asarray(x)[bad], eig[bad], elements)


def dmu_dx(x: np.ndarray, g_reduced: np.ndarray) -> np.ndarray:
    """Chemical potential derivatives dmu_i/dx_j with x_n dependent, shape (..., n, n-1).

    From mu_i = G + sum_k (delta_ik - x_k) dG/dx_k (k < n), differentiating gives
    dmu_i/dx_j = sum_k (delta_ik - x_k) G''_kj. This equals the bracket
    (dmu_i/dx_j - dmu_i/dx_n) of Andersson and Agren, J. Appl. Phys. 72 (1992)
    1350, as written in eq. (2) of Jonsson, ISIJ Int. 35 (1995) 1415.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    p = np.eye(n)[:, :-1] - x[..., :-1][..., None, :]  # p[i, k] = delta_ik - x_k
    return p @ g_reduced


def thermodynamic_factor(x: np.ndarray, dmu: np.ndarray, T) -> np.ndarray:
    """Dimensionless thermodynamic factor (x_i / RT) dmu_i/dx_j, shape (..., n, n-1).

    For an ideal solution the upper (n-1) block is the identity and the last
    row is -1, the multicomponent analogue of d ln a / d ln x = 1.
    """
    x = np.asarray(x, dtype=float)
    rt = R * np.asarray(T, dtype=float)[..., None, None]
    return x[..., :, None] * dmu / rt
