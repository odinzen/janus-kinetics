"""Coherent interfacial energy from broken bonds, after Sonderegger and Kozeschnik, Metall. Mater. Trans. A 40
(2009) 499 (the generalized broken-bond model).

Bonds to every neighbour shell k count with the weight f(r_k / r_1) = (r_k / r_1)^-m (eq. 4). Summing over the
shells gives the effective coordination z_L,eff (eq. 7) and the effective number of bonds cut by the interface
per interface atom z_S,eff (eqs. 8, 11), and the interfacial energy follows from the enthalpy of solution:

    gamma = n_S z_S,eff / (N z_L,eff) Delta E_sol,   n_S = n_V^(2/3)          (eqs. 9, 10)

with n_V atoms per volume and N Avogadro's number. Shells beyond the k-th are smeared into a continuum, which adds
4 pi n_V r_1^m r_k^(3-m) / (m - 3) to z_L,eff and pi n_V^(4/3) r_1^m r_k^(4-m) / (m - 4) to z_S,eff (Appendix II,
eqs. 14-15 for m = 6). Both terms diverge for m <= 4, so the exponent must exceed 4.

The bonds are counted here by lattice sums, not with the paper's periodic volume elements. In a Bravais lattice
every lattice vector v projects onto the interface normal n as a whole number of interplanar spacings, so the bonds
of type v cut per unit area are exactly n_V |v . n| / 2 (counting v and -v once) and

    z_S,eff = n_V^(1/3) / 2 sum_v |v . n| f(|v| / r_1),

which is the paper's count (it reproduces Table II to 0.1 %). Averaged over orientations, |v . n| becomes |v| / 2.
The structure enters only through fcc or bcc, both Bravais lattices.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

import numpy as np
from pycalphad import Database

from .multicomponent_precipitation import PhaseMinimizer, PrecipitateThermo

Structure = Literal["fcc", "bcc"]

N_A = 6.02214076e23  # 1/mol

# in units of half the cubic lattice parameter: (first-neighbour distance squared, atoms per unit volume)
_CELL = {"fcc": (2, 0.5), "bcc": (3, 0.25)}


@lru_cache(maxsize=8)
def _shells(structure: Structure, shells: int) -> tuple[np.ndarray, np.ndarray]:
    """Lattice vectors out to the ``shells``-th neighbour shell and the squared radii of the shells."""
    if structure not in _CELL:
        raise ValueError(f"structure must be 'fcc' or 'bcc', not {structure!r}")
    if shells < 1:
        raise ValueError("at least one neighbour shell is needed")
    n = 4
    while True:
        g = np.arange(-n, n + 1)
        p = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
        if structure == "fcc":
            p = p[p.sum(axis=1) % 2 == 0]
        else:
            p = p[(p % 2 == p[:, :1] % 2).all(axis=1)]
        r2 = (p * p).sum(axis=1)
        p, r2 = p[r2 > 0], r2[r2 > 0]
        # only shells inside the enumerated cube are complete
        radii = np.unique(r2[r2 <= n * n])
        if radii.size >= shells:
            keep = r2 <= radii[shells - 1]
            return p[keep], radii[:shells]
        n *= 2


def _check_exponent(exponent: float) -> None:
    if exponent <= 4:
        raise ValueError("the bond exponent must exceed 4; the long-range bond sums diverge otherwise")


def effective_coordination(structure: Structure, exponent: float = 6.0, shells: int = 100) -> float:
    """z_L,eff, the shell-weighted number of bonds per atom, eq. (7) with the continuum tail of eq. (31)."""
    _check_exponent(exponent)
    p, radii = _shells(structure, shells)
    r1sq, nv = _CELL[structure]
    w = ((p * p).sum(axis=1) / r1sq) ** (-exponent / 2)
    rk = np.sqrt(radii[-1])
    return float(w.sum() + 4 * np.pi * nv * r1sq ** (exponent / 2) * rk ** (3 - exponent) / (exponent - 3))


def broken_bond_factor(structure: Structure, normal: Sequence[float] | None = None, exponent: float = 6.0,
                       shells: int = 100) -> float:
    """The structural factor z_S,eff / z_L,eff of eq. (9) for an interface with Miller-index ``normal``, or its
    average over all orientations when ``normal`` is None. Sonderegger and Kozeschnik tabulate it for m = 6-8
    (Table II) and give 0.3291 (fcc) and 0.3280 (bcc) as the r^-6 mean (Table III)."""
    _check_exponent(exponent)
    p, radii = _shells(structure, shells)
    r1sq, nv = _CELL[structure]
    r = np.sqrt((p * p).sum(axis=1))
    w = (r * r / r1sq) ** (-exponent / 2)
    if normal is None:
        proj = r / 2
    else:
        n = np.asarray(normal, dtype=float)
        if n.shape != (3,) or not np.any(n):
            raise ValueError("the interface normal needs three indices, not all zero")
        proj = np.abs(p @ (n / np.linalg.norm(n)))
    rk = np.sqrt(radii[-1])
    zs = nv ** (1 / 3) / 2 * (proj * w).sum() + np.pi * nv ** (4 / 3) * r1sq ** (exponent / 2) * rk ** (4 - exponent) / (exponent - 4)
    return float(zs / effective_coordination(structure, exponent, shells))


def broken_bond_energy(solution_enthalpy: float, molar_volume: float, structure: Structure = "fcc",
                       normal: Sequence[float] | None = None, exponent: float = 6.0, factor: float | None = None) -> float:
    """Interfacial energy in J/m^2, eqs. (9)-(10), from the enthalpy of solution of the precipitate in the matrix
    (J per mole of atoms, positive when dissolving absorbs heat) and the matrix molar volume (m^3/mol). The
    structural factor is :func:`broken_bond_factor` unless ``factor`` is given."""
    if molar_volume <= 0:
        raise ValueError("the molar volume must be positive")
    if factor is None:
        factor = broken_bond_factor(structure, normal, exponent)
    n_s = (N_A / molar_volume) ** (2 / 3)
    return float(n_s * factor * solution_enthalpy / N_A)


def solution_enthalpy(db: Database, components: Sequence[str], matrix_phase: str, matrix_x: Sequence[float],
                      precipitate_phase: str, precipitate_x: Sequence[float], T: float, dT: float = 0.5) -> float:
    """Enthalpy of dissolving one mole of atoms of precipitate in an infinite matrix, J/mol: the thought
    experiment of Sonderegger and Kozeschnik's Appendix I (eqs. 19-20), sum_j x_j^P H_j^M - H^P, with H_j^M the
    partial molar enthalpies of the matrix. It is taken from the Gibbs energy of solution
    Delta G = sum_j x_j^P mu_j^M - G^P by Gibbs-Helmholtz, Delta H = -T^2 d(Delta G / T)/dT at fixed compositions.

    Each phase is held as one homogeneous state with its internal degrees of freedom (ordering) relaxed, by
    :class:`PhaseMinimizer` started from pycalphad's equilibrium at T. A phase at its solvus composition would
    otherwise split into two composition sets on one side of T, and the derivative would follow the tie-line."""
    comps = [c.upper() for c in components]
    xm, xp = np.asarray(matrix_x, dtype=float), np.asarray(precipitate_x, dtype=float)
    for name, x in (("matrix", xm), ("precipitate", xp)):
        if x.shape != (len(comps),) or np.any(x < 0) or abs(x.sum() - 1) > 1e-9:
            raise ValueError(f"the {name} composition needs one non-negative mole fraction per component, summing to 1")
    if np.any(xm[xp > 0] <= 0):
        raise ValueError("every element of the precipitate must be present in the matrix")
    if T <= dT or dT <= 0:
        raise ValueError("need 0 < dT < T")

    def homogeneous(phase, x):
        # elements a phase lacks are left out, as a sublattice cannot hold a zero site fraction in the minimiser
        present = [c for c, xi in zip(comps, x) if xi > 0]
        xs = x[x > 0]
        minimizer = PhaseMinimizer(db, phase, present)
        y0 = PrecipitateThermo(db, phase, present).site_fractions(xs, T)
        return present, lambda t: minimizer.solve(xs, y0, t)

    m_present, matrix = homogeneous(matrix_phase.upper(), xm)
    _, precipitate = homogeneous(precipitate_phase.upper(), xp)

    def g_over_t(t):
        mu = dict(zip(m_present, matrix(t)[2][0]))
        return (sum(xj * mu[c] for c, xj in zip(comps, xp) if xj > 0) - precipitate(t)[1][0]) / t

    return float(-T * T * (g_over_t(T + dT) - g_over_t(T - dT)) / (2 * dT))
