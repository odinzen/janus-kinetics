"""Phases with interstitial elements: a substitutional sublattice plus an interstitial one with vacancies.

Andersson and Agren, J. Appl. Phys. 72 (1992) 1350. Substitutional elements share
one partial molar volume V_S and interstitials have none (eqs. 4-5), so the natural
composition variable is the u-fraction u_k = x_k / sum_{j in S} x_j (eq. 9): the site
fraction on the substitutional sublattice, and (a^I/a^S) y_k^I for an interstitial
on a sublattice with a^I sites per a^S substitutional ones (eq. 10). With
c_k = u_k / V_S (eq. 8) the lattice-fixed fluxes are

    J'_k = -c_k Omega_k grad mu_k           (substitutional, eq. 62)
    J'_k = -c_k y_Va Omega_k grad mu_k      (interstitial, eq. 63; y_Va the vacant interstitial fraction)

and in the frame fixed with respect to the substitutional elements the
diffusivities are eq. (64), reduced to the independent u-fractions by eq. (65):

    D_kj = sum_{i in S} (delta_ik - u_k) u_i Omega_i dmu_i/du_j + [k interstitial] u_k y_Va Omega_k dmu_k/du_j

The independent variables are every interstitial u and all substitutional u but
the dependent one. Fluxes then follow J_k = -sum_j D_kj du_j/dz / V_S, so the
planar solvers apply unchanged to u-fraction profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import symengine as se
from pycalphad import Database, Model, variables as v
from tinydb import where

from .errors import MissingParameterError, UnstableCompositionError, UnsupportedPhaseError
from .mobility import MOBILITY_TYPES, sublattice_expansion
from .schedule import Temperature, temperature_at
from .thermo import ONE_ATMOSPHERE, R

COMPOSITION_FLOOR = 1e-12


@dataclass(frozen=True)
class InterstitialLattice:
    """One substitutional sublattice and one interstitial sublattice with vacancies; others vacancy-only."""

    phase_name: str
    substitutional: tuple[str, ...]  # dependent element last
    interstitial: tuple[str, ...]
    s_index: int
    i_index: int
    s_ratio: float
    i_ratio: float
    vacancy_sublattices: tuple[int, ...]

    @property
    def components(self) -> tuple[str, ...]:
        return self.substitutional + self.interstitial

    @property
    def independent(self) -> tuple[str, ...]:
        return self.substitutional[:-1] + self.interstitial

    @property
    def ratio(self) -> float:
        """a^I / a^S, the largest possible interstitial u-fraction."""
        return self.i_ratio / self.s_ratio

    @classmethod
    def from_database(cls, db: Database, phase_name: str, substitutional, interstitial, dependent=None):
        phase_name = phase_name.upper()
        if phase_name not in db.phases:
            raise UnsupportedPhaseError(f"phase {phase_name} is not in the database")
        subs = [el.upper() for el in substitutional]
        ints = [el.upper() for el in interstitial]
        if not subs or not ints or len(set(subs + ints)) != len(subs + ints) or "VA" in subs + ints:
            raise UnsupportedPhaseError("need distinct substitutional and interstitial elements (VA is implied)")
        dependent = subs[-1] if dependent is None else dependent.upper()
        if dependent not in subs:
            raise UnsupportedPhaseError(f"dependent element {dependent} must be substitutional")
        subs = [el for el in subs if el != dependent] + [dependent]
        phase = db.phases[phase_name]
        s_set, i_set, va = {v.Species(e) for e in subs}, {v.Species(e) for e in ints}, v.Species("VA")
        s_index = i_index = None
        vacancy = []
        for idx, subl in enumerate(phase.constituents):
            subl = set(subl)
            if s_set <= subl and not (i_set & subl):
                if va in subl:
                    raise UnsupportedPhaseError(f"{phase_name}: vacancies on the substitutional sublattice are not supported")
                s_index = idx
            elif i_set <= subl and va in subl and not (s_set & subl):
                i_index = idx
            elif subl & (s_set | i_set) == set() and va in subl:
                vacancy.append(idx)
            else:
                raise UnsupportedPhaseError(
                    f"{phase_name}: sublattice {idx} holds {sorted(str(x) for x in subl)}; expected one sublattice "
                    f"with {subs}, one with {ints} and VA, and vacancy-only sublattices")
        if s_index is None or i_index is None:
            raise UnsupportedPhaseError(f"{phase_name}: no substitutional sublattice with {subs} and interstitial "
                                        f"sublattice with {ints} + VA")
        return cls(phase_name, tuple(subs), tuple(ints), s_index, i_index, float(phase.sublattices[s_index]),
                   float(phase.sublattices[i_index]), tuple(vacancy))


class InterstitialPhase:
    """Thermodynamics, mobilities and diffusivities of a phase with interstitial elements, in u-fractions.

    Composition arguments are u-fractions in the order of :attr:`independent`
    (substitutional elements except the dependent one, then the interstitials), or
    all of them in the order of :attr:`components`.
    """

    def __init__(self, db: Database, phase_name: str, substitutional, interstitial, dependent=None,
                 mobility_db: Database | None = None, pressure: float = ONE_ATMOSPHERE):
        lat = InterstitialLattice.from_database(db, phase_name, substitutional, interstitial, dependent)
        self.lattice = lat
        comps = lat.components
        self._u = [se.Symbol(f"U_{c}") for c in comps]
        u = dict(zip(comps, self._u))
        y_int = {c: u[c] / lat.ratio for c in lat.interstitial}
        y_va = 1 - sum(y_int.values())
        self._y_va = y_va
        subs = {v.Y(lat.phase_name, lat.s_index, c): u[c] for c in lat.substitutional}
        subs.update({v.Y(lat.phase_name, lat.i_index, c): y for c, y in y_int.items()})
        subs[v.Y(lat.phase_name, lat.i_index, "VA")] = y_va
        subs.update({v.Y(lat.phase_name, idx, "VA"): se.Integer(1) for idx in lat.vacancy_sublattices})

        model = Model(db, list(comps) + ["VA"], lat.phase_name)
        # GM is per mole of atoms; per mole of substitutional atoms multiply by atoms per formula
        # unit (a^S + a^I sum y_int) and divide by a^S
        atoms = lat.s_ratio + lat.i_ratio * sum(y_int.values())
        gu = (model.GM.xreplace(subs) * atoms / lat.s_ratio).xreplace({v.P: se.Float(pressure)})
        stray = gu.free_symbols - set(self._u) - {v.T}
        if stray:
            raise ValueError(f"unresolved symbols in GM of {lat.phase_name}: {sorted(map(str, stray))}")
        grad = [gu.diff(s) for s in self._u]
        args = self._u + [v.T]
        self._g = se.Lambdify(args, [gu] + grad, cse=True)
        n = len(comps)
        self._hess = se.Lambdify(args, [grad[i].diff(self._u[j]) for i in range(n) for j in range(n)], cse=True)

        mob_db = mobility_db or db
        self._mobility = se.Lambdify(args, [self._mobility_expr(mob_db, c, u, y_int, y_va) for c in comps], cse=True)
        self._yva = se.Lambdify(args, [y_va], cse=True)

    # --- composition -----------------------------------------------------------------------
    @property
    def components(self) -> tuple[str, ...]:
        return self.lattice.components

    @property
    def independent(self) -> tuple[str, ...]:
        return self.lattice.independent

    def full_composition(self, u) -> np.ndarray:
        """All u-fractions (..., n) from the independent ones (..., n-1), or checked if already full."""
        lat = self.lattice
        u = np.asarray(u, dtype=float)
        u = u[None] if u.ndim == 0 else u
        n_s = len(lat.substitutional)
        if u.shape[-1] == len(lat.independent):
            s_part = u[..., : n_s - 1]
            full = np.concatenate([s_part, 1.0 - s_part.sum(axis=-1, keepdims=True), u[..., n_s - 1:]], axis=-1)
        elif u.shape[-1] == len(lat.components):
            full = u
            if not np.allclose(full[..., :n_s].sum(axis=-1), 1.0, atol=1e-10):
                raise ValueError("substitutional u-fractions must sum to one")
        else:
            raise ValueError(f"last axis must have length {len(lat.independent)} or {len(lat.components)}")
        s, i = full[..., :n_s], full[..., n_s:]
        if s.min() < -1e-6 or s.max() > 1 + 1e-6 or i.min() < -1e-6 or i.sum(axis=-1).max() > lat.ratio + 1e-6:
            raise ValueError(f"u-fractions out of range: substitutional must lie in [0, 1] and the interstitials "
                             f"sum to at most a^I/a^S = {lat.ratio:g}")
        floored = np.clip(full, COMPOSITION_FLOOR, None)
        floored[..., :n_s] /= floored[..., :n_s].sum(axis=-1, keepdims=True)
        return floored

    def u_from_x(self, x) -> np.ndarray:
        """Independent u-fractions from mole fractions given in the order of :attr:`components`."""
        x = np.asarray(x, dtype=float)
        n_s = len(self.lattice.substitutional)
        u = x / x[..., :n_s].sum(axis=-1, keepdims=True)
        return np.concatenate([u[..., : n_s - 1], u[..., n_s:]], axis=-1)

    # --- thermodynamics ----------------------------------------------------------------------
    def _args(self, u, T):
        T = np.broadcast_to(np.asarray(T, dtype=float), u.shape[:-1])
        return np.concatenate([u, T[..., None]], axis=-1)

    def _gibbs(self, u, T):
        out = self._g(self._args(u, T))
        return out[..., 0], out[..., 1:]

    def _full_hessian(self, u, T):
        n = u.shape[-1]
        return self._hess(self._args(u, T)).reshape(u.shape[:-1] + (n, n))

    def chemical_potentials(self, u, T) -> np.ndarray:
        """mu_k in J/mol for every component, shape (..., n).

        With G_u the Gibbs energy per mole of substitutional atoms as a function of
        all u, mu_k = dG_u/du_k for an interstitial (adding one leaves the number of
        substitutional atoms unchanged) and mu_k = G_u + dG_u/du_k - sum_j u_j dG_u/du_j
        for a substitutional element.
        """
        u = self.full_composition(u)
        g, grad = self._gibbs(u, T)
        n_s = len(self.lattice.substitutional)
        mu = grad.copy()
        mu[..., :n_s] = g[..., None] + grad[..., :n_s] - (u * grad).sum(axis=-1, keepdims=True)
        return mu

    def _projection(self) -> np.ndarray:
        """Columns map an independent change to the full u vector (the dependent absorbs substitutional changes)."""
        lat = self.lattice
        n, n_s = len(lat.components), len(lat.substitutional)
        t = np.zeros((n, n - 1))
        for col in range(n - 1):
            if col < n_s - 1:
                t[col, col], t[n_s - 1, col] = 1.0, -1.0
            else:
                t[col + 1, col] = 1.0
        return t

    def dmu_du(self, u, T, check: bool = True) -> np.ndarray:
        """dmu_k/du_j for every component k and independent j, shape (..., n, n-1).

        Raises UnstableCompositionError where the curvature of G_u over the
        independent u-fractions is not positive definite.
        """
        u = self.full_composition(u)
        h = self._full_hessian(u, T)
        n_s = len(self.lattice.substitutional)
        full = h.copy()
        full[..., :n_s, :] = h[..., :n_s, :] - np.einsum("...l,...lj->...j", u, h)[..., None, :]
        t = self._projection()
        if check:
            curvature = np.swapaxes(t, 0, 1) @ h @ t
            eig = np.linalg.eigvalsh(curvature)
            bad = eig.min(axis=-1) <= 0
            if np.any(bad):
                raise UnstableCompositionError(np.atleast_2d(u)[np.atleast_1d(bad)], np.atleast_2d(eig)[np.atleast_1d(bad)],
                                               self.components)
        return full @ t

    # --- kinetics ----------------------------------------------------------------------------
    def _mobility_expr(self, db, element, u, y_int, y_va):
        lat = self.lattice
        if lat.phase_name not in db.phases:
            raise MissingParameterError(f"the mobility database has no phase {lat.phase_name}")
        active = {v.Species(c) for c in lat.components} | {v.Species("VA"), v.Species("*")}
        params = db.search((where("phase_name") == lat.phase_name)
                           & (where("parameter_type").one_of(MOBILITY_TYPES))
                           & (where("diffusing_species") == v.Species(element))
                           & (where("constituent_array").test(lambda arr: all(set(s) <= active for s in arr))))
        for host in lat.substitutional:
            ok = any(p["constituent_array"][lat.s_index] in ((v.Species(host),), (v.Species("*"),))
                     and p["constituent_array"][lat.i_index] in ((v.Species("VA"),), (v.Species("*"),)) for p in params)
            if not ok:
                raise MissingParameterError(f"no MQ or MF parameter for {element} diffusing in pure {host} with empty "
                                            f"interstitial sites ({lat.phase_name})")
        y_of = {lat.s_index: {c: u[c] for c in lat.substitutional},
                lat.i_index: {**y_int, "VA": y_va}}
        y_of.update({idx: {"VA": se.Integer(1)} for idx in lat.vacancy_sublattices})
        return sublattice_expansion(params, y_of)

    def mobility(self, u, T) -> np.ndarray:
        """Omega_k = exp((MF + MQ)/RT)/RT for every component, shape (..., n), m^2 mol/(J s)."""
        u = self.full_composition(u)
        rt = R * np.asarray(T, dtype=float)[..., None]
        return np.exp(self._mobility(self._args(u, T)) / rt) / rt

    def interdiffusion_matrix(self, u, T) -> np.ndarray:
        """Eq. (64)-(65) of Andersson and Agren in the independent u-fractions, shape (..., n-1, n-1), m^2/s."""
        full = self.full_composition(u)
        lat = self.lattice
        n, n_s = len(lat.components), len(lat.substitutional)
        omega = self.mobility(full, T)
        y_va = self._yva(self._args(full, T))[..., 0]
        weight = full * omega
        weight[..., n_s:] *= y_va[..., None]  # interstitials jump only into vacant sites (eq. 63)
        dmu = self.dmu_du(full, T)
        rows = [k for k in range(n) if k != n_s - 1]
        p = np.zeros(full.shape[:-1] + (n - 1, n))
        for r, k in enumerate(rows):
            p[..., r, :n_s] = -full[..., k, None]
            if k < n_s:
                p[..., r, k] += 1.0
            else:
                p[..., r, k] = 1.0
        return p @ (weight[..., :, None] * dmu)

    def interdiffusion_function(self, temperature: Temperature):
        """D as a function of (u, t) for the solvers, at a constant T or along a schedule."""
        return lambda u, t: self.interdiffusion_matrix(u, temperature_at(temperature, t))
