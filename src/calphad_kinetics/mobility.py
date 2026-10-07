"""Composition-dependent atomic mobilities read from MQ and MF database parameters.

The Thermo-Calc style mobility database stores, for each diffusing element B,

* ``MQ(PHASE&B, ...)``, the activation term (usually written as -Q_B, often with
  R*T*LN(M0) folded in), and
* ``MF(PHASE&B, ...)``, the frequency-factor term R*T*LN(M0_B), optional.

Both are expanded over composition like Gibbs energy parameters, and pycalphad
already parses them into its Database, with the constituents of each parameter
sorted alphabetically. The expansion is written out here per diffusing element.
pycalphad's own Redlich-Kister routine is not used for it, because its check for
higher-order ternary terms does not filter by diffusing species, so one element's
ternary parameters would change another element's mobility.
"""

from __future__ import annotations

import numpy as np
import symengine as se
from collections import defaultdict

from pycalphad import Database, Model, variables as v
from tinydb import where

from ._lattice import SubstitutionalLattice
from .errors import MissingParameterError, UnsupportedPhaseError
from .thermo import R

MOBILITY_TYPES = ("MF", "MQ")  # order of the compiled outputs


class MobilityModel:
    """Andersson-Agren atomic mobilities M_B(x, T) for one substitutional phase.

    Phi_B = sum_i x_i Phi_B^i + sum_{i<j} x_i x_j sum_r Phi_B^{ij,r} (x_i - x_j)^r
    + ternary terms, the expansion of eq. (67) in Andersson and Agren, J. Appl.
    Phys. 72 (1992) 1350. The ternary term with v_ijk^s = x_s + (1 - x_i - x_j - x_k)/3,
    the split Phi_B = RT Theta_B - Q_B and M_B = exp(Phi_B / RT) / RT are eqs.
    (3)-(6) of Jonsson, ISIJ Int. 35 (1995) 1415. Andersson and Agren's eq. (66)
    writes the mobility without the 1/RT; the MQ/MF database convention follows
    Jonsson.

    Every diffusing element must have an end-member parameter (MQ or MF) in every
    pure element of the system, otherwise Phi_B would silently fall to zero near
    that corner and the mobility would be meaningless.
    """

    def __init__(self, db: Database, lattice: SubstitutionalLattice):
        self.lattice = lattice
        if lattice.phase_name not in db.phases:
            raise MissingParameterError(f"the mobility database has no phase {lattice.phase_name}")
        n_subl = len(db.phases[lattice.phase_name].constituents)
        if n_subl != 1 + len(lattice.vacancy_sublattices):
            raise MissingParameterError(
                f"{lattice.phase_name} has {n_subl} sublattices in the mobility database but "
                f"{1 + len(lattice.vacancy_sublattices)} in the thermodynamic one; they must be declared alike")
        active = {v.Species(s) for s in lattice.pycalphad_components} | {v.Species("*")}
        functions = {se.Symbol(name): value for name, value in db.symbols.items()}

        def valid(array) -> bool:
            return all(set(subl) <= active for subl in array)

        exprs = []
        for ptype in MOBILITY_TYPES:
            for el in lattice.elements:
                self._require_end_members(db, el)
                params = db.search(
                    (where("phase_name") == lattice.phase_name)
                    & (where("parameter_type") == ptype)
                    & (where("diffusing_species") == v.Species(el))
                    & (where("constituent_array").test(valid))
                )
                exprs.append(Model.symbol_replace(self._expansion(params), functions))
        args = list(lattice.x_symbols) + [v.T]
        stray = {s for e in exprs for s in e.free_symbols} - set(args)
        if stray:
            raise ValueError(f"unresolved symbols in mobility parameters: {sorted(map(str, stray))}")
        self._f = se.Lambdify(args, exprs, cse=True)

    def _expansion(self, params) -> se.Basic:
        """Phi for one diffusing element from its own parameters only.

        End members x_i Phi^i, binary terms x_i x_j Phi^{ij,r} (x_i - x_j)^r with i, j
        in pycalphad's sorted order, and ternary terms x_i x_j x_k sum_s v_s Phi^{ijk,s}
        with v_s = x_s + (1 - x_i - x_j - x_k)/3 (Jonsson, ISIJ Int. 35 (1995) 1415,
        eqs. 5-6); a ternary given at order 0 only is the symmetric term x_i x_j x_k Phi^0.
        """
        lat = self.lattice
        x = dict(zip(lat.elements, lat.x_symbols))
        total = se.Integer(0)
        ternary = defaultdict(dict)
        for p in params:
            names = [str(s.name) for s in p["constituent_array"][lat.mixing_sublattice]]
            order, value = int(p["parameter_order"]), p["parameter"]
            if names == ["*"]:
                # the wildcard applies whatever the constituent; the x_i sum to one
                total += value
            elif len(names) == 1:
                total += x[names[0]] * value
            elif len(names) == 2:
                i, j = names
                total += x[i] * x[j] * value * (x[i] - x[j]) ** order
            elif len(names) == 3:
                ternary[tuple(names)][order] = value
            else:
                raise UnsupportedPhaseError(f"mobility interactions among {len(names)} elements are not supported: {names}")
        for names, orders in ternary.items():
            product = x[names[0]] * x[names[1]] * x[names[2]]
            if set(orders) == {0}:
                total += product * orders[0]
                continue
            rest = (1 - x[names[0]] - x[names[1]] - x[names[2]]) / 3
            for s, value in orders.items():
                total += product * (x[names[s]] + rest) * value
        return total

    def _require_end_members(self, db: Database, element: str) -> None:
        lat = self.lattice
        for host in lat.elements:
            array = [None] * (1 + len(lat.vacancy_sublattices))
            array[lat.mixing_sublattice] = (v.Species(host),)
            for idx in lat.vacancy_sublattices:
                array[idx] = (v.Species("VA"),)
            wildcard = list(array)
            wildcard[lat.mixing_sublattice] = (v.Species("*"),)
            found = db.search(
                (where("phase_name") == lat.phase_name)
                & (where("parameter_type").one_of(MOBILITY_TYPES))
                & (where("diffusing_species") == v.Species(element))
                & ((where("constituent_array") == tuple(array)) | (where("constituent_array") == tuple(wildcard)))
            )
            if not found:
                raise MissingParameterError(
                    f"no MQ or MF end-member parameter for {element} diffusing in pure {host} "
                    f"({lat.phase_name}); the mobility near that corner would be undefined"
                )

    def parameters(self, x: np.ndarray, T) -> tuple[np.ndarray, np.ndarray]:
        """Return (MF, MQ) sums in J/mol for each diffusing element, each shaped (..., n)."""
        x = self.lattice.full_composition(x)
        T = np.broadcast_to(np.asarray(T, dtype=float), x.shape[:-1])
        out = self._f(np.concatenate([x, T[..., None]], axis=-1))
        n = self.lattice.n
        return out[..., :n], out[..., n:]

    def phi(self, x: np.ndarray, T) -> np.ndarray:
        """Phi_B = MF_B + MQ_B = RT ln(M0_B) - Q_B in J/mol, shape (..., n)."""
        mf, mq = self.parameters(x, T)
        return mf + mq

    def mobility(self, x: np.ndarray, T, magnetic_factor: np.ndarray | float = 1.0) -> np.ndarray:
        """Atomic mobility M_B in m^2 mol / (J s), shape (..., n).

        ``magnetic_factor`` is the multiplier mg-Omega of eq. (3) in Jonsson,
        ISIJ Int. 35 (1995) 1415, which accounts for ferromagnetic ordering. It is
        1 in the paramagnetic state and is supplied by the caller; see ROADMAP.md.
        """
        rt = R * np.asarray(T, dtype=float)[..., None]
        return np.exp(self.phi(x, T) / rt) / rt * magnetic_factor

    def tracer_diffusivity(self, x: np.ndarray, T, magnetic_factor: np.ndarray | float = 1.0) -> np.ndarray:
        """D*_B = RT M_B in m^2/s, neglecting the correlation factor (Jonsson 1995, eq. 1)."""
        rt = R * np.asarray(T, dtype=float)[..., None]
        return rt * self.mobility(x, T, magnetic_factor)
