"""Map a pycalphad substitutional phase onto plain mole-fraction symbols."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import symengine as se
from pycalphad import Database, variables as v

from .errors import UnsupportedPhaseError


@dataclass(frozen=True)
class SubstitutionalLattice:
    """One mixing sublattice holding every element, other sublattices vacancy-only.

    ``elements`` keeps the order the caller gave, with the dependent (solvent)
    element moved last; every composition array in the package uses this order.
    """

    phase_name: str
    elements: tuple[str, ...]
    mixing_sublattice: int
    vacancy_sublattices: tuple[int, ...]
    x_symbols: tuple[se.Symbol, ...]

    @classmethod
    def from_database(
        cls, db: Database, phase_name: str, elements: list[str] | tuple[str, ...], dependent: str | None = None
    ) -> SubstitutionalLattice:
        phase_name = phase_name.upper()
        if phase_name not in db.phases:
            raise UnsupportedPhaseError(f"phase {phase_name} is not in the database")
        # Keep the caller's order so composition arrays mean what the caller wrote;
        # only the dependent element moves to the end.
        names = [el.upper() for el in elements if el.upper() != "VA"]
        if len(set(names)) != len(names):
            raise UnsupportedPhaseError(f"element listed twice in {list(elements)}")
        if len(names) < 2:
            raise UnsupportedPhaseError("at least two elements are needed for diffusion")
        dependent = names[-1] if dependent is None else dependent.upper()
        if dependent not in names:
            raise UnsupportedPhaseError(f"dependent element {dependent} is not among {names}")
        ordered = tuple([el for el in names if el != dependent] + [dependent])

        wanted = {v.Species(el) for el in ordered}
        va = v.Species("VA")
        mixing, vacancy = None, []
        for idx, subl in enumerate(db.phases[phase_name].constituents):
            active = set(subl) & (wanted | {va})
            if wanted <= active:
                if mixing is not None:
                    raise UnsupportedPhaseError(f"{phase_name}: elements appear on more than one sublattice")
                mixing = idx
                if va in active:
                    raise UnsupportedPhaseError(f"{phase_name}: vacancies on the substitutional sublattice are not supported")
            elif active == {va}:
                vacancy.append(idx)
            else:
                others = sorted(str(s) for s in subl)
                raise UnsupportedPhaseError(
                    f"{phase_name}: sublattice {idx} holds {others} and of the requested elements "
                    f"{sorted(str(s) for s in active)}. This version handles one substitutional sublattice "
                    "holding every element plus vacancy-only sublattices; interstitial elements (C, N, O, H "
                    "on a second sublattice), ionic species and ordered multi-sublattice phases are not yet supported"
                )
        if mixing is None:
            raise UnsupportedPhaseError(f"{phase_name}: no sublattice contains all of {list(ordered)}")
        symbols = tuple(se.Symbol(f"X_{el}") for el in ordered)
        return cls(phase_name, ordered, mixing, tuple(vacancy), symbols)

    @property
    def n(self) -> int:
        return len(self.elements)

    @property
    def pycalphad_components(self) -> list[str]:
        return list(self.elements) + (["VA"] if self.vacancy_sublattices else [])

    def site_fraction_map(self) -> dict:
        """Replacement dict taking pycalphad site fractions to mole fraction symbols."""
        subs = {v.Y(self.phase_name, self.mixing_sublattice, el): x for el, x in zip(self.elements, self.x_symbols)}
        subs.update({v.Y(self.phase_name, idx, "VA"): se.Integer(1) for idx in self.vacancy_sublattices})
        return subs

    def full_composition(self, x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Accept (..., n-1) independent or (..., n) full mole fractions; return (..., n).

        A scalar is taken as the single independent fraction of a binary. Fractions
        outside [0, 1] by more than ``tol`` raise, since clipping them would
        silently evaluate a different alloy.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[None]
        if x.shape[-1] == self.n - 1:
            full = np.concatenate([x, 1.0 - x.sum(axis=-1, keepdims=True)], axis=-1)
        elif x.shape[-1] == self.n:
            if not np.allclose(x.sum(axis=-1), 1.0, atol=1e-10):
                raise ValueError("full mole fractions must sum to one")
            full = x
        else:
            raise ValueError(f"last axis must have length {self.n - 1} or {self.n}, got {x.shape[-1]}")
        lo, hi = full.min(), full.max()
        if lo < -tol or hi > 1 + tol:
            raise ValueError(
                f"mole fractions of {self.elements} must lie in [0, 1]; got values from {lo:.4g} to {hi:.4g} "
                "(percent instead of fractions, independent fractions summing above one, or an oscillating "
                "profile that needs a finer grid or a shorter step)")
        return full
