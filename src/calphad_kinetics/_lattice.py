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

    ``elements`` is ordered with the dependent (solvent) element last, which is
    the order used for every composition array in the package.
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
        names = sorted({el.upper() for el in elements} - {"VA"})
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
                raise UnsupportedPhaseError(
                    f"{phase_name}: sublattice {idx} holds {sorted(str(s) for s in active)}; this version "
                    "handles one substitutional sublattice plus vacancy-only interstitial sublattices"
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

    def full_composition(self, x: np.ndarray) -> np.ndarray:
        """Accept (..., n-1) independent or (..., n) full mole fractions; return (..., n)."""
        x = np.asarray(x, dtype=float)
        if x.shape[-1] == self.n - 1:
            return np.concatenate([x, 1.0 - x.sum(axis=-1, keepdims=True)], axis=-1)
        if x.shape[-1] == self.n:
            if not np.allclose(x.sum(axis=-1), 1.0, atol=1e-10):
                raise ValueError("full mole fractions must sum to one")
            return x
        raise ValueError(f"last axis must have length {self.n - 1} or {self.n}, got {x.shape[-1]}")
