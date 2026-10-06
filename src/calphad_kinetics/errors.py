"""Exceptions raised by calphad-kinetics."""

from __future__ import annotations

import numpy as np


class UnstableCompositionError(ValueError):
    """The Gibbs energy curvature is not positive definite at some composition.

    Such a composition lies inside a spinodal, which sits inside a miscibility
    gap or two-phase field. A single-phase diffusion model has no physical
    meaning there, so the calculation stops instead of patching the curvature.
    Model the problem with a multiphase (moving-boundary or dispersed-phase)
    model instead.
    """

    def __init__(self, compositions: np.ndarray, eigenvalues: np.ndarray, elements: tuple[str, ...]):
        self.compositions = np.atleast_2d(compositions)
        self.eigenvalues = np.atleast_2d(eigenvalues)
        self.elements = elements
        first = ", ".join(f"{el}={val:.6g}" for el, val in zip(elements, self.compositions[0]))
        super().__init__(
            f"Thermodynamic factor is not positive definite at {len(self.compositions)} "
            f"composition(s), first at ({first}), smallest curvature eigenvalue "
            f"{self.eigenvalues[0].min():.6g} J/mol. The composition is inside a "
            "miscibility gap or two-phase field; use a multiphase model instead."
        )


class MissingParameterError(KeyError):
    """A mobility parameter needed for a well-defined model is absent from the database."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class UnsupportedPhaseError(ValueError):
    """The phase constitution is outside what this version can model."""


class ConvergenceError(RuntimeError):
    """The nonlinear (Picard) iteration of a time step did not converge."""
