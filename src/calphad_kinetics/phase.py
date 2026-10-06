"""Single-phase diffusion data assembled from one pycalphad Database."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pycalphad import Database

from ._lattice import SubstitutionalLattice
from .diffusivity import number_fixed_matrix, volume_fixed_matrix
from .mobility import MobilityModel
from .thermo import GibbsEnergy, check_stability, dmu_dx, reduced_hessian, thermodynamic_factor

Frame = Literal["number", "volume"]

# The ideal-mixing curvature RT/x_i diverges at x_i = 0 while the product
# x_i M_i dmu_i/dx_i stays finite, so compositions are floored before evaluation.
COMPOSITION_FLOOR = 1e-12


class SolutionPhase:
    """Thermodynamics and mobilities of one substitutional solution phase.

    Composition arguments accept either the n-1 independent mole fractions or all
    n mole fractions, in the order of :attr:`elements` (dependent element last).
    """

    def __init__(
        self,
        db: Database,
        phase_name: str,
        elements: list[str] | tuple[str, ...],
        dependent: str | None = None,
        mobility_db: Database | None = None,
    ):
        self.lattice = SubstitutionalLattice.from_database(db, phase_name, elements, dependent)
        self.gibbs = GibbsEnergy(db, self.lattice)
        self.mobility_model = MobilityModel(mobility_db or db, self.lattice)

    @property
    def elements(self) -> tuple[str, ...]:
        return self.lattice.elements

    def _x(self, x) -> np.ndarray:
        x = self.lattice.full_composition(x)
        x = np.clip(x, COMPOSITION_FLOOR, None)
        return x / x.sum(axis=-1, keepdims=True)

    def reduced_hessian(self, x, T) -> np.ndarray:
        """d2G/dx_k dx_j with the dependent element eliminated, J/mol, shape (..., n-1, n-1)."""
        return reduced_hessian(self.gibbs.hessian(self._x(x), T))

    def dmu_dx(self, x, T) -> np.ndarray:
        """dmu_i/dx_j (dependent eliminated), J/mol, shape (..., n, n-1).

        Raises UnstableCompositionError if any composition is inside a spinodal.
        """
        x = self._x(x)
        g2 = reduced_hessian(self.gibbs.hessian(x, T))
        check_stability(x, g2, self.elements)
        return dmu_dx(x, g2)

    def thermodynamic_factor(self, x, T) -> np.ndarray:
        """Dimensionless (x_i/RT) dmu_i/dx_j, shape (..., n, n-1)."""
        x = self._x(x)
        return thermodynamic_factor(x, self.dmu_dx(x, T), T)

    def mobility(self, x, T, magnetic_factor=1.0) -> np.ndarray:
        return self.mobility_model.mobility(self._x(x), T, magnetic_factor)

    def interdiffusion_matrix(
        self,
        x,
        T,
        frame: Frame = "number",
        partial_volumes: np.ndarray | None = None,
        magnetic_factor=1.0,
    ) -> np.ndarray:
        """Reduced interdiffusion matrix D~ in m^2/s, shape (..., n-1, n-1).

        ``frame="number"`` is the number-fixed (lattice-fixed reduced) frame;
        ``frame="volume"`` needs ``partial_volumes`` (m^3/mol, one per element).
        """
        x = self._x(x)
        dmu = self.dmu_dx(x, T)
        mob = self.mobility_model.mobility(x, T, magnetic_factor)
        if frame == "number":
            return number_fixed_matrix(x, mob, dmu)
        if frame == "volume":
            if partial_volumes is None:
                raise ValueError("the volume-fixed frame needs partial_volumes")
            return volume_fixed_matrix(x, mob, dmu, partial_volumes)
        raise ValueError(f"unknown frame {frame!r}")
