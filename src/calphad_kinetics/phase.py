"""Single-phase diffusion data assembled from one pycalphad Database."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pycalphad import Database

from ._lattice import SubstitutionalLattice
from .diffusivity import intrinsic_matrix, lattice_velocity, number_fixed_matrix, volume_fixed_matrix
from .mobility import MobilityModel
from .schedule import Temperature, temperature_at
from .thermo import GibbsEnergy, check_stability, dmu_dx, reduced_hessian, thermodynamic_factor

Frame = Literal["number", "volume"]

# The ideal-mixing curvature RT/x_i diverges at x_i = 0 while the product
# x_i M_i dmu_i/dx_i stays finite, so compositions are floored before evaluation.
COMPOSITION_FLOOR = 1e-12


@dataclass(frozen=True)
class ShortCircuit:
    """Fast diffusion paths (dislocations or grain boundaries) averaged into the lattice.

    Hart, Acta Metall. 5 (1957) 597, eq. (5), for self-diffusion along dislocations:
    an atom that spends a fraction f of its time in the paths sees
    D_eff = f D_path + (1 - f) D_lattice. Two extensions are this package's, not
    Hart's. The same time-fraction argument is applied to grain boundaries, and,
    through D* = RT M (Andersson and Agren, J. Appl. Phys. 72 (1992) 1350,
    eq. (61)), to each element's mobility and so to every diffusion matrix built
    from it. Hart's condition is that a diffusion distance spans many paths; a
    coarse microstructure with isolated boundaries is outside it.

    ``fraction`` is Hart's f, the fraction of atoms in the paths. ``mobility_db``
    holds MQ/MF parameters for diffusion along the paths, entered under the same
    phase name and constitution as the lattice parameters.
    """

    fraction: float
    mobility_db: Database

    def __post_init__(self):
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"short-circuit fraction must lie in [0, 1], got {self.fraction}")


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
        short_circuit: ShortCircuit | None = None,
    ):
        self.lattice = SubstitutionalLattice.from_database(db, phase_name, elements, dependent)
        self.gibbs = GibbsEnergy(db, self.lattice)
        self.mobility_model = MobilityModel(mobility_db or db, self.lattice)
        self.short_circuit = short_circuit
        self._path_model = MobilityModel(short_circuit.mobility_db, self.lattice) if short_circuit else None

    def _mobility(self, x: np.ndarray, T, magnetic_factor) -> np.ndarray:
        m = self.mobility_model.mobility(x, T, magnetic_factor)
        if self._path_model is None:
            return m
        f = self.short_circuit.fraction
        return (1.0 - f) * m + f * self._path_model.mobility(x, T, magnetic_factor)

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

    def chemical_potentials(self, x, T) -> np.ndarray:
        """Chemical potentials of every element in J/mol, shape (..., n)."""
        return self.gibbs.chemical_potentials(self._x(x), T)

    def thermodynamic_factor(self, x, T) -> np.ndarray:
        """Dimensionless (x_i/RT) dmu_i/dx_j, shape (..., n, n-1)."""
        x = self._x(x)
        return thermodynamic_factor(x, self.dmu_dx(x, T), T)

    def mobility(self, x, T, magnetic_factor=1.0) -> np.ndarray:
        """Atomic mobilities in m^2 mol / (J s), including any short-circuit paths."""
        return self._mobility(self._x(x), T, magnetic_factor)

    def intrinsic_matrix(self, x, T, magnetic_factor=1.0) -> np.ndarray:
        """Lattice-frame diffusivities D^I_kj in m^2/s, shape (..., n, n-1)."""
        x = self._x(x)
        return intrinsic_matrix(x, self._mobility(x, T, magnetic_factor), self.dmu_dx(x, T))

    def lattice_velocity(self, x, dx_dz, T, magnetic_factor=1.0) -> np.ndarray:
        """Kirkendall (marker) velocity relative to the number-fixed frame, m/s.

        ``dx_dz`` are the independent mole-fraction gradients (1/m), shape (..., n-1).
        """
        return lattice_velocity(self.intrinsic_matrix(x, T, magnetic_factor), np.asarray(dx_dz, dtype=float))

    def interdiffusion_matrix(
        self,
        x,
        T,
        frame: Frame = "number",
        partial_volumes: np.ndarray | None = None,
        magnetic_factor=1.0,
    ) -> np.ndarray:
        """Reduced interdiffusion matrix D~ in m^2/s, shape (..., n-1, n-1).

        ``frame="number"`` is the number-fixed frame (the lattice frame differs by
        the Kirkendall velocity; see :meth:`intrinsic_matrix`);
        ``frame="volume"`` needs ``partial_volumes`` (m^3/mol, one per element).
        """
        x = self._x(x)
        dmu = self.dmu_dx(x, T)
        mob = self._mobility(x, T, magnetic_factor)
        if frame == "number":
            return number_fixed_matrix(x, mob, dmu)
        if frame == "volume":
            if partial_volumes is None:
                raise ValueError("the volume-fixed frame needs partial_volumes")
            return volume_fixed_matrix(x, mob, dmu, partial_volumes)
        raise ValueError(f"unknown frame {frame!r}")

    def interdiffusion_function(self, temperature: Temperature, frame: Frame = "number", partial_volumes=None):
        """D~ as a function of (x, t) for the solvers, at a constant T or along a schedule."""
        return lambda x, t: self.interdiffusion_matrix(x, temperature_at(temperature, t), frame, partial_volumes)

    def lattice_velocity_function(self, temperature: Temperature):
        """Marker velocity as a function of (x, dx_dz, t), for ``solve_planar(..., lattice_velocity=...)``."""
        return lambda x, g, t: self.lattice_velocity(x, g, temperature_at(temperature, t))
