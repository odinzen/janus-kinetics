"""Single-phase diffusion data assembled from one pycalphad Database."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pycalphad import Database
from tinydb import where

from ._lattice import SubstitutionalLattice
from .diffusivity import (
    intrinsic_matrix,
    lattice_velocity,
    number_fixed_matrix,
    to_mole_fractions,
    volume_fixed_lattice_velocity,
    volume_fixed_matrix,
)
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
        self._warn_about_omitted_terms(db, mobility_db or db)
        self.short_circuit = short_circuit
        self._path_model = MobilityModel(short_circuit.mobility_db, self.lattice) if short_circuit else None

    def _warn_about_omitted_terms(self, db: Database, mob_db: Database) -> None:
        """Say so when the databases ask for physics the mobilities here leave out."""
        name = self.lattice.phase_name
        magnetic = db.search((where("phase_name") == name) & (where("parameter_type").one_of(["TC", "BMAGN"])))
        if magnetic:
            warnings.warn(f"{name} has a magnetic Gibbs energy, but the mobilities omit the ferromagnetic factor "
                          "(pass magnetic_factor to scale them); Fe-rich mobilities near or below Tc will be too high",
                          stacklevel=3)
        unused = mob_db.search((where("phase_name") == name) & ~(where("parameter_type").one_of(["G", "L", "TC", "BMAGN", "MQ", "MF"])))
        kinds = sorted({p["parameter_type"] for p in unused})
        if kinds:
            warnings.warn(f"{name}: parameters of type {kinds} are present but not used by this package", stacklevel=3)

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
        return self._dmu(self._x(x), T)

    def _dmu(self, x: np.ndarray, T) -> np.ndarray:
        """dmu_dx for compositions already checked and floored by ``_x``."""
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
        return intrinsic_matrix(x, self._mobility(x, T, magnetic_factor), self._dmu(x, T))

    def lattice_velocity(self, x, gradient, T, magnetic_factor=1.0, partial_volumes=None) -> np.ndarray:
        """Kirkendall (marker) velocity, m/s.

        Relative to the number-fixed frame, with ``gradient`` the independent
        mole-fraction gradients (1/m); with ``partial_volumes`` relative to the
        volume-fixed frame, with ``gradient`` the independent concentration
        gradients (mol/m^4). Shape (..., n-1) either way.
        """
        if partial_volumes is None:
            return lattice_velocity(self.intrinsic_matrix(x, T, magnetic_factor), np.asarray(gradient, dtype=float))
        x = self._x(x)
        return volume_fixed_lattice_velocity(x, self._mobility(x, T, magnetic_factor), self._dmu(x, T),
                                             partial_volumes, gradient)

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
        dmu = self._dmu(x, T)
        mob = self._mobility(x, T, magnetic_factor)
        if frame == "number":
            return number_fixed_matrix(x, mob, dmu)
        if frame == "volume":
            if partial_volumes is None:
                raise ValueError("the volume-fixed frame needs partial_volumes")
            return volume_fixed_matrix(x, mob, dmu, partial_volumes)
        raise ValueError(f"unknown frame {frame!r}")

    def interdiffusion_function(self, temperature: Temperature, frame: Frame = "number", partial_volumes=None):
        """D~ as a function of the solver variables and t, at a constant T or along a schedule.

        In the number-fixed frame the variables are the independent mole fractions.
        In the volume-fixed frame (``partial_volumes`` given) they are the
        independent concentrations c (mol/m^3), so ``solve_planar`` conserves moles
        while the molar volume varies with composition; convert with
        :func:`janus_kinetics.diffusivity.to_concentrations` and ``to_mole_fractions``.
        """
        if frame == "volume":
            if partial_volumes is None:
                raise ValueError("the volume-fixed frame needs partial_volumes")
            vol = np.asarray(partial_volumes, dtype=float)
            return lambda c, t: self.interdiffusion_matrix(to_mole_fractions(c, vol), temperature_at(temperature, t),
                                                           "volume", vol)
        return lambda x, t: self.interdiffusion_matrix(x, temperature_at(temperature, t), frame, partial_volumes)

    def lattice_velocity_function(self, temperature: Temperature, partial_volumes=None):
        """Marker velocity as a function of (variables, gradients, t), for ``solve_planar(..., lattice_velocity=...)``.

        The variables are mole fractions, or concentrations when ``partial_volumes`` is given.
        """
        if partial_volumes is None:
            return lambda x, g, t: self.lattice_velocity(x, g, temperature_at(temperature, t))
        vol = np.asarray(partial_volumes, dtype=float)
        return lambda c, g, t: self.lattice_velocity(to_mole_fractions(c, vol), g, temperature_at(temperature, t),
                                                     partial_volumes=vol)
