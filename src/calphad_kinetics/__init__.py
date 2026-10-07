"""Multicomponent diffusion kinetics built on pycalphad."""

from .diffusivity import intrinsic_matrix, lattice_velocity, number_fixed_matrix, volume_fixed_matrix
from .homogenization import LocalEquilibriumTable, average_kinetic_factor, solve_homogenized
from .errors import ConvergenceError, MissingParameterError, UnstableCompositionError, UnsupportedPhaseError
from .mobility import MobilityModel
from .moving_boundary import (
    FixedInterface,
    LocalEquilibrium,
    MovingBoundaryResult,
    PhaseDissolvedError,
    solve_moving_boundary,
)
from .phase import ShortCircuit, SolutionPhase
from .solver import Boundary, DiffusionResult, solve_planar, uniform_faces
from .thermo import GibbsEnergy, R, dmu_dx, reduced_hessian, thermodynamic_factor

__version__ = "0.1.0.dev0"

__all__ = [
    "Boundary",
    "ConvergenceError",
    "DiffusionResult",
    "FixedInterface",
    "GibbsEnergy",
    "LocalEquilibriumTable",
    "LocalEquilibrium",
    "MissingParameterError",
    "MobilityModel",
    "MovingBoundaryResult",
    "PhaseDissolvedError",
    "R",
    "ShortCircuit",
    "SolutionPhase",
    "UnstableCompositionError",
    "UnsupportedPhaseError",
    "average_kinetic_factor",
    "dmu_dx",
    "intrinsic_matrix",
    "lattice_velocity",
    "number_fixed_matrix",
    "reduced_hessian",
    "solve_homogenized",
    "solve_moving_boundary",
    "solve_planar",
    "thermodynamic_factor",
    "uniform_faces",
    "volume_fixed_matrix",
]
