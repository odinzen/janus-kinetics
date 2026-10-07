"""Multicomponent diffusion kinetics built on pycalphad."""

from .couple_analysis import interdiffusion_from_profile, intrinsic_from_profile
from .diffusivity import (
    intrinsic_matrix,
    lattice_velocity,
    number_fixed_matrix,
    to_concentrations,
    to_mole_fractions,
    volume_fixed_lattice_velocity,
    volume_fixed_matrix,
)
from .interstitial import InterstitialPhase
from .grain_boundary import FisherBoundary, boundary_product_from_slope, solve_fisher
from .homogenization import LocalEquilibriumTable, TemperatureTable, average_kinetic_factor, solve_homogenized
from .datasets import campbell2002_fcc_ni, jonsson1995_bcc_crfeni
from .errors import ConvergenceError, MissingParameterError, UnstableCompositionError, UnsupportedPhaseError
from .mobility import MobilityModel
from .moving_boundary import (
    FixedInterface,
    Layer,
    LocalEquilibrium,
    MovingBoundaryResult,
    PhaseDissolvedError,
    StackResult,
    solve_layers,
    solve_moving_boundary,
)
from .particle_reaction import conversion, conversion_integral, growth_constant_from_conversion, layer_thickness
from .phase import ShortCircuit, SolutionPhase
from .precipitation import (
    DiluteSolution,
    Kinetics,
    PrecipitationResult,
    growth_rate,
    nucleation_rate,
    solubility_from_pycalphad,
    solve_kwn,
)
from .reaction_layers import LineCompound, ParabolicGrowth, ReactionResult, integrated_diffusivity, parabolic_growth, solve_reaction_layers
from .schedule import TemperatureSchedule
from .solver import Boundary, DiffusionResult, solve_planar, uniform_faces
from .thermo import GibbsEnergy, R, dmu_dx, reduced_hessian, thermodynamic_factor

__version__ = "0.1.0"

__all__ = [
    "Boundary",
    "ConvergenceError",
    "DiffusionResult",
    "DiluteSolution",
    "FisherBoundary",
    "FixedInterface",
    "GibbsEnergy",
    "InterstitialPhase",
    "Kinetics",
    "Layer",
    "LineCompound",
    "LocalEquilibriumTable",
    "LocalEquilibrium",
    "MissingParameterError",
    "MobilityModel",
    "MovingBoundaryResult",
    "ParabolicGrowth",
    "PhaseDissolvedError",
    "PrecipitationResult",
    "R",
    "ReactionResult",
    "ShortCircuit",
    "SolutionPhase",
    "StackResult",
    "TemperatureSchedule",
    "TemperatureTable",
    "UnstableCompositionError",
    "UnsupportedPhaseError",
    "average_kinetic_factor",
    "campbell2002_fcc_ni",
    "conversion",
    "conversion_integral",
    "boundary_product_from_slope",
    "dmu_dx",
    "growth_constant_from_conversion",
    "growth_rate",
    "interdiffusion_from_profile",
    "integrated_diffusivity",
    "intrinsic_from_profile",
    "intrinsic_matrix",
    "jonsson1995_bcc_crfeni",
    "lattice_velocity",
    "layer_thickness",
    "nucleation_rate",
    "number_fixed_matrix",
    "parabolic_growth",
    "reduced_hessian",
    "solubility_from_pycalphad",
    "solve_fisher",
    "solve_homogenized",
    "solve_kwn",
    "solve_layers",
    "solve_moving_boundary",
    "solve_planar",
    "solve_reaction_layers",
    "thermodynamic_factor",
    "to_concentrations",
    "to_mole_fractions",
    "uniform_faces",
    "volume_fixed_lattice_velocity",
    "volume_fixed_matrix",
]
