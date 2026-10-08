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
from .homogenization import (
    LocalEquilibriumTable,
    OnDemandEquilibriumTable,
    TemperatureTable,
    average_kinetic_factor,
    solve_homogenized,
)
from .datasets import campbell2002_fcc_ni, helander1999_bcc_alni, jonsson1995_bcc_crfeni, kjellqvist2008_cro
from .errors import ConvergenceError, MissingParameterError, UnstableCompositionError, UnsupportedPhaseError
from .mobility import MobilityModel
from .multicomponent_precipitation import (
    MulticomponentResult,
    NucleationSettings,
    PhaseMinimizer,
    PrecipitateThermo,
    junction_barrier_factor,
    misfit_energy,
    solve_multicomponent_precipitation,
)
from .ordered import B2Phase, ordering_energy
from .oxide_scale import ScaleGrowth, metal_oxide_potential, parabolic_scale_growth
from .oxides import AnionPath, GrainBoundaryMobility, JumpPath, OxideCationDiffusion
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
from .solidification import (
    BackDiffusionResult,
    ScheilResult,
    kobayashi,
    scheil_gulliver,
    segregation_index,
    segregation_profile,
    solve_back_diffusion,
)
from .solver import Boundary, DiffusionResult, solve_planar, uniform_faces
from .thermo import GibbsEnergy, R, dmu_dx, reduced_hessian, thermodynamic_factor

__version__ = "0.1.0"

__all__ = [
    "AnionPath",
    "B2Phase",
    "BackDiffusionResult",
    "Boundary",
    "ConvergenceError",
    "DiffusionResult",
    "DiluteSolution",
    "FisherBoundary",
    "FixedInterface",
    "GibbsEnergy",
    "GrainBoundaryMobility",
    "InterstitialPhase",
    "JumpPath",
    "Kinetics",
    "Layer",
    "LineCompound",
    "LocalEquilibrium",
    "LocalEquilibriumTable",
    "MissingParameterError",
    "MobilityModel",
    "MovingBoundaryResult",
    "MulticomponentResult",
    "NucleationSettings",
    "OnDemandEquilibriumTable",
    "OxideCationDiffusion",
    "ParabolicGrowth",
    "PhaseDissolvedError",
    "PhaseMinimizer",
    "PrecipitateThermo",
    "PrecipitationResult",
    "R",
    "ReactionResult",
    "ScaleGrowth",
    "ScheilResult",
    "ShortCircuit",
    "SolutionPhase",
    "StackResult",
    "TemperatureSchedule",
    "TemperatureTable",
    "UnstableCompositionError",
    "UnsupportedPhaseError",
    "average_kinetic_factor",
    "boundary_product_from_slope",
    "campbell2002_fcc_ni",
    "conversion",
    "conversion_integral",
    "dmu_dx",
    "growth_constant_from_conversion",
    "growth_rate",
    "helander1999_bcc_alni",
    "integrated_diffusivity",
    "interdiffusion_from_profile",
    "intrinsic_from_profile",
    "intrinsic_matrix",
    "jonsson1995_bcc_crfeni",
    "junction_barrier_factor",
    "kjellqvist2008_cro",
    "kobayashi",
    "lattice_velocity",
    "layer_thickness",
    "metal_oxide_potential",
    "misfit_energy",
    "nucleation_rate",
    "number_fixed_matrix",
    "ordering_energy",
    "parabolic_growth",
    "parabolic_scale_growth",
    "reduced_hessian",
    "scheil_gulliver",
    "segregation_index",
    "segregation_profile",
    "solubility_from_pycalphad",
    "solve_back_diffusion",
    "solve_fisher",
    "solve_homogenized",
    "solve_kwn",
    "solve_layers",
    "solve_moving_boundary",
    "solve_multicomponent_precipitation",
    "solve_planar",
    "solve_reaction_layers",
    "thermodynamic_factor",
    "to_concentrations",
    "to_mole_fractions",
    "uniform_faces",
    "volume_fixed_lattice_velocity",
    "volume_fixed_matrix",
]
