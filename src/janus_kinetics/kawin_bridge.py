"""Hand janus-kinetics results to kawin for precipitation and strength, and run kawin on janus's inputs.

kawin (Ury et al., Acta Mater. 255 (2023) 118988) is an optional dependency, installed with
``pip install janus-kinetics[kawin]``. This module imports it only when a function here needs it, so
janus-kinetics itself never requires it.

The two packages read the same pycalphad databases but number compositions differently. janus keeps the
solvent last and passes the independent mole fractions in the caller's order; kawin takes the solvent
first and the solutes after it. Listing kawin's elements as solvent, then janus's independent elements in
janus's order, makes the two composition arrays the same numbers, which is what every function here does.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pycalphad import Database, variables as v
from tinydb import where

from ._lattice import SubstitutionalLattice
from .mobility import MOBILITY_TYPES

__all__ = [
    "KawinPrecipitation",
    "independent_copy",
    "kawin_elements",
    "kawin_history",
    "kawin_model",
    "kawin_thermodynamics",
    "local_compositions",
    "merge_mobilities",
]


def _require_kawin():
    try:
        import kawin  # noqa: F401
    except ImportError as exc:
        raise ImportError("this needs kawin; install it with pip install janus-kinetics[kawin]") from exc


def independent_copy(db: Database) -> Database:
    """A copy of ``db`` sharing nothing with it. pycalphad's deepcopy of a Database copies the parameters but
    shares the phase and function dictionaries, and kawin adds a DIS_ phase for each order-disorder model to
    the database it is given, which then breaks janus's equilibria on the original."""
    return pickle.loads(pickle.dumps(db))


def kawin_elements(elements: Sequence[str], dependent: str | None = None) -> list[str]:
    """kawin's element list for janus's ``elements``: the solvent first, the independent elements in janus's
    order, then VA."""
    names = [el.upper() for el in elements if el.upper() != "VA"]
    dependent = names[-1] if dependent is None else dependent.upper()
    if dependent not in names:
        raise ValueError(f"dependent element {dependent} is not among {names}")
    return [dependent] + [el for el in names if el != dependent] + ["VA"]


def merge_mobilities(db: Database, mobility_db: Database, phase: str, elements: Sequence[str]) -> Database:
    """A copy of ``db`` holding also the MQ and MF parameters of ``phase`` from ``mobility_db``, so a program
    that reads one database (kawin does) sees janus's thermodynamics and mobilities together.

    A mobility database often writes the phase with a vacancy-only sublattice the thermodynamic one lacks,
    fcc as (Al,Cr,Ni)(Va) against (Al,Cr,Ni), say; the two are the same lattice, so each parameter is
    re-keyed onto ``db``'s sublattices. Parameters naming elements outside ``elements`` are left out, as are
    functions the copied parameters do not use. Parameters of that kind already in ``db`` are kept and a
    clash raises, since two values for one mobility would be summed.
    """
    phase = phase.upper()
    names = [el.upper() for el in elements if el.upper() != "VA"]
    target = SubstitutionalLattice.from_database(db, phase, names)
    source = SubstitutionalLattice.from_database(mobility_db, phase, names)
    wanted = {v.Species(el) for el in names} | {v.Species("*")}
    merged = independent_copy(db)
    present = merged.search((where("phase_name") == phase) & (where("parameter_type").one_of(MOBILITY_TYPES)))
    seen = {(p["parameter_type"], str(p["diffusing_species"].name), p["constituent_array"], p["parameter_order"])
            for p in present}
    for p in mobility_db.search((where("phase_name") == phase) & (where("parameter_type").one_of(MOBILITY_TYPES))):
        mixing = p["constituent_array"][source.mixing_sublattice]
        if p["diffusing_species"].name not in names or not set(mixing) <= wanted:
            continue
        array = [(v.Species("VA"),)] * len(db.phases[phase].constituents)
        array[target.mixing_sublattice] = mixing
        key = (p["parameter_type"], str(p["diffusing_species"].name), tuple(array), p["parameter_order"])
        if key in seen:
            raise ValueError(f"{phase} already holds {key[0]} for {key[1]} in {key[2]}")
        seen.add(key)
        q = dict(p)
        q["constituent_array"] = tuple(array)
        merged._parameters.insert(q)
        for symbol in p["parameter"].free_symbols:
            name = str(symbol)
            if name in mobility_db.symbols and name not in merged.symbols:
                merged.symbols[name] = mobility_db.symbols[name]
    return merged


@dataclass(frozen=True)
class KawinPrecipitation:
    """kawin's states of one precipitate in the names of :class:`MulticomponentResult`: times (s), mean
    radius (m), number density (m^-3), volume fraction, the matrix mole fractions (t, n-1) in janus's order,
    the driving force per volume of precipitate (J/m^3, as kawin keeps it), and the yield strength (Pa)
    if a strength model was coupled."""

    times: np.ndarray
    mean_radius: np.ndarray
    number_density: np.ndarray
    volume_fraction: np.ndarray
    matrix_composition: np.ndarray
    driving_force: np.ndarray
    strength: np.ndarray | None = None


def kawin_thermodynamics(db: Database, matrix_phase: str, precipitate_phases: Sequence[str], elements: Sequence[str],
                         mobility_db: Database | None = None, dependent: str | None = None, **kwargs):
    """kawin's MulticomponentThermodynamics on janus's database (merged with ``mobility_db`` by
    :func:`merge_mobilities` when the mobilities sit in a file of their own) and janus's element order.
    kawin gets a copy of the database, never ``db`` itself (see :func:`independent_copy`). Keyword arguments
    go to kawin."""
    _require_kawin()
    from kawin.thermo import MulticomponentThermodynamics

    db = independent_copy(db) if mobility_db is None else merge_mobilities(db, mobility_db, matrix_phase, elements)
    return MulticomponentThermodynamics(db, kawin_elements(elements, dependent),
                                        [matrix_phase.upper()] + [p.upper() for p in precipitate_phases], **kwargs)


def kawin_model(thermodynamics, precipitate_phase: str, alloy: Sequence[float], T, settings, solutes: Sequence[str],
                atoms_per_cell: float | None = None):
    """An unsolved kawin PrecipitateModel for the precipitate, alloy and temperature that
    :func:`solve_multicomponent_precipitation` takes, with janus's :class:`NucleationSettings`: the same
    interfacial energy, the same molar volume for both phases, homogeneous nucleation on
    ``settings.site_density`` sites per m^3. kawin takes the unit-cell volume and ``atoms_per_cell``; by
    default the count is the one that makes its cell edge equal ``settings.lattice_parameter`` (4 for fcc
    when the two are consistent), and the cell volume is then that count times the atomic volume, so
    kawin's molar volume is ``settings.molar_volume`` either way. ``T`` is a temperature in K or a kawin TemperatureParameters.

    Couple a kawin strength model with ``model.addCouplingModel`` before solving. Only bulk nucleation
    without misfit energy is carried over; anything else raises, since kawin describes it differently.
    """
    _require_kawin()
    from kawin.precipitation import MatrixParameters, PrecipitateModel, PrecipitateParameters

    from .multicomponent_precipitation import N_A

    if settings.site != "bulk" or settings.misfit != 0:
        raise NotImplementedError("only bulk nucleation without misfit energy is carried over to kawin")
    atomic_volume = settings.molar_volume / N_A
    if atoms_per_cell is None:
        atoms_per_cell = settings.lattice_parameter ** 3 / atomic_volume
    cell = atoms_per_cell * atomic_volume
    matrix = MatrixParameters([s.upper() for s in solutes])
    matrix.initComposition = [float(x) for x in np.atleast_1d(alloy)][: len(solutes)]
    matrix.volume.setVolume(cell, "VA", atoms_per_cell)
    matrix.nucleationSites.setBulkDensity(settings.site_density)
    precipitate = PrecipitateParameters(precipitate_phase.upper())
    precipitate.gamma = settings.gamma
    precipitate.volume.setVolume(cell, "VA", atoms_per_cell)
    precipitate.nucleation.setNucleationType("bulk")
    return PrecipitateModel(matrix, precipitate, thermodynamics=thermodynamics, temperature=T)


def kawin_history(model, strength_model=None) -> KawinPrecipitation:
    """The solved ``model``'s states (its first precipitate) as a :class:`KawinPrecipitation`; with
    ``strength_model`` (coupled before solving) also kawin's total yield strength at each state."""
    data = model.toDict()
    strength = None
    if strength_model is not None:
        strength = np.asarray(strength_model.totalStrength(model), dtype=float)
    return KawinPrecipitation(
        times=np.asarray(data["time"], dtype=float),
        mean_radius=np.asarray(data["Ravg"], dtype=float)[:, 0],
        number_density=np.asarray(data["precipitateDensity"], dtype=float)[:, 0],
        volume_fraction=np.asarray(data["volFrac"], dtype=float)[:, 0],
        matrix_composition=np.asarray(data["composition"], dtype=float),
        driving_force=np.asarray(data["drivingForce"], dtype=float)[:, 0],
        strength=strength,
    )


def local_compositions(z: np.ndarray, x: np.ndarray, positions: Sequence[float]) -> np.ndarray:
    """Mole fractions (len(positions), n-1) of a janus profile ``x`` (cells, n-1) at cell centres ``z``,
    interpolated linearly to ``positions`` (m), for instance the dendrite axis and the arm edge of a
    homogenized casting to age in kawin."""
    z, x = np.asarray(z, dtype=float), np.asarray(x, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if np.any(positions < z[0] - 1e-15) or np.any(positions > z[-1] + 1e-15):
        raise ValueError(f"positions must lie between the first and last cell centres, {z[0]:g} and {z[-1]:g} m")
    return np.stack([np.interp(positions, z, x[:, j]) for j in range(x.shape[1])], axis=-1)
