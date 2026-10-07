"""Published parameter sets shipped with the package."""

from __future__ import annotations

from importlib.resources import files

from pycalphad import Database


def jonsson1995_bcc_crfeni() -> Database:
    """Atomic mobilities of Cr, Fe and Ni in bcc Cr-Fe-Ni, paramagnetic part.

    Transcribed from the Appendix (p. 1421) of B. Jonsson, ISIJ Int. 35 (1995)
    1415. Mobility (MQ/MF) parameters only: combine it with a thermodynamic
    description through ``SolutionPhase(..., mobility_db=...)``. The ferromagnetic
    factor of the paper is not included, so it applies above the Curie temperature.
    """
    return Database(str(files("calphad_kinetics") / "data" / "jonsson1995_bcc_crfeni_mob.tdb"))
