"""Published parameter sets shipped with the package."""

from __future__ import annotations

from importlib.resources import files

from pycalphad import Database


def jonsson1995_bcc_crfeni() -> Database:
    """Atomic mobilities of Cr, Fe and Ni in bcc Cr-Fe-Ni, paramagnetic part.

    Transcribed from the Appendix (p. 1421) of B. Jonsson, ISIJ Int. 35 (1995)
    1415. It holds mobility (MQ/MF) parameters only, to be combined with a
    thermodynamic description through ``SolutionPhase(..., mobility_db=...)``. The ferromagnetic
    factor of the paper's eq. (3) is not included. It is largest below the Curie
    temperature but still lowers Fe-rich mobilities some way above it, so results
    for Fe-rich alloys within a few hundred kelvin of Tc are approximate.
    """
    return Database(str(files("calphad_kinetics") / "data" / "jonsson1995_bcc_crfeni_mob.tdb"))
