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


def campbell2002_fcc_ni() -> Database:
    """Atomic mobilities in fcc Ni-Al-Co-Cr-Hf-Mo-Re-Ta-Ti-W, for Ni-base superalloys.

    Table 2 of Campbell, Boettinger and Kattner, Acta Mater. 50 (2002) 775, with
    their estimation rule for end members the table does not list (p. 778). Use it
    with a thermodynamic description whose FCC_A1 has the same (elements)(VA)
    sublattices; the authors assessed it against Kattner's Ni-superalloy
    thermodynamics, so other descriptions shift the chemical diffusivities somewhat
    (their Tables 5 and 7 show the size of that effect).
    """
    return Database(str(files("calphad_kinetics") / "data" / "campbell2002_fcc_ni_mob.tdb"))
