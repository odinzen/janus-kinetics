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
    temperature but still lowers Fe-rich mobilities above it: against the paper's
    Fig. 5 the Fe tracer diffusivity in pure bcc Fe comes out 0.12 decades too high
    at 1250 C and 0.38 at 960 C, while Cr-rich alloys agree within 0.07.
    """
    return Database(str(files("janus_kinetics") / "data" / "jonsson1995_bcc_crfeni_mob.tdb"))


def campbell2002_fcc_ni() -> Database:
    """Atomic mobilities in fcc Ni-Al-Co-Cr-Hf-Mo-Re-Ta-Ti-W, for Ni-base superalloys.

    Table 2 of Campbell, Boettinger and Kattner, Acta Mater. 50 (2002) 775, with
    their estimation rule for end members the table does not list (p. 778). The
    authors assessed it against Kattner's Ni-superalloy thermodynamics, so other
    descriptions shift the chemical diffusivities somewhat (their Tables 5 and 7
    show the size of that effect).
    """
    return Database(str(files("janus_kinetics") / "data" / "campbell2002_fcc_ni_mob.tdb"))


def kjellqvist2008_cro() -> Database:
    """Thermodynamics of Cr-O between Cr and Cr2O3: bcc Cr with interstitial O, corundum
    (Cr+2,Cr+3)2(Cr+3,Va)1(O-2)3 and O2 gas.

    The Cr-O part of Table 5 of Kjellqvist, Selleby and Sundman, Calphad 32 (2008)
    577, for :func:`janus_kinetics.oxides.chromia`. Hallstrom et al. (2013) fitted
    their Cr2O3 mobilities with Taylor and Dinsdale's Cr-O description, which this
    one builds on; their oxygen tracer curves (Fig. 7) come out within 0.07 decades
    with it. The antiferromagnetic and structure factors are not printed and are
    the usual ones (see the file header); they change nothing above 600 K that
    matters here.
    """
    return Database(str(files("janus_kinetics") / "data" / "kjellqvist2008_cro.tdb"))


# Appendix A (p. 1151) of Helander and Agren, Acta Mater. 47 (1999) 1141: ordering
# contributions dQ^order_kij (J/mol) to the activation energy of element k in B2
# Al-Ni, keyed (k, i, j) with i, j in either order; VA marks structural vacancies.
HELANDER1999_ALNI_ORDERING = {
    ("AL", "NI", "AL"): 359700.0,
    ("AL", "AL", "VA"): -1071200.0,
    ("AL", "NI", "VA"): 0.0,
    ("NI", "NI", "AL"): 314400.0,
    ("NI", "AL", "VA"): -107000.0,
    ("NI", "NI", "VA"): 0.0,
}


def helander1999_bcc_alni() -> tuple[Database, dict]:
    """Mobilities of Al and Ni in B2-ordered bcc Al-Ni: the disordered database and the ordering terms.

    Appendix A of Helander and Agren, Acta Mater. 47 (1999) 1141, for
    :class:`janus_kinetics.B2Phase`. The disordered parameters are for the
    hypothetical bcc A2 phase (the same for pure Al and pure Ni by their
    assumption, p. 1147). The authors fitted them with the Al-Ni thermodynamics of
    Ansara et al. (1997); Al-Fe is not shipped because its magnetic term (their
    eq. 8) is not usable as printed.
    """
    return (Database(str(files("janus_kinetics") / "data" / "helander1999_bcc_alni_mob.tdb")),
            dict(HELANDER1999_ALNI_ORDERING))
