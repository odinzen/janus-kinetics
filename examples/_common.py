"""Shared pieces of the examples: an ideal bcc description and figure output."""

from pathlib import Path

from pycalphad import Database

OUTPUT = Path(__file__).parent / "output"

# Ideal-solution bcc Cr-Fe-Ni. It stands in for an assessed Gibbs energy so the
# examples run on published mobilities alone; with it the thermodynamic factor is
# exactly one and every diffusion effect comes from the mobilities.
IDEAL_BCC = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT CR BCC_A2 51.996 0 0 !
ELEMENT FE BCC_A2 55.847 0 0 !
ELEMENT NI FCC_A1 58.69 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE BCC_A2 % 2 1 3 !
CONSTITUENT BCC_A2 :CR,FE,NI:VA: !
PARAMETER G(BCC_A2,CR:VA;0) 298.15 0; 6000 N !
PARAMETER G(BCC_A2,FE:VA;0) 298.15 0; 6000 N !
PARAMETER G(BCC_A2,NI:VA;0) 298.15 0; 6000 N !
"""


def ideal_bcc() -> Database:
    return Database(IDEAL_BCC)


def figure_path(name: str) -> Path:
    OUTPUT.mkdir(exist_ok=True)
    return OUTPUT / name
