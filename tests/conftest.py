"""Small synthetic databases for the tests.

The values are invented for testing only; nothing here is an assessment. Each
test compares the package against a closed-form expression evaluated on the
same parameters.
"""

import pytest
from pycalphad import Database

from calphad_kinetics.datasets import jonsson1995_bcc_crfeni

HEADER = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT A BLANK 1 0 0 !
ELEMENT B BLANK 1 0 0 !
ELEMENT C BLANK 1 0 0 !
TYPE_DEFINITION % SEQ * !
"""

# Mobility parameters in the usual form MQ = -Q + R*T*LN(M0).
MOBILITIES = """
PARAMETER MQ(SOL&A,A{va};0) 298.15 -150000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(SOL&A,B{va};0) 298.15 -170000+R*T*LN(2E-4); 6000 N !
PARAMETER MQ(SOL&A,C{va};0) 298.15 -160000+R*T*LN(5E-5); 6000 N !
PARAMETER MQ(SOL&B,A{va};0) 298.15 -140000+R*T*LN(3E-5); 6000 N !
PARAMETER MQ(SOL&B,B{va};0) 298.15 -180000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(SOL&B,C{va};0) 298.15 -155000+R*T*LN(8E-5); 6000 N !
PARAMETER MQ(SOL&C,A{va};0) 298.15 -145000+R*T*LN(6E-5); 6000 N !
PARAMETER MQ(SOL&C,B{va};0) 298.15 -165000+R*T*LN(1.5E-4); 6000 N !
PARAMETER MQ(SOL&C,C{va};0) 298.15 -152000+R*T*LN(9E-5); 6000 N !
PARAMETER MQ(SOL&A,A,B{va};0) 298.15 12000; 6000 N !
PARAMETER MQ(SOL&B,B,C{va};1) 298.15 -8000; 6000 N !
"""

IDEAL_TERNARY = HEADER + """
PHASE SOL % 1 1 !
CONSTITUENT SOL :A,B,C: !
""" + MOBILITIES.format(va="")

# Same chemistry with an interstitial vacancy sublattice, so the lattice
# mapping (site fractions of VA set to one) is exercised.
REGULAR_TERNARY = HEADER + """
PHASE SOL % 2 1 1 !
CONSTITUENT SOL :A,B,C:VA: !
PARAMETER G(SOL,A:VA;0) 298.15 -1000-2*T; 6000 N !
PARAMETER G(SOL,B:VA;0) 298.15 500; 6000 N !
PARAMETER G(SOL,C:VA;0) 298.15 0; 6000 N !
PARAMETER G(SOL,A,B:VA;0) 298.15 -12000+3*T; 6000 N !
PARAMETER G(SOL,A,B:VA;1) 298.15 4000; 6000 N !
PARAMETER G(SOL,A,C:VA;0) 298.15 8000; 6000 N !
PARAMETER G(SOL,B,C:VA;0) 298.15 -5000; 6000 N !
PARAMETER G(SOL,B,C:VA;1) 298.15 -2500; 6000 N !
PARAMETER G(SOL,A,B,C:VA;0) 298.15 15000; 6000 N !
""" + MOBILITIES.format(va=":VA")

# Binary with L0 = 30 kJ/mol > 2RT at 1000 K, so x = 0.5 is inside the spinodal.
GAP_BINARY = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT A BLANK 1 0 0 !
ELEMENT B BLANK 1 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE SOL % 1 1 !
CONSTITUENT SOL :A,B: !
PARAMETER G(SOL,A,B;0) 298.15 30000; 6000 N !
PARAMETER MQ(SOL&A,A;0) 298.15 -150000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(SOL&A,B;0) 298.15 -150000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(SOL&B,A;0) 298.15 -150000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(SOL&B,B;0) 298.15 -150000+R*T*LN(1E-4); 6000 N !
"""

# Regular-solution parameters of REGULAR_TERNARY at any T (J/mol), for closed forms.
REGULAR_L = {("A", "B"): (lambda T: -12000 + 3 * T, 4000.0), ("A", "C"): (lambda T: 8000.0, 0.0),
             ("B", "C"): (lambda T: -5000.0, -2500.0)}
REGULAR_LABC = 15000.0


@pytest.fixture(scope="session")
def ideal_db():
    return Database(IDEAL_TERNARY)


@pytest.fixture(scope="session")
def regular_db():
    return Database(REGULAR_TERNARY)


@pytest.fixture(scope="session")
def gap_db():
    return Database(GAP_BINARY)


@pytest.fixture(scope="session")
def jonsson_db():
    return jonsson1995_bcc_crfeni()
