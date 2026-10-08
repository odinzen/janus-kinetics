"""Phases with interstitial elements, after Andersson and Agren, J. Appl. Phys. 72 (1992) 1350, sections II.B and VI.A."""

import os

import numpy as np
import pycalphad
import pytest
from pycalphad import Database, equilibrium, variables as v

from janus_kinetics import R
from janus_kinetics.interstitial import InterstitialPhase
from conftest import HEADER

T = 1200.0

# A(1)(C,VA)(3): ideal mixing on the interstitial sublattice, three interstitial sites per A atom
IDEAL = HEADER + """
PHASE BCC % 2 1 3 !
CONSTITUENT BCC :A:C,VA: !
PARAMETER G(BCC,A:VA;0) 298.15 0; 6000 N !
PARAMETER G(BCC,A:C;0) 298.15 30000; 6000 N !
PARAMETER MQ(BCC&C,A:VA;0) 298.15 -90000+R*T*LN(2E-6); 6000 N !
PARAMETER MQ(BCC&C,A:C;0) 298.15 -95000+R*T*LN(2E-6); 6000 N !
PARAMETER MQ(BCC&A,A:VA;0) 298.15 -200000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(BCC&A,A:C;0) 298.15 -210000+R*T*LN(1E-4); 6000 N !
"""

# (A,B)(1)(C,VA)(3), regular in both sublattices, for the ternary checks
TERNARY = HEADER + """
PHASE BCC % 2 1 3 !
CONSTITUENT BCC :A,B:C,VA: !
PARAMETER G(BCC,A:VA;0) 298.15 0; 6000 N !
PARAMETER G(BCC,B:VA;0) 298.15 1500; 6000 N !
PARAMETER G(BCC,A:C;0) 298.15 30000; 6000 N !
PARAMETER G(BCC,B:C;0) 298.15 10000; 6000 N !
PARAMETER G(BCC,A,B:VA;0) 298.15 -6000; 6000 N !
PARAMETER G(BCC,A,B:C;0) 298.15 4000; 6000 N !
PARAMETER G(BCC,A:C,VA;0) 298.15 -9000; 6000 N !
PARAMETER MQ(BCC&C,A:VA;0) 298.15 -90000+R*T*LN(2E-6); 6000 N !
PARAMETER MQ(BCC&C,B:VA;0) 298.15 -110000+R*T*LN(2E-6); 6000 N !
PARAMETER MQ(BCC&C,A:C;0) 298.15 -95000; 6000 N !
PARAMETER MQ(BCC&C,A,B:VA;0) 298.15 12000; 6000 N !
PARAMETER MQ(BCC&A,A:VA;0) 298.15 -200000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(BCC&A,B:VA;0) 298.15 -190000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(BCC&B,A:VA;0) 298.15 -205000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(BCC&B,B:VA;0) 298.15 -185000+R*T*LN(1E-4); 6000 N !
"""


@pytest.fixture(scope="module")
def ideal():
    return InterstitialPhase(Database(IDEAL), "BCC", ["A"], ["C"])


@pytest.fixture(scope="module")
def ternary():
    return InterstitialPhase(Database(TERNARY), "BCC", ["A", "B"], ["C"], dependent="A")


def test_ideal_interstitial_diffusivity_in_closed_form(ideal):
    """With mu_C = const + RT ln(y_C / y_Va) and u_C = 3 y_C, eqs. (63)-(65) give
    D_CC = RT [Omega_C + 3 y_C Omega_A / (1 - y_C)]: the tracer value plus the frame term."""
    for y in (1e-4, 0.02, 0.2):
        u = np.array([3 * y])
        d = ideal.interdiffusion_matrix(u, T)[0, 0]
        omega_a, omega_c = ideal.mobility(u, T)
        assert d == pytest.approx(R * T * (omega_c + 3 * y * omega_a / (1 - y)), rel=1e-10)


def test_interstitial_chemical_potential_is_ideal(ideal):
    for y in (0.01, 0.3):
        mu_c = ideal.chemical_potentials(np.array([3 * y]), T)[1]
        assert mu_c == pytest.approx(30000 / 3 + R * T * np.log(y / (1 - y)), rel=1e-10)


def test_mobility_expansion_over_two_sublattices(ternary):
    """Phi_C = y_A y_Va Phi_A:Va + y_B y_Va Phi_B:Va + y_A y_C Phi_A:C + y_A y_B y_Va L (eq. 67, per sublattice)."""
    u = np.array([0.3, 0.45])  # u_B, u_C
    ya, yb, yc = 0.7, 0.3, 0.15
    yva = 1 - yc
    rtl = lambda d: R * T * np.log(d)
    phi_c = (ya * yva * (-90000 + rtl(2e-6)) + yb * yva * (-110000 + rtl(2e-6)) + ya * yc * -95000
             + ya * yb * yva * 12000)
    omega = ternary.mobility(u, T)
    assert R * T * np.log(R * T * omega[2]) == pytest.approx(phi_c, rel=1e-12)


def test_gibbs_duhem(ternary):
    u = np.array([0.3, 0.45])
    full = ternary.full_composition(u)
    mu = ternary.chemical_potentials(u, T)
    g = ternary._gibbs(full, T)[0]
    assert full @ mu == pytest.approx(g, rel=1e-12)


def test_fluxes_from_first_principles(ternary):
    """J'_k from eqs. (62)-(63) with finite-difference potential gradients, moved to the substitutional
    number-fixed frame, against -sum_j D_kj du_j/dz (all over V_S)."""
    def profile(z):
        return np.array([0.30 + 0.05 * np.tanh(z), 0.45 - 0.10 * np.tanh(0.6 * z + 0.3)])

    z0, dz = 0.2, 1e-5
    u = profile(z0)
    full = ternary.full_composition(u)
    du_dz = (profile(z0 + dz) - profile(z0 - dz)) / (2 * dz)
    dmu_dz = (ternary.chemical_potentials(profile(z0 + dz), T) - ternary.chemical_potentials(profile(z0 - dz), T)) / (2 * dz)
    omega = ternary.mobility(u, T)
    y_va = 1 - full[2] / 3
    j_lattice = -full * omega * dmu_dz  # components ordered B, A, C
    j_lattice[2] *= y_va
    s_sum = j_lattice[:2].sum()
    j_frame = j_lattice - full * s_sum  # substitutional elements and the interstitial alike
    d = ternary.interdiffusion_matrix(u, T)
    np.testing.assert_allclose(-d @ du_dz, j_frame[[0, 2]], rtol=1e-6)


def test_carbon_potential_matches_pycalphad_in_austenite():
    """mu_C and mu_FE of single-phase FCC_A1 against pycalphad.equilibrium, Fe-C database shipped with pycalphad."""
    path = os.path.join(os.path.dirname(pycalphad.__file__), "tests", "databases", "cfe_broshe.tdb")
    db = Database(path)
    mob = Database(path)  # thermodynamics only; mobilities are added for the constructor
    for host, el in (("FE", "C"), ("FE", "FE")):
        mob.add_parameter("MQ", "FCC_A1", [[host], ["VA"]], 0, -200000, diffusing_species=el)
    phase = InterstitialPhase(db, "FCC_A1", ["FE"], ["C"], mobility_db=mob)
    x_c = 0.03
    eq = equilibrium(db, ["FE", "C", "VA"], ["FCC_A1"], {v.T: T, v.P: 101325, v.N: 1, v.X("C"): x_c})
    mu_ref = dict(zip(eq.component.values, np.squeeze(eq.MU.values)))
    mu = phase.chemical_potentials(np.array([x_c / (1 - x_c)]), T)
    assert mu[0] == pytest.approx(mu_ref["FE"], abs=0.5)
    assert mu[1] == pytest.approx(mu_ref["C"], abs=0.5)


def test_u_fraction_conversion_and_limits(ternary):
    np.testing.assert_allclose(ternary.u_from_x(np.array([0.2, 0.6, 0.2])), [0.25, 0.25])  # order B, A, C
    with pytest.raises(ValueError, match="out of range"):
        ternary.interdiffusion_matrix(np.array([0.3, 3.5]), T)  # more C than interstitial sites
