"""The fcc Ni-superalloy mobility database of Campbell, Boettinger and Kattner, Acta Mater. 50 (2002) 775."""

import sys
from pathlib import Path

import numpy as np
import pytest

from calphad_kinetics import MobilityModel, R
from calphad_kinetics._lattice import SubstitutionalLattice
from calphad_kinetics.datasets import campbell2002_fcc_ni

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import make_campbell2002  # noqa: E402

ELEMENTS = make_campbell2002.ELEMENTS


@pytest.fixture(scope="module")
def model():
    db = campbell2002_fcc_ni()
    return MobilityModel(db, SubstitutionalLattice.from_database(db, "FCC_A1", list(ELEMENTS), dependent="NI"))


def composition(model, fractions: dict) -> np.ndarray:
    x = np.full(len(model.lattice.elements), 1e-12)
    for el, value in fractions.items():
        x[model.lattice.elements.index(el)] = value
    return x / x.sum()


def mq(model, element, x, T):
    return model.parameters(x, T)[1][model.lattice.elements.index(element)]


def test_shipped_file_matches_the_generator():
    assert make_campbell2002.OUT.read_text(encoding="utf-8") == make_campbell2002.render()


# Table 4 (p. 785): Q (J/mol) of self-diffusion and -Q / (R T_M) at the SGTE fcc melting point
TABLE4 = {"NI": (-287000, 1728, 20.0), "AL": (-142000, 933.5, 18.3), "CR": (-235000, 1475, 19.2),
          "CO": (-286175, 1768, 19.5), "HF": (-235350, 1952, 14.5), "MO": (-254975, 1740, 17.6),
          "RE": (-382950, 3084, 14.9), "TA": (-268253, 1416, 22.8), "TI": (-256900, 900, 34.3),
          "W": (-311420, 2229, 16.8)}


@pytest.mark.parametrize("element", sorted(TABLE4))
def test_self_diffusion_activation_energies_of_table4(model, element):
    q, t_m, ratio = TABLE4[element]
    x = composition(model, {element: 1.0})
    m1, m2 = mq(model, element, x, 1000.0), mq(model, element, x, 2000.0)
    constant = 2 * m1 - m2  # MQ = a + b T, so a = 2 MQ(1000) - MQ(2000)
    assert constant == pytest.approx(q, abs=5.0)  # Table 4 rounds W to -311,420
    assert -constant / (8.314 * t_m) == pytest.approx(ratio, abs=0.06)


def test_mo_solute_diffusivity_in_ni_at_1382K(model):
    """p. 781: 'the calculated diffusivity is 5.50 x 10^-15 m^2/s' for Mo in Ni at 1382 K.

    Table 2 gives 5.42e-15 at infinite dilution, 1.5 % lower. The paper does not
    say at what Mo content it evaluated the number; about 1.7 at.% Mo, through the
    Mo end member, would give 5.50e-15. The check therefore allows 2 %.
    """
    x = composition(model, {"NI": 1.0})
    d = np.exp(mq(model, "MO", x, 1382.0) / (R * 1382.0))  # D* = RT M = exp(MQ / RT), eqs. (4) and (6)
    assert d == pytest.approx(5.42e-15, rel=1e-3)
    assert d == pytest.approx(5.50e-15, rel=2e-2)


def test_cr_in_cobalt_activation_energy(model):
    """p. 781: 'For Cr diffusion in gamma-Co, the assessed activation energy of 247 kJ/mole'."""
    x = composition(model, {"CO": 1.0})
    assert -(2 * mq(model, "CR", x, 1000.0) - mq(model, "CR", x, 2000.0)) == pytest.approx(247e3, rel=2e-3)


def test_hf_in_ni_against_fig5(model):
    """Fig. 5 (p. 783), assessed line read by eye: log10 D = -13.0 at 1/T = 7e-4 and -16.9 at 1e-3 (+-0.15)."""
    x = composition(model, {"NI": 1.0})
    for inv_t, expected in ((7e-4, -13.0), (1e-3, -16.9)):
        T = 1 / inv_t
        assert np.log10(np.exp(mq(model, "HF", x, T) / (R * T))) == pytest.approx(expected, abs=0.15)


def test_estimated_end_members_follow_the_rule_of_p778(model):
    """Mo in pure Al is not in Table 2, so it takes the value of Mo in pure Ni; Mo in Co is listed."""
    for host in ("AL", "CR", "TA"):
        assert mq(model, "MO", composition(model, {host: 1.0}), 1300.0) == pytest.approx(
            mq(model, "MO", composition(model, {"NI": 1.0}), 1300.0), rel=1e-9)
    assert mq(model, "MO", composition(model, {"CO": 1.0}), 1300.0) != pytest.approx(
        mq(model, "MO", composition(model, {"NI": 1.0}), 1300.0), rel=1e-3)


# Fig. 4(b,c) (p. 783), assessed lines read by eye at 1/T = 7.4e-4 K^-1 (+-0.05 in log10 D):
# Ni and W tracer diffusivities in Ni-W alloys with x_W = 0.017, 0.053, 0.092
FIG4 = {"NI": (-14.83, -15.07, -15.33), "W": (-15.52, -15.66, -15.80)}


@pytest.mark.parametrize("element", ["NI", "W"])
def test_ni_w_tracer_lines_of_fig4(model, element):
    """The spacing between the three alloys comes from the Ni-W interaction parameters."""
    T = 1 / 7.4e-4
    got = [np.log10(np.exp(mq(model, element, composition(model, {"NI": 1 - xw, "W": xw}), T) / (R * T)))
           for xw in (0.017, 0.053, 0.092)]
    np.testing.assert_allclose(got, FIG4[element], atol=0.05)


@pytest.mark.skip(reason="Campbell et al. Table 7 (p. 788) compares measured Ni-Al-Cr interdiffusion matrices at "
                         "1100 C with ones calculated from these mobilities and the thermodynamics of Kattner [29] or "
                         "Saunders [27]; neither is public, and no citable open fcc Ni-Al-Cr assessment is in hand")
def test_ni_al_cr_matrices_against_table7():
    pass
