"""Nucleation of a new phase at an interface, against d'Heurle, J. Mater. Res. 3 (1988) 167: the specific thickness
of eq. (12) and Table II, the molar volumes and driving forces of his nickel silicide examples (pp. 174-176, Table
III), the equal-energy lens of eqs. (14)-(16) and the hemisphere condition of eq. (19); and against the package's
own grain-boundary lens of Clemm and Fisher, built independently."""

import numpy as np
import pytest

from janus_kinetics.interface_nucleation import interface_nucleus, molar_volume, specific_thickness
from janus_kinetics.multicomponent_precipitation import junction_barrier_factor

CAL = 4.184  # J
ERG_PER_CM2 = 1e-3  # J/m^2
CAL_PER_CM3 = CAL * 1e6  # J/m^3
NM3 = 1e-27


def test_table_2_specific_thicknesses():
    """d_s = Delta sigma / Delta h for the seven pairs of Table II, in nm from erg/cm^2 and cal/cm^3."""
    table = [(2000, 500, 0.06), (2000, 1000, 0.12), (2000, 2000, 0.24), (500, 500, 0.24), (500, 1000, 0.48),
             (500, 2000, 0.96), (100, 500, 1.20)]
    for dh, ds, printed in table:
        d = specific_thickness(ds * ERG_PER_CM2, dh * CAL_PER_CM3) * 1e9
        assert d == pytest.approx(printed, abs=0.006)


def test_nickel_silicide_examples():
    """Molar volumes from the unit cells: Ni2Si (orthorhombic 0.499 x 0.372 x 0.701 nm, Z = 4) 19.6 cm^3, Ni3Si2
    (1.223 x 1.081 x 0.6924 nm, Z = 16) 34.44, NiSi2 (cubic 0.541 nm, Z = 4) 23.8, as printed. Ni2Si's heat of
    formation, -34.1 kcal/mol, then gives Delta h = 1740 cal/cm^3, and Ni3Si2's -8.7 kcal per 2 Ni3Si2 about 126,
    'of the order of 135'. d_s for Ni2Si at 1000 erg/cm^2 comes to 0.137 nm where p. 174 prints 0.15 (Table II's own
    0.12 at 2000 cal/cm^3 scales to 0.138). For NiSi the paper prints 285 cal/cm^3; its own inputs (-8.7 kcal for
    Ni2Si + Si = 2 NiSi, 14.52 cm^3/mol) give 300; the printed 285 gives d_s = 0.84 nm, printed 0.85."""
    v_ni2si = molar_volume(0.499 * 0.372 * 0.701 * NM3, 4)
    assert v_ni2si * 1e6 == pytest.approx(19.64, rel=0.005)
    assert molar_volume(1.223 * 1.081 * 0.6924 * NM3, 16) * 1e6 == pytest.approx(34.44, rel=0.002)
    assert molar_volume(0.541**3 * NM3, 4) * 1e6 == pytest.approx(23.8, rel=0.005)
    dh_ni2si = 34.1e3 / 19.64  # cal/cm^3
    assert dh_ni2si == pytest.approx(1740, rel=0.005)
    assert specific_thickness(1000 * ERG_PER_CM2, dh_ni2si * CAL_PER_CM3) * 1e9 == pytest.approx(0.137, abs=0.002)
    dh_ni3si2 = 8.7e3 / (2 * 34.44)
    assert dh_ni3si2 == pytest.approx(135, rel=0.1)
    assert 8.7e3 / (2 * 14.52) == pytest.approx(300, rel=0.005)
    assert specific_thickness(1000 * ERG_PER_CM2, 285 * CAL_PER_CM3) * 1e9 == pytest.approx(0.839, abs=0.001)


def test_table_3_nisi2_driving_force():
    """Table III: Delta G' = G(NiSi2) - G(NiSi) - G(Si) at 300, 700, 1100 K, 1.30, 1.71, 2.35 kcal per mole of NiSi2;
    over 23.8 cm^3 that is 55 to 99 cal/cm^3, 'from about 50 at room temperature to about 100 close to 800 C'."""
    rows = [(24.6, 1.35, 27.25, 1.30), (31.05, 4.18, 36.94, 1.71), (40.46, 8.42, 51.23, 2.35)]
    dg = []
    for nisi, si, nisi2, printed in rows:
        assert nisi2 - (nisi + si) == pytest.approx(printed, abs=0.005)
        dg.append(printed * 1e3 / 23.8)
    assert dg[0] == pytest.approx(50, rel=0.12) and dg[-1] == pytest.approx(100, rel=0.02)


def test_equal_energies_give_eqs_14_to_16():
    """All three energies sigma: the caps meet the interface at 60 degrees (120 between all three interfaces), the
    nucleus adds (5 pi / 4) R^2 of interface and holds (5 pi / 12) R^3, R* = 2 sigma / Delta h, its height normal to
    the interface is R*, and a sphere of the same volume has radius (5/16)^(1/3) R* = 0.68 R*, 'about 0.7 R*'."""
    sigma, dh = 0.5, 4e9
    nuc = interface_nucleus(sigma, sigma, sigma, dh)
    r = 2 * sigma / dh
    assert nuc.angle_a == pytest.approx(np.pi / 3, rel=1e-12) and nuc.angle_b == pytest.approx(np.pi / 3, rel=1e-12)
    assert nuc.cap_radius_a == pytest.approx(r, rel=1e-12)
    assert nuc.area_energy == pytest.approx(sigma * 5 * np.pi / 4 * r**2, rel=1e-12)
    assert nuc.volume == pytest.approx(5 * np.pi / 12 * r**3, rel=1e-12)
    assert nuc.height == pytest.approx(r, rel=1e-12)
    assert (3 * nuc.volume / (4 * np.pi)) ** (1 / 3) / r == pytest.approx(0.68, abs=0.005)


def test_critical_lens_is_a_stationary_point():
    """For any admissible energies the barrier is a third of the interfacial energy added, as for any critical
    nucleus of fixed shape, and scales as sigma^3 / Delta g^2 (eq. 7)."""
    for s1, s2, s3 in [(0.4, 0.5, 0.6), (0.9, 0.5, 0.5), (0.2, 0.5, 0.6), (0.5, 0.3, 0.7)]:
        nuc = interface_nucleus(s1, s2, s3, 3e9)
        assert nuc.barrier == pytest.approx(nuc.area_energy / 3, rel=1e-12)
        twice = interface_nucleus(2 * s1, 2 * s2, 2 * s3, 3e9)
        assert twice.barrier / nuc.barrier == pytest.approx(8.0, rel=1e-12)
        assert interface_nucleus(s1, s2, s3, 6e9).barrier / nuc.barrier == pytest.approx(0.25, rel=1e-12)


def test_hemisphere_condition_eq_19():
    """A cap is a hemisphere when sigma_1^2 + sigma_2^2 = sigma_3^2: with 0.3, 0.4, 0.5 J/m^2 the cap on the A side
    meets the old interface at 90 degrees, its rim radius equals its cap radius and its height that radius."""
    nuc = interface_nucleus(0.3, 0.4, 0.5, 2e9)
    assert nuc.angle_a == pytest.approx(np.pi / 2, rel=1e-12)
    assert nuc.rim_radius == pytest.approx(nuc.cap_radius_a, rel=1e-12)
    # by hand: the A cap, radius 2 (0.4) / Delta g, is a hemisphere; on the B side cos theta = (0.25 + 0.09 - 0.16)
    # / (2 0.5 0.3) = 0.6, so that cap, radius 2 (0.5) / Delta g, rises 0.4 of its radius: height 1.2 / Delta g, and
    # both caps meet the rim at 0.8 / Delta g
    assert np.cos(nuc.angle_b) == pytest.approx(0.6, rel=1e-12)
    assert nuc.height == pytest.approx(1.2 / 2e9, rel=1e-12)
    assert nuc.cap_radius_b * np.sin(nuc.angle_b) == pytest.approx(0.8 / 2e9, rel=1e-12)


def test_symmetric_lens_matches_clemm_and_fisher():
    """With sigma_2 = sigma_3 the interface lens is the grain-boundary lens of Clemm and Fisher (Acta Metall. 3
    (1955) 70), coded separately in the precipitation module with k = sigma_1 / (2 sigma): the barrier over the
    homogeneous one, 16 pi sigma^3 / (3 Delta g^2), is their factor for every k."""
    sigma, dg = 0.5, 2e9
    homogeneous = 16 * np.pi * sigma**3 / (3 * dg**2)
    for k in (0.05, 0.3, 0.5, 0.8, 0.95):
        nuc = interface_nucleus(2 * k * sigma, sigma, sigma, dg)
        assert nuc.barrier / homogeneous == pytest.approx(junction_barrier_factor("boundary", k), rel=1e-10)


def test_driving_force_falls_sixfold_from_ab_to_ab125():
    """p. 179: if AB releases 20 kcal/mol over a volume 2v and AB1.25 24 kcal/mol over 2.25v, forming AB1.25 from AB
    releases 4 kcal over 2.25v, 5.6 times less per volume than forming AB ('some six times'), and at fixed
    interfacial energy the barrier is 32 times higher (the paper rounds to 36 from 6^2)."""
    first, second = 20 / 2.0, (24 - 20) / 2.25
    assert first / second == pytest.approx(5.625, rel=1e-12)
    a = interface_nucleus(0.5, 0.5, 0.5, first * 1e9)
    b = interface_nucleus(0.5, 0.5, 0.5, second * 1e9)
    assert b.barrier / a.barrier == pytest.approx(5.625**2, rel=1e-12)


def test_input_checks():
    with pytest.raises(ValueError, match="wets"):
        interface_nucleus(1.0, 0.4, 0.5, 1e9)
    with pytest.raises(ValueError, match="17"):
        interface_nucleus(0.2, 0.3, 0.6, 1e9)
    with pytest.raises(ValueError, match="positive"):
        interface_nucleus(0.5, 0.5, 0.5, -1e9)
    with pytest.raises(ValueError, match="driving force"):
        specific_thickness(0.5, 0.0)
    with pytest.raises(ValueError, match="formula unit"):
        molar_volume(1e-28, 0)
