"""Each worked example runs (at coarse resolution) and gives the result its docstring claims."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

import fe_cr_couple  # noqa: E402
import film_reaction  # noqa: E402
import ni_sn_kirkendall  # noqa: E402
import oxygen_uptake  # noqa: E402
import precipitation_table1  # noqa: E402
import surface_depletion  # noqa: E402
import thin_film_anneal  # noqa: E402
import two_phase_couple  # noqa: E402


def test_fe_cr_couple():
    out = fe_cr_couple.run(coarse=True)
    res = out["result"]
    inv = res.inventory()
    assert abs(inv[-1, 0] - inv[0, 0]) < 1e-10 * inv[0, 0]
    assert abs(res.x[-1, 0, 0] - 0.30) < 1e-3 and abs(res.x[-1, -1, 0] - 0.70) < 1e-3  # still semi-infinite
    # Cr is the faster diffuser in the Fe-rich alloys, so the lattice shifts toward the Cr-rich end, parabolically
    assert res.markers[-1, 0] > 0
    assert res.markers[-1, 0] / res.markers[1, 0] == pytest.approx(2.0, rel=3e-2)


def test_surface_depletion_matches_the_constant_d_estimate():
    """The Cr removed, 2 dx sqrt(D t / pi) for constant D (the integral of the erf flux), with D~_CrCr
    taken halfway between bulk and surface; D varies by 30% over the profile, so 10% agreement is expected."""
    out = surface_depletion.run(coarse=True)
    res, phase = out["result"], out["phase"]
    lost = res.inventory()[0, 0] - res.inventory()[-1, 0]
    d = phase.interdiffusion_matrix(np.array([[0.25, 0.02]]), surface_depletion.T)[0, 0, 0]
    t = surface_depletion.HOURS * 3600
    assert lost == pytest.approx(2 * 0.1 * np.sqrt(d * t / np.pi), rel=0.1)
    assert res.x[-1, :, 1].max() > 0.0205  # Ni is pushed uphill by the Cr gradient


def test_precipitation_example_runs_the_full_sequence():
    res = precipitation_table1.run(coarse=True)["result"]
    n = res.number_density
    assert 0 < np.argmax(n) < len(n) - 1
    assert res.matrix_fraction[-1] == pytest.approx(0.01, abs=1e-3)


def test_two_phase_example_agrees_between_models():
    out = two_phase_couple.run(coarse=True)
    shift_sharp = out["sharp"].interface[-1] - out["half"]
    shift_homog = out["alpha_thickness"] - out["half"]
    assert shift_homog == pytest.approx(shift_sharp, rel=0.15)


def test_thin_film_anneal_follows_the_temperature_schedule():
    """The reaction layer grows while hot and stops when cooled, and both interfaces sit at the
    pycalphad tie-lines of the current temperature (900 K mid-hold, 600 K at the end)."""
    from calphad_kinetics.precipitation import solubility_from_pycalphad

    out = thin_film_anneal.run(coarse=True)
    res, db = out["result"], out["db"]
    inv = res.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-10 * inv[0, 0]
    beta = res.thickness()[:, 1]
    t = res.times
    hold_end = np.searchsorted(t, 66.0)
    assert beta[hold_end] > 5 * beta[0]
    assert beta[-1] == pytest.approx(beta[hold_end], rel=0.05)  # little growth once cooled
    for i, temperature in ((np.searchsorted(t, 36.0), 900.0), (len(t) - 1, 600.0)):
        a, b_lo = solubility_from_pycalphad(db, ["A", "B"], "ALPHA", "BETA", "B", temperature, 0.1)
        b_hi, g = solubility_from_pycalphad(db, ["A", "B"], "BETA", "GAMMA", "B", temperature, 0.9)
        np.testing.assert_allclose([res.interface_left[i, 0, 0], res.interface_right[i, 0, 0],
                                    res.interface_left[i, 1, 0], res.interface_right[i, 1, 0]],
                                   [a, b_lo, b_hi, g], atol=2e-4)


def test_oxygen_uptake_follows_the_held_potential():
    """Ideal interstitial: y_O / (1 - y_O) = exp((mu_O - G_M:O + G_M:Va) / RT) at the surface, so the
    surface holds 0.0645 O per M at 900 K and 0.0360 after cooling to 600 K; cooling then drives some
    oxygen back out."""
    from calphad_kinetics import R

    out = oxygen_uptake.run(coarse=True)
    res, surface = out["result"], out["surface"]
    for t, temp in ((40.0, 900.0), (80.0, 600.0)):
        ratio = np.exp((oxygen_uptake.MU_O + 100000.0) / (R * temp))
        assert surface.face(t, np.array([0.01]), 1)[0][0] == pytest.approx(ratio / (1 + ratio), rel=1e-9)
    uptake = res.inventory()[:, 0]
    hold_end = list(res.times).index(70.0)
    assert np.all(np.diff(uptake[: hold_end + 1]) > 0)
    assert uptake[-1] < uptake[hold_end]  # outgassing as the surface solubility falls


def test_ni_sn_kirkendall():
    out = ni_sn_kirkendall.run(coarse=True)
    split = out["x2_over_x1"]
    assert np.all(np.diff(split) > 0) and 0.5 < split[0] and split[-1] < 2.0  # Table 2.1 limits
    assert ni_sn_kirkendall.marker_ratio(9.5) == pytest.approx(1.839, abs=1e-3)


def test_film_reaction_ends_with_the_film_inventory_in_ms():
    res = film_reaction.run(coarse=True)["result"]
    assert [name for _, name in res.consumed] == ["M", "M2S"]
    metal, ms = film_reaction.PHASES[0], film_reaction.PHASES[2]
    expected = film_reaction.FILM * metal.concentrations()[0] / ms.concentrations()[0]
    assert res.thickness["MS"][-1] == pytest.approx(expected, rel=1e-6)
