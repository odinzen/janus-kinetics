"""Each worked example runs (at coarse resolution) and gives the result its docstring claims."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

import fe_cr_couple  # noqa: E402
import interconnect_electromigration  # noqa: E402
import chromia_scale  # noqa: E402
import film_reaction  # noqa: E402
import ni_sn_kirkendall  # noqa: E402
import oxygen_uptake  # noqa: E402
import precipitation_table1  # noqa: E402
import silicon_oxidation  # noqa: E402
import aluminium_heat_treatment  # noqa: E402
import solidify_and_homogenize  # noqa: E402
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
    from janus_kinetics.precipitation import solubility_from_pycalphad

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
    from janus_kinetics import R

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


def test_chromia_scale_gives_the_papers_thicknesses():
    """Hallstrom et al. (2013), section 4.5.2: 340 nm at 625 C (100 nm grains) and 660 nm at
    700 C (200 nm grains) after 24 h; the coarse quadrature still lands within 2 %."""
    out = chromia_scale.run(coarse=True)
    for label, paper in (("625 C, 100 nm grains", 340e-9), ("700 C, 200 nm grains", 660e-9)):
        assert out[label].thickness(86400.0) == pytest.approx(paper, rel=0.02)


def test_solidify_and_homogenize():
    """The cast profile rises from the first solid at the axis to the last at the arm edge, the anneal keeps
    every element (cylinder inventory), and the Cr segregation index falls throughout."""
    from janus_kinetics.solidification import segregation_index

    out = solidify_and_homogenize.run(coarse=True)
    res, scheil = out["anneal"], out["scheil"]
    edges, solid = scheil.increments()
    first = solid[np.argmax(np.diff(edges) > 1e-6)]
    assert res.x[0, 0, 0] == pytest.approx(first[0], rel=0.01)
    assert res.x[0, -1, 0] > res.x[0, 0, 0] and res.x[0, -1, 1] > res.x[0, 0, 1]
    inv = res.inventory()
    assert np.max(np.abs(inv - inv[0])) < 1e-10 * np.max(inv[0])
    assert np.all(np.diff(segregation_index(res.x[:, :, 1])) < 0)


def test_interconnect_electromigration_reaches_blechs_steady_state():
    """Korhonen et al.'s numbers (G = 16.5 MPa/um, kappa = 2.6 um^2/h) in a 50 um line: the cathode end follows their
    eq. (13), 212 MPa at 50 h, and reaches Blech's G L / 2 = 412 MPa by 1000 h."""
    from janus_kinetics import finite_line_stress, steady_stress

    res = interconnect_electromigration.run(coarse=True)["result"]
    ex = interconnect_electromigration
    assert res.stress[2][0] == pytest.approx(finite_line_stress(res.x[0], 50 * 3600, ex.LENGTH, ex.KAPPA, ex.G), rel=0.02)
    np.testing.assert_allclose(res.stress[-1], steady_stress(res.x, ex.LENGTH, ex.G), rtol=0, atol=0.01 * ex.G * ex.LENGTH)


def test_silicon_oxidation_uses_deal_and_groves_constants():
    """Every curve satisfies x^2 + A x = B (t + tau) with its table's constants, and the second oxidation step starts
    from the oxide the first left (0.090 um after 30 min of dry oxygen at 1100 C)."""
    ex = silicon_oxidation
    out = ex.run(coarse=True)
    t = out["t"]
    for temp in ex.WET:
        a, b = ex.constants(ex.WET, temp)
        x = out["wet"][temp]
        np.testing.assert_allclose(x * x + a * x, b * t, rtol=1e-10, atol=1e-24)
    a, b = ex.constants(ex.DRY, 1100)
    xi = ex.X_I * ex.UM
    tau = (xi * xi + a * xi) / b
    x = out["dry"][1100]
    np.testing.assert_allclose(x * x + a * x, b * (t + tau), rtol=1e-10)
    switch = ex.two_step(np.array([0.5 * ex.HOUR]))[0]
    assert switch == pytest.approx(0.0896e-6, rel=1e-3)
    aw, bw = ex.constants(ex.WET, 1000)
    end = out["two_step"][-1]
    assert end * end + aw * end == pytest.approx(switch * switch + aw * switch + bw * 2 * ex.HOUR, rel=1e-10)


def test_aluminium_heat_treatment_ages_and_reheats():
    """Coarse run of the AA 6082 example: about 0.008 of Mg2Si after 1e4 s at 180 C, as Myhr and Grong state; the
    three tempers start apart (under below peak below over-aged), meet within 0.0002 by 300 C, never exceed the
    equilibrium fraction by more than the capillary margin, and are gone at the 500 C solvus."""
    out = aluminium_heat_treatment.run(coarse=True)
    ageing = out["ageing"]
    k = np.argmin(np.abs(np.log(ageing.times / 1e4)))
    assert ageing.volume_fraction[k] == pytest.approx(0.008, rel=0.03)
    f = out["heating"]
    assert f["under-aged"][0] < f["peak-aged"][0] < f["over-aged"][0]
    at300 = [v[out["temps"] == 300][0] for v in f.values()]
    assert max(at300) - min(at300) < 0.0002
    for v in f.values():
        assert np.all(v <= out["equilibrium"] + 1e-4)
        assert v[out["temps"] >= 505].max() < 1e-6
