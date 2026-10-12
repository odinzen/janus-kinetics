"""Precipitation along heat treatments, against Myhr and Grong, Acta Mater. 48 (2000) 1605, with the AA 6082 inputs of
their Table 2. Figure values were read at 400 dpi on axes calibrated to the plot frames: Figs. 3a-b, 5a-b and the
sigma = 0.05 markers of Fig. 9a traced automatically, the rest read off the renders."""

import dataclasses
import functools

import numpy as np
import pytest

from janus_kinetics import TemperatureSchedule
from janus_kinetics.heat_treatment import (
    aa6082,
    log_normal_distribution,
    lsw_distribution,
    solve_heat_treatment,
    stretched_radii,
)
from janus_kinetics.thermo import R

AGEING = 453.15  # 180 C
ALLOY = aa6082()


@functools.lru_cache(maxsize=None)
def _radii():
    return stretched_radii(0.02e-9, 15e-9, 1.005, 5e-6)


@functools.lru_cache(maxsize=None)
def _aged(t, gamma=0.2):
    return solve_heat_treatment(dataclasses.replace(ALLOY, gamma=gamma), AGEING, t, radii=_radii())


def test_table_2_gives_the_starting_values_of_the_figures():
    """At 180 C and C = C_0: the barrier (A0/RT)^3 / ln(C/C_e)^2 is 'about 3' (p. 1609), the nucleation rate
    5.2e18 against Fig. 3a's plateau of 4.9e18, the equilibrium fraction 0.0099 as on Fig. 9a, and the solvus
    of the alloy 500 C, which puts 480 C 'about 20 C below' it (p. 1612)."""
    ce = ALLOY.solubility(AGEING)
    barrier = (ALLOY.a0 / (R * AGEING)) ** 3 / np.log(ALLOY.c0 / ce) ** 2
    assert barrier == pytest.approx(3.0, abs=0.05)
    assert ALLOY.nucleation_rate(ALLOY.c0, AGEING) == pytest.approx(4.9e18, rel=0.07)
    assert ALLOY.equilibrium_fraction(AGEING) == pytest.approx(0.0099, abs=0.0001)
    solvus = ALLOY.qs / (R * np.log(ALLOY.cs / ALLOY.c0))
    assert solvus - 273.15 == pytest.approx(500.0, abs=1.0)
    assert ALLOY.equilibrium_fraction(solvus + 1) == 0.0
    assert ALLOY.critical_radius(ALLOY.c0, AGEING) == pytest.approx(
        2 * ALLOY.gamma * ALLOY.molar_volume / (R * AGEING * np.log(ALLOY.c0 / ce)), rel=1e-12)


def test_eq_4_holds_its_sign_for_the_smallest_particles():
    """Below 2 sigma V_m / (RT ln(C_p / C_e)), 0.43 nm at 180 C, C_i of eq. (5) passes C_p and eq. (4) would let
    the particle grow; it dissolves instead."""
    r = np.array([0.1e-9, 0.3e-9, 0.42e-9, 0.44e-9, 0.6e-9, 0.8e-9])
    v = ALLOY.growth_rate(r, ALLOY.c0, AGEING)
    assert np.all(v[:5] < 0) and v[-1] < 0  # all below r* = 0.81 nm
    assert ALLOY.growth_rate(1e-9, ALLOY.c0, AGEING) > 0


# Fig. 3, ageing at 180 C: (t s, mean radius nm) of panel (b), (t s, particles per m^3) of panel (a)
FIG3_RADIUS = [(1.0e3, 1.496), (3.0e3, 2.427), (1.0e4, 3.775), (3.0e4, 4.214), (1.0e5, 4.400), (1.0e6, 5.321),
               (3.0e6, 6.241), (8.5e6, 7.72)]
FIG3_NUMBER = [(1e2, 5.0e20), (1e3, 4.93e21), (1e4, 2.886e22), (1e5, 2.41e22), (1e6, 1.428e22), (3e6, 8.89e21),
               (8.5e6, 4.62e21)]


def test_ageing_at_180C_follows_fig_3():
    """Mean radius within 3 % and number density within 5 % from 100 s to 3e6 s, and f = 0.008 at 1e4 s, the
    peak-aged state of p. 1612. The last number density, at 8.5e6 s, depends on the class width: uniform classes
    of 0.015 nm (the paper's 1000 over 15 nm) give -3.4 %, 0.0075 nm +0.7 %, 0.00375 nm +3.3 %, the 0.02 nm used
    here -5 %, as the upwind scheme spreads the distribution a little; it is held to 8 %."""
    times = sorted({t for t, _ in FIG3_RADIUS + FIG3_NUMBER})
    res = solve_heat_treatment(ALLOY, AGEING, 8.5e6, radii=_radii(), save_times=times)
    at = dict(zip(np.round(res.times, 3), range(len(res.times))))
    for t, r in FIG3_RADIUS:
        assert res.mean_radius[at[t]] * 1e9 == pytest.approx(r, rel=0.03), t
    for t, n in FIG3_NUMBER:
        assert res.number_density[at[t]] == pytest.approx(n, rel=0.05 if t < 8e6 else 0.08), t
    assert res.volume_fraction[at[1e4]] == pytest.approx(0.008, rel=0.02)
    np.testing.assert_allclose(res.matrix_content * (1 - res.volume_fraction) + ALLOY.cp * res.volume_fraction,
                               ALLOY.c0, rtol=1e-12)


FIG5_F = [(0.1, 0.00566), (0.2, 0.00501), (0.3, 0.00480), (0.5, 0.00483), (0.8, 0.00505), (1.2, 0.00530),
          (1.59, 0.00545)]
FIG5_RADIUS = [(0.1, 39.36), (0.2, 40.40), (0.5, 44.90), (0.8, 48.53), (1.59, 54.93)]


def _upquench(courant=1.0, tolerance=0.02):
    ts = np.round(np.arange(0.01, 1.6001, 0.01), 2)
    return solve_heat_treatment(ALLOY, 653.15, 1.6, radii=_radii(), save_times=ts, initial=_aged(1e4).state(),
                                courant=courant, tolerance=tolerance)


def test_upquench_to_380C_follows_fig_5():
    """After 1e4 s at 180 C and a jump to 380 C: volume fraction within 1.5 % of Fig. 5b, mean radius within 2 %
    of Fig. 5a, and dissolution ends (r* falls below the mean radius) at 0.32 s, where the paper draws its dotted
    line (0.317 s). Halving the step limits changes nothing by more than 0.5 %."""
    res = _upquench()
    at = dict(zip(np.round(res.times, 2), range(len(res.times))))
    for t, f in FIG5_F:
        assert res.volume_fraction[at[round(t, 2)]] == pytest.approx(f, rel=0.015), t
    for t, r in FIG5_RADIUS:
        assert res.mean_radius[at[round(t, 2)]] * 1e10 == pytest.approx(r, rel=0.02), t
    end = res.times[np.nonzero(res.critical_radius < res.mean_radius)[0][0]]
    assert end == pytest.approx(0.317, abs=0.015)
    fine = _upquench(courant=0.5, tolerance=0.01)
    np.testing.assert_allclose(fine.volume_fraction, res.volume_fraction, rtol=0.005)
    np.testing.assert_allclose(fine.mean_radius, res.mean_radius, rtol=0.005)


def test_starting_distributions_have_the_heights_of_fig_6():
    """Fig. 6 states r = 38 A and f = 0.008 for every curve. Built that way, the peaks are 1.74e31 (LSW) and
    1.69e31, 1.02e31, 6.6e30 and 4.4e30 for b = 0.4-1.0, within 4 % of the heights drawn; but every drawn curve
    sits about 10 % lower in radius (the LSW peak at 39 A, not 43; its cut-off at 49.5 A, not 1.5 x 38 = 57),
    and a number mean of 34 A follows from the b = 0.4 markers. Fig. 7 starts at f = 0.008 and r = 3.8e-9 m, so
    the runs below take the stated values and the drawing is taken to be off in radius."""
    r = _radii()
    drawn = {"lsw": 1.74e31, 0.4: 1.64e31, 0.6: 0.997e31, 0.8: 6.45e30, 1.0: 4.31e30}
    for kind, height in drawn.items():
        n = lsw_distribution(r, 38e-10, 0.008) if kind == "lsw" else log_normal_distribution(r, 38e-10, 0.008, kind)
        w = np.diff(np.concatenate([[r[0] / 2], (r[1:] + r[:-1]) / 2, [1.5 * r[-1] - 0.5 * r[-2]]]))
        assert n @ (w * r) / (n @ w) == pytest.approx(38e-10, rel=0.005)
        assert n @ (w * 4 * np.pi / 3 * r**3) == pytest.approx(0.008, rel=1e-12)
        assert n.max() == pytest.approx(height, rel=0.04), kind


@functools.lru_cache(maxsize=None)
def _at_480(kind):
    r = _radii()
    start = lsw_distribution(r, 38e-10, 0.008) if kind == "lsw" else log_normal_distribution(r, 38e-10, 0.008, kind)
    ts = np.logspace(-5, 3, 81)
    return solve_heat_treatment(ALLOY, 753.15, 1e3, radii=r, save_times=ts, initial=start, dt=1e-7)


def test_upquench_to_480C_follows_fig_7():
    """At 480 C the LSW and b = 0.4 starts dissolve (Fig. 7b: below 1e14 particles per m^3 by 0.04 and 0.29 s);
    the broader ones keep their largest particles, which grow back to the equilibrium fraction, 0.00168 at
    1000 s on Fig. 7a. The b = 1.0 run bottoms out near f = 0.00035 at 0.6 s; b = 0.6 levels off at 1.65e17
    particles per m^3 (ours 1.5e17) with a mean radius of 1.38e-7 m at 1000 s. The b = 0.4 start leaves one
    particle in 1e10 from its far tail; they grow to 0.85 um but hold f < 1e-5, invisible on the figure."""
    lsw, narrow = _at_480("lsw"), _at_480(0.4)
    for res, gone in ((lsw, 0.04), (narrow, 0.29)):
        below = res.times[np.nonzero(res.number_density < 1e14)[0][0]]
        assert gone / 2 < below < gone * 2.5
    assert np.all(narrow.volume_fraction[narrow.times >= 0.1] < 1e-5)
    assert lsw.number_density[-1] == 0.0
    broad = _at_480(1.0)
    i = np.argmin(broad.volume_fraction)
    assert broad.volume_fraction[i] == pytest.approx(0.00035, rel=0.15) and 0.2 < broad.times[i] < 2.0
    for kind in (0.6, 0.8, 1.0):
        assert _at_480(kind).volume_fraction[-1] == pytest.approx(0.00168, rel=0.05)
    mid = _at_480(0.6)
    plateau = mid.number_density[(mid.times >= 10) & (mid.times <= 100)]
    np.testing.assert_allclose(plateau, 1.65e17, rtol=0.15)
    assert mid.mean_radius[-1] == pytest.approx(1.38e-7, rel=0.05)


HEATING = TemperatureSchedule([0, 400], [AGEING, AGEING + 400])


def test_heating_at_1C_per_s_follows_fig_8():
    """Under-, peak- and over-aged material (5e3, 1e4 and 4e4 s at 180 C) heated at 1 C/s: mean radii start at
    3.1, 3.9 and 4.4 nm on Fig. 8b and all three reach 3.65e-8 m near 488 C before the particles dissolve at the
    500 C solvus; the under-aged f starts at 0.0035 on Fig. 8a and all three meet by 300 C."""
    temps = np.arange(180.0, 506.0, 2.0)
    runs = [solve_heat_treatment(ALLOY, HEATING, 325.0, radii=_radii(), save_times=temps - 180,
                                 initial=_aged(t).state()) for t in (5e3, 1e4, 4e4)]
    radius = [res.mean_radius[:temps.size] for res in runs]
    fraction = [res.volume_fraction[:temps.size] for res in runs]
    for rb, f, r0 in zip(radius, fraction, (3.11e-9, 3.9e-9, 4.41e-9)):
        assert rb[0] == pytest.approx(r0, rel=0.05)
        k = np.nanargmax(rb)
        assert rb[k] == pytest.approx(3.65e-8, rel=0.06) and temps[k] == pytest.approx(488, abs=10)
        assert f[temps == 504][0] < 1e-6
    assert fraction[0][0] == pytest.approx(0.0035, rel=0.05)
    at300 = [f[temps == 300][0] for f in fraction]
    assert max(at300) - min(at300) < 0.00015


# Fig. 9a markers (C, f): sigma = 0.05, and sigma = 0.8 where its asterisks could be read
FIG9_005 = [(210, 0.00796), (230, 0.00807), (250, 0.00853), (270, 0.00914), (290, 0.00923), (310, 0.00890),
            (330, 0.00849), (350, 0.00801), (370, 0.00743), (390, 0.00676), (410, 0.00593), (430, 0.00496),
            (450, 0.00376), (470, 0.00238), (490, 0.00079)]
FIG9_08 = [(240, 0.00638), (270, 0.00334), (300, 0.00308), (370, 0.00527), (380, 0.00512), (400, 0.00464),
           (420, 0.00395), (440, 0.00305), (460, 0.00192), (480, 0.00065)]


def test_interfacial_energy_on_heating_follows_fig_9():
    """The peak-aged state heated at 1 C/s with sigma set to 0.05 or 0.8 J/m^2: within 0.0003 of every sigma = 0.05
    marker and 0.00012 of the readable sigma = 0.8 ones. With sigma = 0.8 most particles dissolve and the rest
    grow back; our minimum is 0.0029 at 280-290 C against 0.0025 at 290 C drawn, and the regrowth runs some 8 C behind the
    figure (0.0039 against 0.0049 at 320 C), unchanged by halving the step limits or the class width."""
    temps = np.arange(190.0, 491.0, 10.0)
    peak = _aged(1e4).state()
    for gamma, marks, tol in ((0.05, FIG9_005, 0.0003), (0.8, FIG9_08, 0.00012)):
        res = solve_heat_treatment(dataclasses.replace(ALLOY, gamma=gamma), HEATING, 310.0, radii=_radii(),
                                   save_times=temps - 180, initial=peak)
        f = dict(zip(temps, res.volume_fraction))
        for T, ref in marks:
            assert f[T] == pytest.approx(ref, abs=tol), (gamma, T)
        if gamma == 0.8:
            dip = {T: v for T, v in f.items() if T <= 350}
            assert min(dip.values()) == pytest.approx(0.00254, abs=0.0005) and min(dip, key=dip.get) in (280, 290)
            assert f[340] == pytest.approx(0.00556, abs=0.0002)  # the top of the regrowth, read off the render


def test_input_checks():
    with pytest.raises(ValueError, match="c0 < cp"):
        dataclasses.replace(ALLOY, c0=70.0)
    with pytest.raises(ValueError, match="increasing"):
        solve_heat_treatment(ALLOY, AGEING, 1.0, radii=[1e-9, 0.5e-9, 2e-9])
    with pytest.raises(ValueError, match="one non-negative value per class"):
        solve_heat_treatment(ALLOY, AGEING, 1.0, radii=_radii(), initial=np.zeros(3))
    with pytest.raises(ValueError, match="more solute than the alloy"):
        solve_heat_treatment(ALLOY, AGEING, 1.0, radii=_radii(), initial=lsw_distribution(_radii(), 4e-9, 0.02))
    with pytest.raises(ValueError, match="save times"):
        solve_heat_treatment(ALLOY, AGEING, 1.0, save_times=[2.0])
    with pytest.raises(RuntimeError, match="largest class"):
        solve_heat_treatment(ALLOY, AGEING, 1e6, radii=np.linspace(0.05e-9, 3e-9, 60))
    with pytest.raises(ValueError, match="ratio > 1"):
        stretched_radii(1e-10, 1e-9, 1.0, 1e-8)
