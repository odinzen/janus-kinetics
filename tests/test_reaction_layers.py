"""Stoichiometric reaction layers, checked against van Loo, Prog. Solid State Chem. 20 (1990) 47.

The Ni-Sn numbers are from his pp. 64-65 (Fig. 2.8, Table 2.1, Fig. 2.9 and the
text beside it), the wustite numbers from p. 78 (Fig. 2.22 and the two integrated
diffusivities above it). His intrinsic diffusivities relate to tracer mobilities
through eq. 14 with r = 1, D_Ni / D_Sn = (M_Ni / M_Sn)(V_Ni / V_Sn).
"""

import numpy as np
import pytest
from pycalphad import Database

from janus_kinetics.reaction_layers import (
    LineCompound,
    integrated_diffusivity,
    parabolic_growth,
    solve_reaction_layers,
)

# Volumes per formula unit from the reaction 2 Ni3Sn + Ni3Sn4 -> 3 Ni3Sn2, 103 : 125 : 225
# (p. 64); units cancel in every ratio below.
NI3SN = LineCompound("Ni3Sn", 0.25, 51.5 / 4 * 1e-6)
NI3SN4 = LineCompound("Ni3Sn4", 4 / 7, 125 / 7 * 1e-6)
V_NI_OVER_V_SN = 0.38  # p. 65


def ni_sn_couple(m_sn_over_m_ni):
    layer = LineCompound("Ni3Sn2", 0.4, 75 / 5 * 1e-6, integrated_diffusivity=1e-15, tracer_ratio=m_sn_over_m_ni)
    return [NI3SN, layer, NI3SN4]


def marker_split(growth):
    """(x1, x2) of Fig. 2.8: layer on the Ni3Sn side and on the Ni3Sn4 side of the markers."""
    a0, a1 = growth.interfaces
    return growth.marker - a0, a1 - growth.marker


@pytest.mark.parametrize("ratio, expected", [(np.inf, 0.5), (0.0, 2.0)])
def test_table_2_1_marker_ratio_when_one_element_moves(ratio, expected):
    """Only Sn diffusing gives x2/x1 = 0.5, only Ni diffusing gives 2 (Table 2.1)."""
    g = parabolic_growth(ni_sn_couple(ratio), 1000.0)
    x1, x2 = marker_split(g)
    assert g.marker_phase == "Ni3Sn2"
    assert x2 / x1 == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("ratio", [0.0, 0.04, 1.0, 7.0, np.inf])
def test_penetration_ratio_follows_the_reaction_not_the_diffusing_element(ratio):
    """d2/d1 = 125/103 whichever element moves (p. 64 and point a on p. 65)."""
    g = parabolic_growth(ni_sn_couple(ratio), 1000.0)
    assert g.consumed["Ni3Sn4"] / g.consumed["Ni3Sn"] == pytest.approx(125 / 103, rel=1e-10)


def test_marker_shift_between_ratio_5_and_infinity():
    """For a 100 um layer the markers for D_Ni/D_Sn = 5 and infinity differ by 3.4 um (p. 65)."""
    def x1_per_100um(d_ratio):
        m_sn_over_m_ni = V_NI_OVER_V_SN / d_ratio
        g = parabolic_growth(ni_sn_couple(m_sn_over_m_ni), 1000.0)
        x1, x2 = marker_split(g)
        return 100 * x1 / (x1 + x2)

    assert x1_per_100um(5.0) - x1_per_100um(np.inf) == pytest.approx(3.4, abs=0.05)


def test_measured_marker_ratio_maps_to_about_9_5():
    """x2/x1 = 1.85 'leads to' D_Ni/D_Sn of about 9.5 with volumes alone (p. 65).

    The curve is steep there (Fig. 2.9): 9.5 gives 1.84 and 1.85 gives 10.3, so the
    check is on x2/x1 at the printed ratio.
    """
    g = parabolic_growth(ni_sn_couple(V_NI_OVER_V_SN / 9.5), 1000.0)
    x1, x2 = marker_split(g)
    assert x2 / x1 == pytest.approx(1.85, abs=0.015)


def fe_o(right, volumes, ratio=1.0):
    fe = LineCompound("Fe", 0.0, volumes[0])
    feo = LineCompound("FeO", 0.5, volumes[1], integrated_diffusivity=3e-13, tracer_ratio=ratio)
    return [fe, feo, LineCompound(right[0], right[1], volumes[2])]


@pytest.mark.parametrize("volumes", [(7.09e-6, 6.3e-6, 6.4e-6), (1e-5, 1e-5, 1e-5)])
@pytest.mark.parametrize("ratio", [0.0, 1.0, np.inf])
def test_wustite_grows_twice_as_thick_against_magnetite(volumes, ratio):
    """D_int = (1/16) d1^2/2t in Fe/Fe3O4 and (1/4) d2^2/2t in Fe/O2, so d1 = 2 d2 (p. 78).

    Molar volumes and the split of the flux between the elements drop out, as
    van Loo notes for line compounds.
    """
    g1 = parabolic_growth(fe_o(("Fe3O4", 4 / 7), volumes, ratio), 1200.0)
    g2 = parabolic_growth(fe_o(("O2", 1.0), volumes, ratio), 1200.0)
    k1, k2 = g1.growth_constants["FeO"], g2.growth_constants["FeO"]
    assert 3e-13 == pytest.approx(k1 / 16, rel=1e-10)
    assert 3e-13 == pytest.approx(k2 / 4, rel=1e-10)
    assert np.sqrt(k1 / k2) == pytest.approx(2.0, rel=1e-10)


def thermo_couple():
    """A-B couple with two compounds; G per mole of atoms in J/mol, mobilities Arrhenius."""
    def mob(m0_a, m0_b, q):
        return lambda T: (m0_a * np.exp(-q / (8.314 * T)), m0_b * np.exp(-q / (8.314 * T)))

    return [
        LineCompound("A", 0.0, 1.0e-5, gibbs=0.0),
        LineCompound("A2B", 1 / 3, 1.1e-5, gibbs=lambda T: -30000.0 + 2.0 * T, mobility=mob(2e-8, 5e-9, 150e3)),
        LineCompound("AB", 0.5, 1.2e-5, gibbs=lambda T: -38000.0 + 2.0 * T, mobility=mob(1e-9, 3e-9, 160e3)),
        LineCompound("B", 1.0, 1.3e-5, gibbs=0.0),
    ]


def test_mobility_route_matches_its_integrated_diffusivity():
    """Layers given by mobilities grow exactly as layers given the same D_int and ratio."""
    phases, T = thermo_couple(), 1100.0
    g = parabolic_growth(phases, T)
    swapped = list(phases)
    for k in (1, 2):
        m_a, m_b = phases[k].mobility(T)
        swapped[k] = LineCompound(phases[k].name, phases[k].x, phases[k].molar_volume,
                                  integrated_diffusivity=integrated_diffusivity(*phases[k - 1:k + 2], T),
                                  tracer_ratio=m_b / m_a)
    g2 = parabolic_growth(swapped, T)
    for name in ("A2B", "AB"):
        assert g2.growth_constants[name] == pytest.approx(g.growth_constants[name], rel=1e-10)
    assert g2.marker == pytest.approx(g.marker, rel=1e-10)


def test_integrated_diffusivity_from_chord_slopes():
    """D_int = x_A x_B (x_A M_B + x_B M_A) (S_right - S_left) with chord slopes S."""
    a, a2b, ab, _ = thermo_couple()
    T = 1000.0
    s_left = (-30000.0 + 2 * T) / (1 / 3)
    s_right = ((-38000.0 + 2 * T) - (-30000.0 + 2 * T)) / (0.5 - 1 / 3)
    m_a, m_b = a2b.mobility(T)
    expected = (2 / 3) * (1 / 3) * ((2 / 3) * m_b + (1 / 3) * m_a) * (s_right - s_left)
    assert integrated_diffusivity(a, a2b, ab, T) == pytest.approx(expected, rel=1e-12)


def test_time_stepping_reproduces_the_similarity_solution():
    phases = thermo_couple()
    t = 3600.0
    g = parabolic_growth(phases, 1100.0)
    res = solve_reaction_layers(phases, t, 1100.0, save_times=[60.0, 600.0])
    for name, k in g.growth_constants.items():
        assert res.thickness[name][-1] == pytest.approx(np.sqrt(2 * k * t), rel=1e-6)
    assert res.markers[-1, 0] == pytest.approx(g.marker * np.sqrt(t), rel=1e-6)
    assert np.all(np.isinf(res.thickness["A"]))


def test_single_layer_integrates_d_int_over_the_schedule():
    """d dd/dt = K D_int(T(t)) for one layer, so d^2 = 2 K * integral of D_int dt exactly."""
    from scipy.integrate import quad

    from janus_kinetics import TemperatureSchedule

    schedule = TemperatureSchedule([0.0, 100.0, 400.0, 500.0], [700.0, 1000.0, 1000.0, 800.0])
    d_int = lambda T: 1e-12 * np.exp(-120e3 / (8.314 * T))
    phases = [LineCompound("A", 0.0, 1e-5), LineCompound("AB", 0.5, 1e-5, integrated_diffusivity=d_int,
                                                         tracer_ratio=1.0), LineCompound("B", 1.0, 1e-5)]
    res = solve_reaction_layers(phases, 500.0, schedule)
    geometry = (1.0 - 0.0) / ((1.0 - 0.5) * (0.5 - 0.0))
    exposure = quad(lambda t: d_int(schedule(t)), 0.0, 500.0, points=[100.0, 400.0], epsabs=0, epsrel=1e-12)[0]
    assert res.thickness["AB"][-1] == pytest.approx(np.sqrt(2 * geometry * exposure), rel=1e-6)


def moles(res, phases, i):
    total = np.zeros(2)
    for p in phases:
        d = res.thickness[p.name][i]
        if np.isfinite(d):
            total += d * p.concentrations()
    return total


@pytest.mark.parametrize("side", ["left", "right"])
def test_film_converts_to_compounds_in_turn(side):
    """A 100 nm film is used up, then the compound next to it, leaving only the
    compound next to the substrate, whose final thickness the film inventory fixes."""
    phases = thermo_couple()
    film, T = 100e-9, 1100.0
    if side == "left":
        ends, order, survivor, element = (film, None), ["A", "A2B"], "AB", 0
    else:
        ends, order, survivor, element = (None, film), ["B", "AB"], "A2B", 1
    g = parabolic_growth(phases, T, anchor="right" if side == "left" else "left")
    t_first = (film / g.consumed[order[0]]) ** 2
    res = solve_reaction_layers(phases, 100 * t_first, T, end_thickness=ends,
                                save_times=np.geomspace(1e-3 * t_first, 100 * t_first, 80))
    assert [name for _, name in res.consumed] == order
    assert res.consumed[0][0] == pytest.approx(t_first, rel=1e-6)
    film_phase = phases[0] if side == "left" else phases[-1]
    final = next(p for p in phases if p.name == survivor)
    expected = film * film_phase.concentrations()[element] / final.concentrations()[element]
    assert res.thickness[survivor][-1] == pytest.approx(expected, rel=1e-6)
    assert all(res.thickness[name][-1] == 0.0 for name in order)


def test_both_elements_conserved_across_phase_consumption():
    phases = thermo_couple()
    g = parabolic_growth(phases, 1100.0)
    t_first = (80e-9 / g.consumed["A"]) ** 2
    res = solve_reaction_layers(phases, 100 * t_first, 1100.0, end_thickness=(80e-9, 300e-9),
                                save_times=np.geomspace(1e-4 * t_first, 100 * t_first, 60))
    assert len(res.consumed) == 2
    assert np.sum(res.times < res.consumed[0][0]) > 10 and np.sum(res.times > res.consumed[1][0]) > 5
    start = moles(res, phases, 0)
    for i in range(len(res.times)):
        assert moles(res, phases, i) == pytest.approx(start, rel=1e-7)


def test_pregrown_layers_relax_towards_parabolic_ratio():
    """An incremental couple with a thick A2B and thin AB: both keep growing and the
    thickness ratio approaches the fresh-couple value (van Loo p. 62, Ti/TiAl3 then Al)."""
    phases = thermo_couple()
    g = parabolic_growth(phases, 1100.0)
    target = np.sqrt(g.growth_constants["AB"] / g.growth_constants["A2B"])
    res = solve_reaction_layers(phases, 1e7, 1100.0, thickness=[5e-6, 1e-9], markers=[0.0, 2e-6])
    ratio = res.thickness["AB"][-1] / res.thickness["A2B"][-1]
    assert ratio == pytest.approx(target, rel=0.05)


def test_compound_above_the_hull_is_refused():
    phases = thermo_couple()
    phases[2] = LineCompound("AB", 0.5, 1.2e-5, gibbs=-10000.0, mobility=(1e-20, 1e-20))
    with pytest.raises(ValueError, match="no driving force"):
        parabolic_growth(phases, 1000.0)


@pytest.mark.parametrize("bad, match", [
    (lambda p: [p[0], p[2], p[1], p[3]], "increasing x"),
    (lambda p: [p[0], LineCompound("A2B", 1 / 3, 1e-5, gibbs=-3e4), p[2], p[3]], "needs a mobility"),
])
def test_bad_sequences_raise(bad, match):
    with pytest.raises(ValueError, match=match):
        parabolic_growth(bad(thermo_couple()), 1000.0)


def test_input_checks():
    with pytest.raises(ValueError, match="tracer_ratio"):
        LineCompound("AB", 0.5, 1e-5, integrated_diffusivity=1e-15)
    with pytest.raises(ValueError, match="not both"):
        LineCompound("AB", 0.5, 1e-5, mobility=(1, 1), integrated_diffusivity=1e-15, tracer_ratio=1.0)
    with pytest.raises(ValueError, match="positive layer thicknesses"):
        solve_reaction_layers(thermo_couple(), 10.0, 1000.0, thickness=[1e-6, 0.0])
    with pytest.raises(ValueError, match="original interface"):
        solve_reaction_layers(thermo_couple(), 10.0, 1000.0, markers=[1e-6])


TDB = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT A BLANK 0 0 0 !
ELEMENT B BLANK 0 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE A2B % 2 2 1 !
CONSTITUENT A2B :A:B: !
PARAMETER G(A2B,A:B;0) 298.15 -90000+6*T; 6000 N !
"""


def test_line_compound_from_pycalphad():
    c = LineCompound.from_pycalphad(Database(TDB), "A2B", ("A", "B"), 1.1e-5)
    assert c.x == pytest.approx(1 / 3)
    assert c.gibbs(1000.0) == pytest.approx((-90000 + 6 * 1000.0) / 3, rel=1e-10)


def wagner_eq20(phases, widths, t):
    """Wagner, Acta Metall. 17 (1969) 99, eq. 20: the integrated diffusivity of each
    line compound i in a couple of pure end members, from all layer widths dx,

        D_int(i) = (1 - N_i) sum_(v<i) (V_i/V_v) N_v dx_i dx_v / 2t + N_i (1 - N_i) dx_i^2 / 2t
                   + N_i sum_(v>i) (V_i/V_v) (1 - N_v) dx_i dx_v / 2t
    """
    out = {}
    inner = range(1, len(phases) - 1)
    for i in inner:
        n_i, v_i, d_i = phases[i].x, phases[i].molar_volume, widths[phases[i].name]
        total = n_i * (1 - n_i) * d_i**2 / (2 * t)
        for j in inner:
            ratio = v_i / phases[j].molar_volume * d_i * widths[phases[j].name] / (2 * t)
            if j < i:
                total += (1 - n_i) * ratio * phases[j].x
            elif j > i:
                total += n_i * ratio * (1 - phases[j].x)
        out[phases[i].name] = total
    return out


THREE_COMPOUNDS = [
    LineCompound("A", 0.0, 1.0e-5),
    LineCompound("A3B", 0.25, 0.9e-5, integrated_diffusivity=2e-14, tracer_ratio=0.3),
    LineCompound("AB", 0.5, 1.3e-5, integrated_diffusivity=5e-15, tracer_ratio=3.0),
    LineCompound("AB3", 0.75, 1.1e-5, integrated_diffusivity=1e-14, tracer_ratio=1.0),
    LineCompound("B", 1.0, 1.6e-5),
]


@pytest.mark.parametrize("couple", ["thermo", "three"])
def test_multilayer_growth_satisfies_wagner_eq_20(couple):
    """Every layer's D_int, from mobilities and Gibbs energies or given directly, is
    recovered from the simulated widths of all layers, with unequal molar volumes."""
    phases, T = (thermo_couple(), 1100.0) if couple == "thermo" else (THREE_COMPOUNDS, 1000.0)
    expected = {p.name: integrated_diffusivity(*phases[k - 1:k + 2], T) for k, p in enumerate(phases[1:-1], 1)}
    g = parabolic_growth(phases, T)
    t = 1.0
    widths = {name: np.sqrt(2 * k * t) for name, k in g.growth_constants.items()}
    for name, value in wagner_eq20(phases, widths, t).items():
        assert value == pytest.approx(expected[name], rel=1e-10)
    t = 5000.0
    res = solve_reaction_layers(phases, t, T)  # the same from the time-stepped widths
    widths = {name: res.thickness[name][-1] for name in expected}
    for name, value in wagner_eq20(phases, widths, t).items():
        assert value == pytest.approx(expected[name], rel=1e-5)


# --- concentric shells around a wire or particle ----------------------------------------------------------

def _particle(ratio):
    return [LineCompound("A", 0.0, 1.0e-5), LineCompound("AB", 0.4, 1.3e-5, integrated_diffusivity=1e-16, tracer_ratio=ratio),
            LineCompound("B", 1.0, 1.1e-5)]


@pytest.mark.parametrize("ratio", [0.3, 0.0, np.inf])
@pytest.mark.parametrize("geometry,shape,q", [("spherical", "sphere", 3), ("cylindrical", "cylinder", 2)])
def test_one_curved_layer_follows_van_loo_eq_26(geometry, shape, q, ratio):
    """One layer around a 1 um core: conversion and thickness as van Loo's eqs. 26 and 25 give them with k_p from
    the planar couple and z = (V_AB / x_A,AB) / V_A, within about 1e-10 whichever element diffuses. The core counts
    as used up at 1e-4 of its radius, 2.4e-8 (sphere) or 1.4e-7 (cylinder) of the conversion time before eq. 26
    has it gone. With one element diffusing the Kirkendall plane rides an interface: the core's surface when only
    A moves, the outer one when only B does."""
    from janus_kinetics.particle_reaction import conversion, conversion_integral, layer_thickness

    phases = _particle(ratio)
    r0 = 1e-6
    kp = parabolic_growth(phases, 1000.0).growth_constants["AB"]
    z = (phases[1].molar_volume / 0.6) / phases[0].molar_volume
    t_full = conversion_integral(1.0, z, shape) * r0**2 / (2 * kp)
    times = t_full * np.array([0.01, 0.1, 0.3, 0.6, 0.9, 0.99])
    res = solve_reaction_layers(phases, 1.5 * t_full, 1000.0, end_thickness=(r0, None), save_times=times,
                                geometry=geometry)
    alpha = 1 - (res.thickness["A"][:6] / r0) ** q
    np.testing.assert_allclose(alpha, conversion(times, kp, r0, z, shape), rtol=0, atol=1e-9)
    np.testing.assert_allclose(res.thickness["AB"][:6] / r0, layer_thickness(alpha, z, shape), rtol=0, atol=1e-9)
    assert [name for _, name in res.consumed] == ["A"]
    assert res.consumed[0][0] / t_full == pytest.approx(1.0, abs=3e-7)
    marker, inner, outer = res.markers[:6, 0], np.array([i[0] for i in res.interfaces[:6]]), np.array([i[1] for i in res.interfaces[:6]])
    if ratio == 0.0:
        np.testing.assert_allclose(marker, inner, rtol=1e-12)
    elif np.isinf(ratio):
        np.testing.assert_allclose(marker, outer, rtol=1e-12)
    else:
        assert np.all((marker > inner) & (marker < outer))


def test_two_curved_layers_agree_with_the_layered_stack_solver():
    """Two line compounds (x = 0.3, 0.6; D_int 2e-17 and 1e-17 m^2/s; equal molar volumes) between a 1 um sphere of
    x = 0.02 and a shell of x = 0.95, against solve_layers with each compound a phase of range delta and
    D = D_int / delta. The interface radii differ by 0.64 and 1.9 nm at 200 and 1000 s for delta = 0.02 and by half
    that for 0.01, against 130 and 400 nm of movement, so the narrow-range phases converge on the line compounds.
    (At 2000 s, with tighter steps in solve_layers, delta = 0.01 and 0.005 give 3.9 and 2.4 nm against 700 nm.)"""
    from janus_kinetics import FixedInterface
    from janus_kinetics.moving_boundary import Layer, solve_layers

    v, x, d_int = 1e-5, (0.02, 0.30, 0.60, 0.95), (2e-17, 1e-17)
    r0, d0, saves = 1e-6, (60e-9, 40e-9), [200.0, 1000.0]
    phases = [LineCompound("core", x[0], v), LineCompound("L1", x[1], v, integrated_diffusivity=d_int[0], tracer_ratio=1.0),
              LineCompound("L2", x[2], v, integrated_diffusivity=d_int[1], tracer_ratio=1.0), LineCompound("out", x[3], v)]
    line = solve_reaction_layers(phases, 1000.0, 1000.0, thickness=list(d0), end_thickness=(r0, None), save_times=saves,
                                 geometry="spherical")
    bounds = [0.0, r0, r0 + d0[0], r0 + d0[0] + d0[1], r0 + d0[0] + d0[1] + 4e-6]
    gaps = []
    for delta, cells in ((0.02, 30), (0.01, 60)):
        lo1, hi1, lo2, hi2 = x[1] - delta / 2, x[1] + delta / 2, x[2] - delta / 2, x[2] + delta / 2
        layers = [Layer(np.full(60, x[0]), 1e-14), Layer(np.linspace(lo1, hi1, cells), d_int[0] / delta),
                  Layer(np.linspace(lo2, hi2, cells), d_int[1] / delta), Layer(np.full(200, x[3]), 1e-14)]
        conds = [FixedInterface([x[0]], [lo1]), FixedInterface([hi1], [lo2]), FixedInterface([hi2], [x[3]])]
        stack = solve_layers(bounds, layers, conds, 1000.0, 250.0, save_times=saves[:1], geometry="spherical",
                             tolerance=1e-5)
        gaps.append([np.max(np.abs(stack.interfaces[i + 1] - line.interfaces[i][:3])) for i in range(2)])
    gaps = np.array(gaps)
    moved = [np.max(np.abs(line.interfaces[i][:3] - np.array(bounds[1:4]))) for i in range(2)]
    assert np.all(gaps[1] < 0.6 * gaps[0])
    assert np.all(gaps[1] < 3e-3 * np.array(moved))


@pytest.mark.parametrize("geometry,q", [("spherical", 3), ("cylindrical", 2)])
def test_curved_layers_conserve_both_elements(geometry, q):
    """A 0.5 um core inside a 2 um outer shell with two layers between: the core and then the first layer are used
    up, and every A and B atom stays accounted for within 4e-8 (at rtol 1e-11 within 3e-10; the loss comes from
    the fast last moments of each vanishing core)."""
    phases = [LineCompound("core", 0.0, 1.0e-5), LineCompound("L1", 0.3, 1.2e-5, integrated_diffusivity=2e-17, tracer_ratio=0.5),
              LineCompound("L2", 0.6, 1.4e-5, integrated_diffusivity=1e-17, tracer_ratio=3.0), LineCompound("out", 1.0, 1.1e-5)]
    res = solve_reaction_layers(phases, 20000.0, 1000.0, end_thickness=(0.5e-6, 2e-6),
                                save_times=np.linspace(100.0, 20000.0, 12), geometry=geometry)
    assert [name for _, name in res.consumed] == ["core", "L1"]

    def atoms(i):
        alive = [p for p in phases if p.name == "out" or res.thickness[p.name][i] > 0]
        edges = [0.0, *res.interfaces[i], res.interfaces[i][-1] + res.thickness["out"][i]]
        vol = [(hi**q - lo**q) / q / p.molar_volume for p, lo, hi in zip(alive, edges[:-1], edges[1:])]
        return np.array([sum(w * (1 - p.x) for w, p in zip(vol, alive)), sum(w * p.x for w, p in zip(vol, alive))])

    total = np.array([atoms(i) for i in range(len(res.times))])
    assert np.max(np.abs(total / total[0] - 1)) < 4e-8


@pytest.mark.parametrize("geometry", ["spherical", "cylindrical"])
def test_a_large_core_grows_its_layers_as_a_plate(geometry):
    """Around a 1 m core, layers of a few hundred nm grow as in the planar couple within 2e-7 (about their
    thickness over the radius)."""
    phases = [LineCompound("core", 0.0, 1.0e-5), LineCompound("L1", 0.3, 1.2e-5, integrated_diffusivity=2e-17, tracer_ratio=0.5),
              LineCompound("L2", 0.6, 1.4e-5, integrated_diffusivity=1e-17, tracer_ratio=3.0), LineCompound("out", 1.0, 1.1e-5)]
    big = solve_reaction_layers(phases, 1000.0, 1000.0, end_thickness=(1.0, None), save_times=[100.0], geometry=geometry)
    flat = solve_reaction_layers(phases, 1000.0, 1000.0, save_times=[100.0])
    for name in ("L1", "L2"):
        np.testing.assert_allclose(big.thickness[name], flat.thickness[name], rtol=2e-7)
        assert big.thickness[name][-1] > 1e-7



@pytest.mark.parametrize("geometry,q", [("spherical", 3), ("cylindrical", 2)])
def test_matrix_markers_ride_the_matrix_lattice(geometry, q):
    """A marker in the outer shell and the shell's outer surface both ride its lattice, and no atoms cross that
    lattice, so the volume between them, R^q - m^q, stays fixed while the lattice itself moves."""
    phases = _particle(0.3)
    res = solve_reaction_layers(phases, 500.0, 1000.0, thickness=[20e-9], end_thickness=(1e-6, 3e-6),
                                markers=[1.5e-6], save_times=[50.0, 200.0], geometry=geometry)
    outer = np.array([i[-1] for i in res.interfaces]) + res.thickness["B"]
    marker = res.markers[:, 0]
    np.testing.assert_allclose(outer**q - marker**q, (outer**q - marker**q)[0], rtol=1e-9)
    assert abs(marker[-1] - marker[0]) > 1e-3 * (marker[-1] - 1e-6)  # the lattice did move

def test_curved_inputs_are_checked():
    phases = _particle(1.0)
    with pytest.raises(ValueError, match="geometry"):
        solve_reaction_layers(phases, 10.0, 1000.0, end_thickness=(1e-6, None), geometry="conical")
    with pytest.raises(ValueError, match="core radius"):
        solve_reaction_layers(phases, 10.0, 1000.0, geometry="spherical")
    with pytest.raises(ValueError, match="core's lattice"):
        solve_reaction_layers(phases, 10.0, 1000.0, end_thickness=(1e-6, None), geometry="spherical", anchor="right")
