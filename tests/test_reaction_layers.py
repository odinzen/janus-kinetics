"""Stoichiometric reaction layers, checked against van Loo, Prog. Solid State Chem. 20 (1990) 47.

The Ni-Sn numbers are from his pp. 64-65 (Fig. 2.8, Table 2.1, Fig. 2.9 and the
text beside it), the wustite numbers from p. 78 (Fig. 2.22 and the two integrated
diffusivities above it). His intrinsic diffusivities relate to tracer mobilities
through eq. 14 with r = 1, D_Ni / D_Sn = (M_Ni / M_Sn)(V_Ni / V_Sn).
"""

import numpy as np
import pytest
from pycalphad import Database

from calphad_kinetics.reaction_layers import (
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

    from calphad_kinetics import TemperatureSchedule

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
