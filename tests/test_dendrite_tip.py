"""The dendrite tip of Kattner, Boettinger and Coriell, Z. Metallkd. 87 (1996) 522, eqs. (2)-(5). Their IN718 and
Rene N4 results rest on an unpublished preliminary database and are not reproduced; the checks are the equations
themselves, their limits, and the paper's statement (p. 527) that with equal liquid diffusivities and no curvature
undercooling the lever rule at the tip temperature gives the tip liquid."""

import functools
import warnings

import numpy as np
import pytest
from pycalphad import Database, equilibrium, variables as v
from scipy.integrate import quad
from scipy.optimize import brentq

from janus_kinetics import SolutionPhase
from janus_kinetics.dendrite_tip import dendrite_tip, dendrite_tip_calphad, ivantsov, liquidus
from janus_kinetics.thermo import R

GAMMA = 2e-7  # K m


def _iv(p):
    """P exp(P) E1(P) = P int_0^inf exp(-P u) / (1 + u) du, by quadrature."""
    return p * quad(lambda u: np.exp(-p * u) / (1 + u), 0, np.inf, epsabs=0, epsrel=1e-13, limit=500)[0]


def test_ivantsov_function():
    """Against quadrature from P = 1e-4 to 1e4, and its two ends: P (ln(1/P) - Euler's gamma) for small P and
    1 - 1/P + 2/P^2 - 6/P^3 for large."""
    for p in (1e-4, 1e-2, 0.3, 1.0, 5.0, 40.0, 299.0, 301.0, 1e3, 1e4):
        assert ivantsov(p) == pytest.approx(_iv(p), rel=1e-10)
    assert ivantsov(1e-7) == pytest.approx(1e-7 * (np.log(1e7) - np.euler_gamma), rel=1e-5)
    assert ivantsov(1e5) == pytest.approx(1 - 1e-5 + 2e-10 - 6e-15, rel=1e-14)
    with pytest.raises(ValueError, match="positive"):
        ivantsov(0.0)


# three solutes, one enriched in the solid (k > 1, liquidus rising with it)
C0 = np.array([0.02, 0.03, 0.01])
K = np.array([0.3, 0.6, 2.0])
M = np.array([-500.0, -200.0, 300.0])
D = np.array([3e-9, 2e-9, 4e-9])


def test_tip_satisfies_eqs_2_to_5():
    """Rebuilt by hand from the returned radius: each tip liquid from eq. (2), the solute gradients
    -V (1 - k_i) C*_i / D_i, all three m_i G_ci positive (the k > 1 solute too, which the printed sign of eq. (4)
    would make negative), and eq. (3); the tip temperature from the linear liquidus less 2 Gamma / R."""
    for V, G in ((1e-4, 0.0), (1e-3, 1e4), (3e-2, 5e3)):
        tip = dendrite_tip(V, G, C0, K, M, D, GAMMA, 1700.0)
        p = V * tip.radius / (2 * D)
        np.testing.assert_allclose(tip.peclet, p, rtol=1e-14)
        c = C0 / (1 - (1 - K) * np.array([_iv(x) for x in p]))
        np.testing.assert_allclose(tip.liquid, c, rtol=1e-10)
        gc = -V * (1 - K) * c / D
        assert np.all(M * gc > 0)
        assert tip.radius == pytest.approx(2 * np.pi * np.sqrt(GAMMA / (np.sum(M * gc) - G)), rel=1e-9)
        assert tip.temperature == pytest.approx(1700.0 + np.sum(M * (c - C0)) - 2 * GAMMA / tip.radius, rel=1e-12)
        np.testing.assert_allclose(tip.solid, K * c, rtol=1e-14)


def test_low_peclet_limit():
    """With no gradient and one solute eq. (3) reads V R^2 = 4 pi^2 Gamma D / (m (k - 1) C*), so
    V R^2 / [4 pi^2 Gamma D / (m (k - 1) C0)] = C0 / C* = 1 - (1 - k) Iv(P) by eq. (2): as V falls, C* -> C0, V R^2
    settles at that constant and the tip temperature approaches the liquidus."""
    k, m, c0, d = 0.3, -500.0, 0.02, 3e-9
    limit = 4 * np.pi**2 * GAMMA * d / (m * (k - 1) * c0)
    for V in (1e-2, 1e-5, 1e-8):
        tip = dendrite_tip(V, 0.0, [c0], [k], [m], d, GAMMA, 1700.0)
        assert V * tip.radius**2 / limit == pytest.approx(1 - (1 - k) * _iv(tip.peclet[0]), rel=1e-10)
    assert abs(V * tip.radius**2 / limit - 1) < 5e-3  # (1 - k) P ln(1/P) at P = 1e-3
    assert 1700.0 - tip.temperature < 0.05


def test_two_identical_solutes_act_as_one():
    """Two solutes with the same k, m and D are one solute of the summed content: same radius and temperature."""
    one = dendrite_tip(1e-3, 2e3, [0.02], [0.3], [-500.0], 3e-9, GAMMA, 1700.0)
    two = dendrite_tip(1e-3, 2e3, [0.012, 0.008], [0.3, 0.3], [-500.0, -500.0], 3e-9, GAMMA, 1700.0)
    assert two.radius == pytest.approx(one.radius, rel=1e-10)
    assert two.temperature == pytest.approx(one.temperature, rel=1e-12)
    assert two.liquid.sum() == pytest.approx(one.liquid[0], rel=1e-10)


def test_temperature_gradient_turns_the_curve():
    """With G > 0 no tip exists below V_c = G D k / (m (k - 1) C0), the limit of constitutional undercooling; just
    above it the radius diverges, P grows, Iv -> 1 and C* -> C0 / k, so the tip sits near the solidus. The tip
    temperature therefore rises from the solidus, peaks and falls with V: the turn of Figs. 4 and 6. A steeper
    gradient widens the tip at a given speed."""
    k, m, c0, d, G = 0.3, -500.0, 0.02, 3e-9, 6e3
    vc = G * d * k / (m * (k - 1) * c0)
    with pytest.raises(ValueError, match="planar"):
        dendrite_tip(0.999 * vc, G, [c0], [k], [m], d, GAMMA, 1700.0)
    solidus = 1700.0 + m * c0 * (1 / k - 1)
    near = dendrite_tip(1.0001 * vc, G, [c0], [k], [m], d, GAMMA, 1700.0)
    assert near.temperature - solidus < 0.01 * (1700.0 - solidus)
    speeds = vc * np.logspace(0.01, 6, 60)
    temps = np.array([dendrite_tip(s, G, [c0], [k], [m], d, GAMMA, 1700.0).temperature for s in speeds])
    top = np.argmax(temps)
    assert 0 < top < len(speeds) - 1
    assert np.all(np.diff(temps[: top + 1]) > 0) and np.all(np.diff(temps[top:]) < 0)
    flat = dendrite_tip(1e-4, 0.0, [c0], [k], [m], d, GAMMA, 1700.0)
    steep = dendrite_tip(1e-4, 4e3, [c0], [k], [m], d, GAMMA, 1700.0)
    assert steep.radius > flat.radius


def test_input_checks():
    with pytest.raises(ValueError, match="lower the liquidus"):
        dendrite_tip(1e-3, 0.0, [0.02], [0.3], [500.0], 3e-9, GAMMA, 1700.0)
    with pytest.raises(ValueError, match="lower the liquidus"):
        dendrite_tip(1e-3, 0.0, [0.02], [2.0], [-500.0], 3e-9, GAMMA, 1700.0)


# --- with the phase equilibrium --------------------------------------------------------------------------------

# ideal solid and liquid A-B-C, fusion entropy 30 J/mol K, melting points A 1700, B 1200, C 1400 K (the
# solidification tests' alloy): the liquidus is closed form, sum_i x_L,i exp(30 (T_M,i - T) / (R T)) = 1
_MELT = {"A": 1700.0, "B": 1200.0, "C": 1400.0}
_ALLOY = [0.05, 0.05]


@functools.lru_cache(maxsize=None)
def _toy():
    lines = ["ELEMENT VA VACUUM 0 0 0 !"] + [f"ELEMENT {e} BLANK 1 0 0 !" for e in "ABC"]
    lines += ["TYPE_DEFINITION % SEQ * !", "PHASE FCC % 1 1 !", "CONSTITUENT FCC :A,B,C: !", "PHASE LIQUID % 1 1 !",
              "CONSTITUENT LIQUID :A,B,C: !"]
    for e in "ABC":
        lines += [f"PARAMETER G(FCC,{e};0) 298.15 0; 6000 N !", f"PARAMETER G(LIQUID,{e};0) 298.15 30*({_MELT[e]}-T); 6000 N !"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(chr(10).join(lines))
        return (db, SolutionPhase(db, "FCC", ["B", "C", "A"], mobilities=False),
                SolutionPhase(db, "LIQUID", ["B", "C", "A"], mobilities=False))


def _k(T):
    return np.array([np.exp(30 * (_MELT[e] - T) / (R * T)) for e in "BCA"])


def test_liquidus_of_an_ideal_ternary():
    """Temperature, solid, partition ratios and slopes against the closed form: T_L solves sum x_i k_i(T) = 1,
    and implicit differentiation gives m_j = (k_j - k_A) / sum_i x_i k_i 30 T_M,i / (R T^2)."""
    _, solid, liq = _toy()
    for x in ([0.05, 0.05], [0.2, 0.01], [0.01, 0.3]):
        full = np.append(x, 1 - sum(x))
        T = brentq(lambda t: full @ _k(t) - 1, 1000.0, 1700.0, xtol=1e-12)
        eq = liquidus(solid, liq, x, 1600.0)
        assert eq.temperature == pytest.approx(T, rel=1e-11)
        np.testing.assert_allclose(eq.partition, _k(T)[:2], rtol=1e-9)
        np.testing.assert_allclose(eq.solid, (full * _k(T))[:2], rtol=1e-9)
        dfdt = np.sum(full * _k(T) * 30 * np.array([_MELT[e] for e in "BCA"])) / (R * T**2)
        np.testing.assert_allclose(eq.slopes, (_k(T)[:2] - _k(T)[2]) / dfdt, rtol=1e-6)


def _lever(T):
    db = _toy()[0]
    eq = equilibrium(db, ["A", "B", "C", "VA"], ["FCC", "LIQUID"],
                     {v.T: T, v.P: 101325, v.N: 1, v.X("B"): _ALLOY[0], v.X("C"): _ALLOY[1]})
    names = [str(p) for p in np.ravel(eq.Phase.values)]
    amounts = np.ravel(eq.NP.values)
    comp = eq.X.values.reshape(len(names), -1)  # components A, B, C
    return {n: (a, comp[i][[1, 2]]) for i, (n, a) in enumerate(zip(names, amounts)) if n}


def test_tip_liquid_is_the_lever_liquid_when_diffusivities_are_equal():
    """p. 527: with equal liquid diffusivities and no curvature undercooling, a lever-rule calculation of the
    alloy at the tip's liquidus temperature T_L(C*) gives the tip liquid, here from pycalphad independently; its
    fraction solid is Iv(P), the common Peclet number. With unequal diffusivities it does not."""
    _, solid, liq = _toy()
    for V in (1e-5, 1e-2, 1.0):
        tip = dendrite_tip_calphad(solid, liq, V, 0.0, _ALLOY, 3e-9, GAMMA, 1650.0)
        lever = _lever(tip.liquidus_temperature)
        np.testing.assert_allclose(lever["LIQUID"][1], tip.liquid, rtol=1e-6)
        np.testing.assert_allclose(lever["FCC"][1], tip.solid, rtol=1e-6)
        assert lever["FCC"][0] == pytest.approx(ivantsov(tip.peclet[0]), rel=1e-6)
    tip = dendrite_tip_calphad(solid, liq, 1e-2, 0.0, _ALLOY, [3e-9, 1e-9], GAMMA, 1650.0)
    lever = _lever(tip.liquidus_temperature)
    assert np.max(np.abs(lever["LIQUID"][1] / tip.liquid - 1)) > 1e-2


def test_calphad_tip_is_self_consistent():
    """The returned tip obeys eqs. (2)-(5) with k and m taken afresh from the liquidus at its own liquid, and with
    those held fixed the linear tip is the same tip."""
    _, solid, liq = _toy()
    for V, G in ((1e-3, 0.0), (1e-1, 1e4)):
        tip = dendrite_tip_calphad(solid, liq, V, G, _ALLOY, [3e-9, 2e-9], GAMMA, 1650.0)
        eq = liquidus(solid, liq, tip.liquid, 1650.0)
        np.testing.assert_allclose(tip.liquid, _ALLOY / (1 - (1 - eq.partition) * ivantsov(tip.peclet)), rtol=1e-8)
        gc = -V * (1 - eq.partition) * tip.liquid / np.array([3e-9, 2e-9])
        assert tip.radius == pytest.approx(2 * np.pi * np.sqrt(GAMMA / (np.sum(eq.slopes * gc) - G)), rel=1e-8)
        assert tip.temperature == pytest.approx(eq.temperature - 2 * GAMMA / tip.radius, abs=1e-8)
        linear = dendrite_tip(V, G, _ALLOY, eq.partition, eq.slopes, [3e-9, 2e-9], GAMMA, 0.0)
        assert linear.radius == pytest.approx(tip.radius, rel=1e-8)


def test_tip_cools_with_speed_and_the_least_partitioned_solute_piles_up_most():
    """Figs. 4-5 at zero gradient: the tip temperature approaches the liquidus as V -> 0 and falls steadily with V,
    and the solute with the smallest k (B here, k about 0.36; Nb in IN718) changes most at the tip."""
    _, solid, liq = _toy()
    tl = liquidus(solid, liq, _ALLOY, 1650.0).temperature
    speeds = np.logspace(-8, 0, 9)
    tips = [dendrite_tip_calphad(solid, liq, s, 0.0, _ALLOY, 3e-9, GAMMA, 1650.0) for s in speeds]
    temps = np.array([t.temperature for t in tips])
    assert 0 < tl - temps[0] < 0.1
    assert np.all(np.diff(temps) < 0)
    rise = tips[-1].liquid / np.array(_ALLOY)
    assert rise[0] > rise[1] > 1
