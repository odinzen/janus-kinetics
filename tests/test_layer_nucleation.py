"""A phase layer that nucleates during a layered run (layer_nucleation), with the nucleation barrier of d'Heurle,
J. Mater. Res. 3 (1988) 167. The case is his: a couple heated until a phase between its two sides, whose driving
force grows with temperature as NiSi2's does (his Table III), suddenly appears."""

import functools
import warnings

import numpy as np
import pytest
from pycalphad import Database
from scipy.optimize import brentq, minimize_scalar

from janus_kinetics import SolutionPhase, TemperatureSchedule
from janus_kinetics.interface_nucleation import interface_nucleus
from janus_kinetics.layer_nucleation import K_B, NucleatingLayer, _insert, _remap, driving_force, solve_layers_with_nucleation
from janus_kinetics.moving_boundary import FixedInterface, Layer, LocalEquilibrium, graded_faces

# A-rich alpha and B-rich gamma, ideal, and an ordered-like beta between them whose entropy makes it gain on both
# with temperature: at the alpha/gamma tie-line its driving force is 1.3 kJ/mol at 500 K and 4.4 at 900 K
TDB = "\n".join(["ELEMENT VA VACUUM 0 0 0 !", "ELEMENT A BLANK 1 0 0 !", "ELEMENT B BLANK 1 0 0 !",
                 "TYPE_DEFINITION % SEQ * !"]
                + [line for ph in ("ALPHA", "GAMMA", "BETA") for line in (f"PHASE {ph} % 1 1 !", f"CONSTITUENT {ph} :A,B: !")]
                + ["PARAMETER G(ALPHA,A;0) 298.15 0; 6000 N !", "PARAMETER G(ALPHA,B;0) 298.15 8000; 6000 N !",
                   "PARAMETER G(GAMMA,A;0) 298.15 8000; 6000 N !", "PARAMETER G(GAMMA,B;0) 298.15 0; 6000 N !",
                   "PARAMETER G(BETA,A;0) 298.15 9000-6*T; 6000 N !", "PARAMETER G(BETA,B;0) 298.15 9000-6*T; 6000 N !",
                   "PARAMETER L(BETA,A,B;0) 298.15 -20000; 6000 N !"]
                + [f"PARAMETER MQ({ph}&{e},*;0) 298.15 R*T*LN(1E-15); 6000 N !" for ph in ("ALPHA", "GAMMA", "BETA") for e in "AB"])
VM, SIGMA = 1e-5, (0.3, 0.3, 0.3)
HEAT = TemperatureSchedule([0.0, 800.0], [500.0, 1300.0])  # 1 K/s
LENGTH = 20e-6


@functools.lru_cache(maxsize=None)
def _phases():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(TDB)
        return tuple(SolutionPhase(db, ph, ["B", "A"]) for ph in ("ALPHA", "GAMMA", "BETA"))


def _tie(T):
    alpha, gamma, _ = _phases()
    return LocalEquilibrium(alpha, gamma, T, [0.1], [0.9]).tie_line()


def _barrier_kt(T):
    """Independently of the run: beta's best driving force at the alpha/gamma tie-line by a bounded scalar search,
    and the equal-energy lens barrier."""
    alpha, _, beta = _phases()
    xa, _ = _tie(T)
    mu = alpha.chemical_potentials(xa, T)
    r = minimize_scalar(lambda x: float(beta.gibbs.gm(np.array([x, 1 - x]), T) - (x * mu[0] + (1 - x) * mu[1])),
                        bounds=(1e-6, 1 - 1e-6), method="bounded")
    return interface_nucleus(*SIGMA, -r.fun / VM).barrier / (K_B * T)


def _couple(schedule, threshold=60.0, check=5.0, t_end=800.0):
    alpha, gamma, beta = _phases()
    xa, xg = _tie(float(schedule(0.0)) if callable(schedule) else schedule)
    layers = [Layer(np.full(60, 0.05), np.eye(1) * 1e-15, "alpha", graded_faces(60, 20, "right")),
              Layer(np.full(60, 0.95), np.eye(1) * 1e-15, "gamma", graded_faces(60, 20, "left"))]
    cand = NucleatingLayer(beta, 0, SIGMA, VM, np.eye(1) * 1e-14, "beta", cells=20, thickness=2e-9, barrier=threshold)
    return solve_layers_with_nucleation([0.0, LENGTH / 2, LENGTH], layers, [LocalEquilibrium(alpha, gamma, schedule, xa, xg)],
                                        t_end, 1.0, schedule, [cand], check, save_times=np.arange(10.0, t_end, 10.0))


@functools.lru_cache(maxsize=None)
def _heated():
    return _couple(HEAT)


def test_beta_nucleates_where_its_barrier_reaches_the_threshold():
    """Heating at 1 K/s, beta's barrier at the interface falls from about 1200 kT at 500 K; it is inserted at the
    first check after the barrier reaches 60 kT, which the independent calculation puts at 897.5 K: the event is at
    900 K, with 59.4 kT, while at the check before (895 K) the barrier was still above 60."""
    run = _heated()
    onset = brentq(lambda T: _barrier_kt(T) - 60.0, 600.0, 1300.0)
    assert onset == pytest.approx(897.5, abs=0.5)
    assert len(run.events) == 1
    ev = run.events[0]
    assert onset <= ev.temperature < onset + 5.0
    assert ev.barrier_kt <= 60.0 < _barrier_kt(ev.temperature - 5.0)
    assert ev.barrier_kt == pytest.approx(_barrier_kt(ev.temperature), rel=1e-4)
    np.testing.assert_allclose(ev.composition, [0.5, 0.5], atol=1e-6)


def test_nucleation_conserves_every_element_and_the_layer_grows():
    """The inserted layer takes a slab from each neighbour, so the B content of the stack is the same just before
    and just after it (to 1e-12) and over the whole run; beta then grows from 2 nm to micrometres."""
    run = _heated()
    totals = np.concatenate([inv[:, 0] for inv in run.inventory()])
    np.testing.assert_allclose(totals, totals[0], rtol=1e-12)
    t, w = run.thickness("beta")
    assert w[0] == pytest.approx(2e-9, rel=1e-12) and w[-1] > 1e-6
    assert np.all(np.diff(w) >= -1e-15)


def test_no_nucleation_while_the_barrier_stays_high():
    """Held at 600 K the barrier is several hundred kT and the couple stays two layers."""
    assert _barrier_kt(600.0) > 300
    run = _couple(600.0, t_end=60.0)
    assert not run.events and all(seg.names == ("alpha", "gamma") for seg in run.segments)


def test_driving_force_is_the_parallel_tangent():
    """At the composition driving_force returns, the new phase's chemical potentials exceed the given ones by the
    same amount for every element (its tangent is parallel), and that amount is minus the driving force."""
    alpha, _, beta = _phases()
    for T in (700.0, 1100.0):
        xa, _ = _tie(T)
        mu = alpha.chemical_potentials(xa, T)
        dg, x = driving_force(beta, mu, T)
        diff = beta.chemical_potentials(x[:-1], T) - mu
        np.testing.assert_allclose(diff, -dg, atol=1e-6 * abs(dg))
        assert dg > 0


def test_nucleation_needs_local_equilibrium():
    alpha, gamma, beta = _phases()
    layers = [Layer(np.full(10, 0.05), np.eye(1) * 1e-15, "alpha"), Layer(np.full(10, 0.95), np.eye(1) * 1e-15, "gamma")]
    cand = NucleatingLayer(beta, 0, SIGMA, VM, np.eye(1) * 1e-14, "beta")
    with pytest.raises(ValueError, match="local equilibrium"):
        solve_layers_with_nucleation([0.0, 1e-5, 2e-5], layers, [FixedInterface([0.1], [0.9])], 2.0, 1.0, 900.0, [cand], 1.0)


def test_remap_is_conservative_on_a_rough_profile():
    """Cell averages over new faces inside the old extent carry exactly the content between them, whatever the
    profile: a jagged two-element profile on uneven cells, remapped onto a shorter uneven grid."""
    rng = np.random.default_rng(3)
    faces = np.concatenate([[0.0], np.cumsum(rng.uniform(0.5, 2.0, 30))])
    x = rng.uniform(0.0, 1.0, (30, 2))
    new = np.sort(rng.uniform(faces[0], faces[-1] * 0.8, 12))
    out = _remap(faces, x, new)
    for k in range(2):
        exact = [np.sum(np.clip(np.minimum(faces[1:], b) - np.maximum(faces[:-1], a), 0, None) * x[:, k])
                 for a, b in zip(new[:-1], new[1:])]
        np.testing.assert_allclose(out[:, k] * np.diff(new), exact, rtol=1e-12, atol=1e-15)


def test_inserted_layer_starts_at_the_nucleus_composition():
    """Uniform neighbours at x_B = 0.1 and 0.7 and a nucleus at 0.5: the lever rule takes a third of the slab from
    the left and two thirds from the right, so the new layer starts exactly at 0.5 and the content is unchanged."""
    alpha, gamma, beta = _phases()
    pos = np.array([0.0, 1e-6, 2e-6])
    layers = [Layer(np.full(10, 0.1), np.eye(1) * 1e-15, "alpha"), Layer(np.full(10, 0.7), np.eye(1) * 1e-15, "gamma")]
    conds = [LocalEquilibrium(alpha, gamma, 900.0, [0.25], [0.75])]
    cand = NucleatingLayer(beta, 0, SIGMA, VM, np.eye(1) * 1e-14, "beta", cells=6, thickness=3e-8)
    new_pos, new_layers, new_conds = _insert(pos, layers, conds, cand, np.array([0.5, 0.5]), 900.0)
    np.testing.assert_allclose(new_pos, [0.0, 1e-6 - 1e-8, 1e-6 + 2e-8, 2e-6], rtol=1e-12)
    np.testing.assert_allclose(new_layers[1].x0, 0.5, rtol=1e-12)
    before = 1e-6 * 0.1 + 1e-6 * 0.7
    widths = np.diff(new_pos)
    after = sum(w * np.mean(np.ravel(lay.x0)) for w, lay in zip(widths, new_layers))
    assert after == pytest.approx(before, rel=1e-12)
    assert [type(c) for c in new_conds] == [LocalEquilibrium, LocalEquilibrium] and new_conds[0].right is beta
