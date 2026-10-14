"""Phase layers that nucleate during a run of :func:`solve_layers`.

A candidate phase can appear at an interface held in local equilibrium. Its driving force there is the parallel
tangent to the interface chemical potentials, Delta G_m = max_x (sum_i x_i mu_i - G(x)), and its nucleus a lens on
the interface (:func:`interface_nucleus`, after d'Heurle, J. Mater. Res. 3 (1988) 167, eqs. 11-19). The run is
checked every ``check_interval``; once the barrier falls to ``barrier`` kT the phase is inserted as a thin layer at
the interface and the run continues with one more layer. The rule (a barrier threshold, the check interval, the
initial thickness) is this package's: classical nucleation gives the rate, but its prefactor and the number of
sites are rarely known, and since the barrier falls steeply as the driving force grows, the onset depends little
on the threshold.

The layer takes its material from both neighbours, a slab from each side of the interface whose widths are chosen,
by the lever rule, so that the mixture lies as close as it can to the nucleus composition (exactly, in a binary);
the new layer starts uniform at that mixture, so every element is conserved. The neighbours are remapped
conservatively onto their shortened extents. Plates only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .interface_nucleation import interface_nucleus
from .moving_boundary import Layer, LocalEquilibrium, StackResult, graded_faces, solve_layers
from .phase import SolutionPhase
from .schedule import Temperature, temperature_at, with_time
from .solver import Boundary, Diffusivity

K_B = 1.380649e-23  # J/K
N_A = 6.02214076e23  # 1/mol


def driving_force(phase: SolutionPhase, potentials: Sequence[float], T: float) -> tuple[float, np.ndarray]:
    """(Delta G_m in J/mol, mole fractions) of the composition of ``phase`` that gains most from forming out of
    material at chemical potentials ``potentials`` (J/mol, one per element): the parallel tangent,
    max_x (sum_i x_i mu_i - G_m(x)). Negative when the phase cannot form."""
    mu = np.asarray(potentials, dtype=float)
    n = mu.size
    if n != len(phase.elements):
        raise ValueError("one chemical potential per element of the phase")

    def gain(xi):
        x = np.append(xi, 1 - np.sum(xi))
        return float(x @ mu - phase.gibbs.gm(x, T))

    if n == 2:
        r = minimize_scalar(lambda v: -gain([v]), bounds=(1e-9, 1 - 1e-9), method="bounded", options={"xatol": 1e-12})
        best = np.array([r.x])
    else:
        def to_x(y):
            e = np.exp(np.clip(y, -50, 50))
            return e / (1 + e.sum())
        starts = [np.zeros(n - 1)] + [np.eye(n - 1)[k] * 3 for k in range(n - 1)] + [-3 * np.ones(n - 1)]
        results = [minimize(lambda y: -gain(to_x(y)), s, method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 4000}) for s in starts]
        best = to_x(min(results, key=lambda r: r.fun).x)
    return gain(best), np.append(best, 1 - best.sum())


@dataclass(frozen=True)
class NucleatingLayer:
    """A phase that may nucleate at interface ``interface`` (between layers i and i+1, held in local equilibrium):
    the interface energies of the old interface and of the new phase against its left and right neighbours (J/m^2),
    its molar volume (m^3/mol), its diffusivity (as for :class:`Layer`), and the layer it starts as."""

    phase: SolutionPhase
    interface: int
    energies: tuple[float, float, float]
    molar_volume: float
    diffusivity: Diffusivity
    name: str = ""
    cells: int = 20
    thickness: float = 2e-9
    grading: float = 4.0
    barrier: float = 60.0


@dataclass
class NucleationEvent:
    time: float
    temperature: float
    interface: int
    name: str
    barrier_kt: float
    driving_force: float
    composition: np.ndarray


@dataclass
class LayeredRun:
    """The run in segments, one :class:`StackResult` (times absolute) per stretch with a fixed set of layers, and
    the nucleation events between them."""

    segments: list
    events: list = field(default_factory=list)

    def thickness(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(times, thickness) of the layer called ``name`` over the segments where it exists."""
        times, widths = [], []
        for seg in self.segments:
            if name in seg.names:
                j = seg.names.index(name)
                times.append(seg.times)
                widths.append(seg.thickness()[:, j])
        if not times:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(times), np.concatenate(widths)

    def inventory(self) -> list[np.ndarray]:
        """Each segment's integral of every independent mole fraction over the stack (planar)."""
        return [seg.inventory() for seg in self.segments]


def _content(faces, x):
    """Cumulative amount of each element from the first face, at each face (piecewise-constant cells)."""
    return np.vstack([np.zeros(x.shape[1]), np.cumsum(np.diff(faces)[:, None] * x, axis=0)])


def _remap(faces, x, new_faces):
    """Conservative cell averages of the piecewise-constant profile (faces, x) on ``new_faces`` inside it."""
    cum = _content(faces, x)
    at = np.column_stack([np.interp(new_faces, faces, cum[:, k]) for k in range(x.shape[1])])
    return np.diff(at, axis=0) / np.diff(new_faces)[:, None]


def _shift_time(fn, t0):
    f = with_time(fn, 1)
    return lambda x, t: f(x, t + t0)


def solve_layers_with_nucleation(positions, layers: list[Layer], conditions: list, t_end: float, dt: float,
                                 temperature: Temperature, candidates: Sequence[NucleatingLayer],
                                 check_interval: float, left: Boundary = Boundary.no_flux(),
                                 right: Boundary = Boundary.no_flux(), save_times=None,
                                 tol: float = 1e-10) -> LayeredRun:
    """:func:`solve_layers` (planar) in stretches of ``check_interval``, inserting each candidate layer once its
    nucleation barrier at its interface falls to its ``barrier`` (in kT) at the temperature ``temperature`` (the
    one the interface conditions follow). Candidates are indexed by the interfaces of the stack they find; after one
    nucleates, the interfaces to its right shift up by one. Returns the run in segments with the events."""
    if check_interval <= 0 or t_end <= 0:
        raise ValueError("check_interval and t_end must be positive")
    pending = list(candidates)
    pos = np.asarray(positions, dtype=float)
    layer_list = list(layers)
    conds = list(conditions)
    run = LayeredRun([])
    saves = np.sort(np.asarray([] if save_times is None else save_times, dtype=float))
    t0 = 0.0
    while t0 < t_end * (1 - 1e-12):
        t1 = min(t0 + check_interval, t_end)
        inside = saves[(saves > t0) & (saves < t1)] - t0
        shifted_layers = [Layer(lay.x0, _shift_time(lay.diffusivity, t0) if callable(lay.diffusivity) else lay.diffusivity,
                                lay.name, lay.faces) for lay in layer_list]
        shifted_conds = []
        for c in conds:
            if isinstance(c, LocalEquilibrium):
                xl, xr = c.unpack(c.initial())
                shifted_conds.append(LocalEquilibrium(c.left, c.right, (lambda t, t0=t0: temperature_at(temperature, t + t0)),
                                                      xl, xr))
            else:
                shifted_conds.append(c)
        seg = solve_layers(pos, shifted_layers, shifted_conds, t1 - t0, min(dt, t1 - t0), left, right,
                           save_times=inside if inside.size else None, tol=tol)
        seg.times = seg.times + t0
        run.segments.append(seg)
        # carry the state over: positions, profiles on each layer's own fractions, interface compositions
        bounds = [0.0, *seg.interfaces[-1], pos[-1]]
        layer_list = [Layer(seg.x[j][-1], lay.diffusivity, lay.name,
                            (seg.faces[j][-1] - bounds[j]) / (bounds[j + 1] - bounds[j]))
                      for j, lay in enumerate(layer_list)]
        conds = [LocalEquilibrium(c.left, c.right, temperature, seg.interface_left[-1, i], seg.interface_right[-1, i])
                 if isinstance(c, LocalEquilibrium) else c for i, c in enumerate(conds)]
        pos = np.array(bounds)
        T = temperature_at(temperature, t1)
        # walk the pending list by position: an insertion shifts the interfaces of those after it, and they must be
        # read at their new indices in this same check
        k = 0
        while k < len(pending):
            cand = pending[k]
            i = cand.interface
            c = conds[i]
            if not isinstance(c, LocalEquilibrium):
                raise ValueError(f"interface {i} is not held in local equilibrium, so no phase can nucleate there")
            mu = c.left.chemical_potentials(c.unpack(c.initial())[0], T)
            dgm, x_new = driving_force(cand.phase, mu, T)
            barrier_kt = np.inf
            if dgm > 0:
                barrier_kt = interface_nucleus(*cand.energies, dgm / cand.molar_volume).barrier / (K_B * T)
            if barrier_kt > cand.barrier:
                k += 1
                continue
            pos, layer_list, conds = _insert(pos, layer_list, conds, cand, x_new, temperature)
            run.events.append(NucleationEvent(t1, T, i, cand.name, barrier_kt, dgm, x_new))
            del pending[k]
            pending = [NucleatingLayer(**{**p.__dict__, "interface": p.interface + 1}) if p.interface > i else p
                       for p in pending]
        t0 = t1
    return run


def _insert(pos, layer_list, conds, cand: NucleatingLayer, x_new, temperature):
    """Carve the new layer out of both sides of interface ``cand.interface``, conserving every element."""
    i = cand.interface
    lo, s, hi = pos[i], pos[i + 1], pos[i + 2]
    left_l, right_l = layer_list[i], layer_list[i + 1]
    left_faces = lo + (s - lo) * np.asarray(left_l.faces if left_l.faces is not None else np.linspace(0, 1, len(left_l.x0) + 1))
    right_faces = s + (hi - s) * np.asarray(right_l.faces if right_l.faces is not None else np.linspace(0, 1, len(right_l.x0) + 1))
    xl_prof, xr_prof = (np.asarray(lay.x0, dtype=float) for lay in (left_l, right_l))
    xl_prof = xl_prof[:, None] if xl_prof.ndim == 1 else xl_prof
    xr_prof = xr_prof[:, None] if xr_prof.ndim == 1 else xr_prof
    h = cand.thickness
    if h >= 0.5 * min(s - lo, hi - s):
        raise ValueError("the nucleated layer would be thicker than half a neighbouring layer; reduce its thickness")
    # mean compositions of a slab of width h on each side, then the split that brings the mixture closest to x_new
    xa = _remap(left_faces, xl_prof, np.array([s - h, s]))[0]
    xb = _remap(right_faces, xr_prof, np.array([s, s + h]))[0]
    target = np.asarray(x_new, dtype=float)[:-1]
    diff = xa - xb
    share = float(np.clip((target - xb) @ diff / max(diff @ diff, 1e-300), 0.05, 0.95))
    h_left, h_right = share * h, (1 - share) * h
    content = (_content(left_faces, xl_prof)[-1] - np.array([np.interp(s - h_left, left_faces, col) for col in _content(left_faces, xl_prof).T])
               + np.array([np.interp(s + h_right, right_faces, col) for col in _content(right_faces, xr_prof).T]))
    x_mix = content / h
    new_left = lo + (s - h_left - lo) * (left_faces - lo) / (s - lo)
    new_right = s + h_right + (hi - s - h_right) * (right_faces - s) / (hi - s)
    layers = list(layer_list)
    layers[i] = Layer(_remap(left_faces, xl_prof, new_left), left_l.diffusivity, left_l.name, left_l.faces)
    layers[i + 1] = Layer(_remap(right_faces, xr_prof, new_right), right_l.diffusivity, right_l.name, right_l.faces)
    layers.insert(i + 1, Layer(np.tile(x_mix, (cand.cells, 1)), cand.diffusivity, cand.name,
                               graded_faces(cand.cells, cand.grading, "both")))
    old = conds[i]
    xl_i, xr_i = old.unpack(old.initial())
    new_conds = list(conds)
    new_conds[i:i + 1] = [LocalEquilibrium(old.left, cand.phase, temperature, xl_i, target),
                          LocalEquilibrium(cand.phase, old.right, temperature, target, xr_i)]
    new_pos = np.concatenate([pos[:i + 1], [s - h_left, s + h_right], pos[i + 2:]])
    return new_pos, layers, new_conds
