"""Microsegregation during solidification: the closed-form limits, Kobayashi's exact solution of the
Brody-Flemings model, a numerical back-diffusion model in a plate, cylinder or sphere, and the
Scheil-Gulliver path from pycalphad equilibria.

The closed forms are as Kobayashi, J. Cryst. Growth 88 (1988) 87, states them for a constant
partition ratio k and a melt of composition C0: the Scheil equation (his eq. 1), the lever rule (eq. 5),
Brody and Flemings (eq. 2), Clyne and Kurz (eqs. 6-7) and Ohnaka (eqs. 6, 8), with
gamma = 2 D / K^2 for parabolic solidification X = K sqrt(t), which is 2 D t_f / L^2 for a local
solidification time t_f over half a dendrite arm spacing L. The exact solution is his eqs. (37)-(40).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from scipy.special import hyp1f1

from .errors import ConvergenceError
from .moving_boundary import (
    Layer,
    LocalEquilibrium,
    PhaseDissolvedError,
    _End,
    _Region,
    _region_step,
    graded_faces,
    solve_layers,
)
from .phase import SolutionPhase
from .schedule import TemperatureSchedule
from .solver import Geometry, _as_callable, cell_measures


def scheil(k: float, fs) -> np.ndarray:
    """Solid composition at the interface over C0, k (1 - f_s)^(k-1): no diffusion in the solid,
    complete mixing in the liquid (Kobayashi eq. 1)."""
    return k * (1 - np.asarray(fs, dtype=float)) ** (k - 1)


def lever_rule(k: float, fs) -> np.ndarray:
    """Interface solid composition over C0 at equilibrium, k / (1 - (1 - k) f_s) (Kobayashi eq. 5)."""
    return k / (1 - (1 - k) * np.asarray(fs, dtype=float))


def brody_flemings(k: float, gamma: float, fs) -> np.ndarray:
    """k [1 - (1 - gamma k) f_s]^((k-1)/(1 - gamma k)) (Kobayashi eq. 2); meaningful for gamma < 1 only."""
    return k * (1 - (1 - gamma * k) * np.asarray(fs, dtype=float)) ** ((k - 1) / (1 - gamma * k))


def clyne_kurz_beta(gamma: float) -> float:
    """beta = gamma [1 - exp(-2/gamma)] - exp(-1/gamma) (Kobayashi eq. 7)."""
    return gamma * (1 - np.exp(-2 / gamma)) - np.exp(-1 / gamma)


def clyne_kurz(k: float, gamma: float, fs, beta: float | None = None) -> np.ndarray:
    """Brody-Flemings with gamma replaced by beta (Kobayashi eq. 6), Clyne and Kurz's beta by default;
    Ohnaka's is beta = gamma / (1 + gamma) (eq. 8)."""
    b = clyne_kurz_beta(gamma) if beta is None else beta
    return k * (1 - (1 - b * k) * np.asarray(fs, dtype=float)) ** ((k - 1) / (1 - b * k))


def _zeta(k: float, gamma: float, terms: int) -> np.ndarray:
    """zeta_n of Kobayashi eq. (39), n = 0..terms-1."""
    m = np.arange(terms - 1)
    ratio = hyp1f1(-m / 2, 1.5, -1 / (2 * gamma)) / hyp1f1(-m / 2, 0.5, -1 / (2 * gamma))
    return np.concatenate([[1.0], np.cumprod(1 - k * ratio)])


def kobayashi(k: float, gamma: float, fs, x=None, terms: int = 10000) -> np.ndarray:
    """Exact solution of the Brody-Flemings model over C0 (Kobayashi eqs. 37-40): the interface solid
    composition at fraction solid ``fs``, or with ``x`` (x* = x / X, 0 at the dendrite spine) the profile
    in the solid. The series converges for 0 < k < 2 and gamma > 0; ``terms`` must grow as f_s
    approaches 1 with k small, where the terms fall off slowly."""
    fs = np.asarray(fs, dtype=float)
    zeta = _zeta(k, gamma, terms)
    n = np.arange(terms)
    powers = fs[..., None] ** n
    if x is None:
        return k * (zeta * powers).sum(axis=-1)
    x = np.asarray(x, dtype=float)
    shape = hyp1f1(-n / 2, 0.5, -(x[..., None] ** 2) / (2 * gamma)) / hyp1f1(-n / 2, 0.5, -1 / (2 * gamma))
    return k * (zeta * shape * powers).sum(axis=-1)


@dataclass
class BackDiffusionResult:
    """Fraction solid, liquid composition and interface solid composition (both over C0) at each step,
    with the solid profile (positions in m, compositions over C0) at the end."""

    fs: np.ndarray
    liquid: np.ndarray
    interface_solid: np.ndarray
    z_faces: np.ndarray
    solid: np.ndarray
    geometry: Geometry

    def mean_solid(self) -> float:
        """Average composition of the solid over C0 at the end; with no liquid left this is 1 by mass balance."""
        w = cell_measures(self.z_faces, self.geometry)
        return float(w @ self.solid / w.sum())


def solve_back_diffusion(k: float, diffusivity: float, length: float, solidification_time: float,
                         n_cells: int = 200, steps: int = 2000, geometry: Geometry = "planar",
                         fraction_solid: Callable[[float], float] | None = None,
                         fs_start: float = 1e-4, fs_end: float = 1.0) -> BackDiffusionResult:
    """The Brody-Flemings model solved numerically: a solid grows from the dendrite spine (r = 0) across
    a volume element of size ``length`` (half the arm spacing) while the liquid stays uniform; the
    interface solid holds k C_L, the solid diffuses with constant D, and nothing leaves the element.

    The solid grows to ``fraction_solid(t / solidification_time)`` (default sqrt, the parabolic growth
    Kobayashi assumes, his eq. 14); in a cylinder or sphere the fraction is that of the volume. Each step
    the solid is solved on its stretched grid by the package's moving-grid scheme, and the liquid
    composition from the element's solute balance. The solid's response is linear in the held interface
    value, so two solves give the exact C_L of the step. The run starts at ``fs_start`` with the solid at
    k C0 (the Scheil and exact solutions agree there to order f_s). Compositions are over C0.
    """
    if not 0 < k < 1:
        raise ValueError("this model needs a partition ratio 0 < k < 1 (solute rejected to the liquid)")
    frac = fraction_solid or (lambda tau: np.sqrt(tau))
    p = {"planar": 0, "cylindrical": 1, "spherical": 2}[geometry]
    total = length ** (p + 1) / (p + 1)

    def radius(fs):
        return length * fs ** (1 / (p + 1))

    # times at which the fraction solid steps evenly in sqrt(fs - fs_start), refining the start
    taus = np.linspace(0.0, 1.0, steps + 1)
    tau_of = _invert(frac, fs_start, fs_end)
    s = np.linspace(np.sqrt(fs_start), np.sqrt(fs_end), steps + 1) ** 2
    times = np.array([tau_of(f) for f in s]) * solidification_time
    dfun, _ = _as_callable(np.array([[diffusivity]]))
    region = _Region(dfun, True, np.linspace(0.0, 1.0, n_cells + 1), np.full((n_cells, 1), k))
    liquid = 1.0  # C_L / C0 in the Scheil start
    out_fs, out_l, out_i = [s[0]], [liquid], [k * liquid]
    z_old = region.faces(0.0, radius(s[0]))
    for j in range(1, steps + 1):
        dt = times[j] - times[j - 1]
        z_new = region.faces(0.0, radius(s[j]))
        w = (z_new[-1] - z_old[-1]) / dt
        solids, contents = [], []
        for c_held in (0.0, 1.0):
            x, _ = _region_step(region, z_old, z_new, dt, times[j], _End(), _End(np.array([k * c_held]), w),
                                1e-12, 1, geometry)
            solids.append(x)
            contents.append(float(cell_measures(z_new, geometry) @ x[:, 0]))
        # solute balance: content(C_L) + C_L * liquid volume = C0 * total, content linear in C_L
        liquid_volume = total - z_new[-1] ** (p + 1) / (p + 1)
        a, b = contents[0], contents[1] - contents[0]
        liquid = (total - a) / (b + liquid_volume)
        region.x = solids[0] + liquid * (solids[1] - solids[0])
        z_old = z_new
        out_fs.append(s[j])
        out_l.append(liquid)
        out_i.append(k * liquid)
    return BackDiffusionResult(np.array(out_fs), np.array(out_l), np.array(out_i), z_old, region.x[:, 0].copy(),
                               geometry)


def _invert(frac: Callable[[float], float], lo_fs: float, hi_fs: float) -> Callable[[float], float]:
    """tau with frac(tau) = fs, by bisection on [0, 1]."""
    def tau_of(fs):
        if fs <= frac(0.0):
            return 0.0
        a, b = 0.0, 1.0
        for _ in range(100):
            m = 0.5 * (a + b)
            a, b = (m, b) if frac(m) < fs else (a, m)
        return 0.5 * (a + b)
    return tau_of


@dataclass
class ScheilResult:
    """Temperatures (K), fraction solid, liquid mole fractions (n_T, n) and the solid formed at each step:
    for every temperature step the phases that formed with their amounts (fraction of the alloy) and
    mole fractions."""

    temperatures: np.ndarray
    fraction_solid: np.ndarray
    liquid_composition: np.ndarray
    solids: list
    components: tuple

    def increments(self) -> tuple[np.ndarray, np.ndarray]:
        """(fraction-solid edges, independent mole fractions of the solid laid down in each step), the input
        :func:`segregation_profile` takes. A step with no solid (above the liquidus) is given the liquid's
        composition; its width is zero, so it carries no weight."""
        comps = []
        for formed, liquid in zip(self.solids, self.liquid_composition[:-1]):
            amount = sum(a for _, a, _ in formed)
            if amount > 0:
                comps.append(sum(a * x for _, a, x in formed)[:-1] / amount)
            else:
                comps.append(np.asarray(liquid, dtype=float))
        return self.fraction_solid.copy(), np.array(comps)

    def solid_phase_fractions(self) -> dict:
        """Amount of each solid phase formed over the whole path, as a fraction of the alloy."""
        totals: dict = {}
        for step in self.solids:
            for name, amount, _ in step:
                totals[name] = totals.get(name, 0.0) + amount
        return totals


def scheil_gulliver(db: Database, components: Sequence[str], phases: Sequence[str], composition: Sequence[float],
                    start: float, step: float = 1.0, liquid: str = "LIQUID", stop_fraction: float = 1e-4,
                    t_min: float = 300.0) -> ScheilResult:
    """Scheil-Gulliver solidification by the algorithm Bocklund et al., Materialia 11 (2020) 100689, give
    (section 2): from ``start`` (K, above the liquidus) the remaining liquid is equilibrated at each
    temperature, the solid that forms is set aside and never changes again, the liquid's fraction is
    reduced by the solid's, and its composition becomes that of the equilibrium liquid, until less than
    ``stop_fraction`` liquid is left. ``composition`` is the n-1 independent mole fractions in the order of
    ``components`` (the last element dependent).
    """
    comps = [c.upper() for c in components]
    x = np.asarray(composition, dtype=float)
    if x.size != len(comps) - 1:
        raise ValueError("give the n-1 independent mole fractions of the alloy")
    f_liquid, T = 1.0, float(start)
    temps, fs, xl, solids = [T], [0.0], [x.copy()], []
    while f_liquid > stop_fraction and T > t_min:
        T -= step
        eq = equilibrium(db, comps + ["VA"], list(phases),
                         {v.T: T, v.P: 101325, v.N: 1, **{v.X(c): xi for c, xi in zip(comps[:-1], x)}})
        names = [str(p) for p in np.ravel(eq.Phase.values)]
        amounts = np.ravel(eq.NP.values)
        comp = eq.X.values.reshape(len(names), -1)
        order = [list(map(str, eq.component.values)).index(c) for c in comps]
        liquid_amount = sum(a for n, a in zip(names, amounts) if n == liquid)
        formed = [(n, a * f_liquid, comp[i][order]) for i, (n, a) in enumerate(zip(names, amounts))
                  if n and n != liquid and a > 0]
        if liquid_amount <= 0:
            solids.append(formed)
            f_liquid = 0.0
        else:
            x_liq = sum(comp[i][order] * a for i, (n, a) in enumerate(zip(names, amounts)) if n == liquid) / liquid_amount
            solids.append(formed)
            f_liquid *= min(liquid_amount, 1.0)  # above the liquidus pycalphad may return 1 + 1e-16
            x = x_liq[:-1]
        temps.append(T)
        fs.append(1 - f_liquid)
        xl.append(x.copy())
    return ScheilResult(np.array(temps), np.array(fs), np.array(xl), solids, tuple(comps))


def segregation_profile(fs_edges, solid_composition, length: float, n_cells: int,
                        geometry: Geometry = "planar") -> tuple[np.ndarray, np.ndarray]:
    """Face positions (m) and cell-averaged solid composition (n_cells, m) across a volume element of size
    ``length`` from the solid laid down in increments: increment i, of composition ``solid_composition[i]``
    (m independent fractions), filled the fraction solid from ``fs_edges[i]`` to ``fs_edges[i + 1]``,
    counted by volume. The solid grows from r = 0, so fraction f sits at r = length f^(1/(p+1)). Solid not
    yet formed when the record ends (fs_edges[-1] < 1) takes the last increment's composition. This turns
    a Scheil or back-diffusion record into the starting profile of a homogenization anneal."""
    edges = np.asarray(fs_edges, dtype=float)
    comp = np.asarray(solid_composition, dtype=float)
    comp = comp[:, None] if comp.ndim == 1 else comp
    if edges.size != comp.shape[0] + 1 or np.any(np.diff(edges) < 0):
        raise ValueError("need one more non-decreasing fraction-solid edge than increments")
    if edges[-1] < 1:
        edges = np.append(edges, 1.0)
        comp = np.vstack([comp, comp[-1]])
    p = {"planar": 0, "cylindrical": 1, "spherical": 2}[geometry]
    faces = np.linspace(0.0, length, n_cells + 1)
    cell_fs = (faces / length) ** (p + 1)  # volume fraction enclosed by each face
    # amount of each increment inside each cell, by overlap of their fraction-solid intervals
    lo = np.maximum(cell_fs[:-1, None], edges[None, :-1])
    hi = np.minimum(cell_fs[1:, None], edges[None, 1:])
    overlap = np.clip(hi - lo, 0.0, None)
    return faces, (overlap @ comp) / overlap.sum(axis=1, keepdims=True)


def segregation_index(x) -> np.ndarray:
    """Residual segregation (max - min) of each profile in ``x`` (times, cells, ...) relative to the first,
    the index homogenization treatments are judged by."""
    x = np.asarray(x, dtype=float)
    spread = x.max(axis=1) - x.min(axis=1)
    return spread / spread[0]


@dataclass
class SolidificationResult:
    """Times (s), temperatures (K), fraction solid (by volume of the element), the liquid's mole fractions (t, n-1),
    the interface solid's (t, n-1), the solid's faces (m) and compositions at each saved time, and whether the
    liquid ran out (``finished``)."""

    times: np.ndarray
    temperature: np.ndarray
    fraction_solid: np.ndarray
    liquid: np.ndarray
    interface_solid: np.ndarray
    solid_faces: list
    solid: list
    finished: bool


def equilibrium_start(db: Database, components: Sequence[str], solid_phase: str, liquid_phase: str,
                      alloy: Sequence[float], T: float) -> tuple[float, np.ndarray, np.ndarray]:
    """(fraction solid, solid and liquid independent mole fractions) of the solid-liquid equilibrium of ``alloy``
    at ``T`` from pycalphad, a start for :func:`solve_solidification` a little below the liquidus."""
    comps = [c.upper() for c in components]
    x = np.asarray(alloy, dtype=float)
    eq = equilibrium(db, comps + ["VA"], [solid_phase, liquid_phase],
                     {v.T: T, v.P: 101325, v.N: 1, **{v.X(c): xi for c, xi in zip(comps[:-1], x)}})
    names = [str(p) for p in np.ravel(eq.Phase.values)]
    amounts = np.ravel(eq.NP.values)
    comp = eq.X.values.reshape(len(names), -1)
    order = [list(map(str, eq.component.values)).index(c) for c in comps]
    found = {n: (a, comp[i][order]) for i, (n, a) in enumerate(zip(names, amounts)) if n}
    if solid_phase not in found or liquid_phase not in found:
        raise ValueError(f"{solid_phase} and {liquid_phase} do not coexist at {T:g} K; start between solidus and liquidus")
    return float(found[solid_phase][0]), found[solid_phase][1][:-1], found[liquid_phase][1][:-1]


def solve_solidification(solid: SolutionPhase, liquid: SolutionPhase, temperature: TemperatureSchedule,
                         t_end: float, length: float, start: tuple[float, Sequence[float], Sequence[float]],
                         solid_diffusivity=None, cells: int = 200, grading: float = 20.0, dt: float = 0.05,
                         save_times=None, geometry: Geometry = "planar", liquid_diffusivity: float = 1e-6,
                         liquid_cells: int = 4, tol: float = 1e-8, tolerance: float | None = None) -> SolidificationResult:
    """Solidification with back-diffusion in the solid and local equilibrium at its interface with the liquid, for
    any number of elements, the volume-element calculation of DICTRA's solidification runs (Borgenstam et al.,
    J. Phase Equilib. 21 (2000) 269; Lippard et al., Metall. Mater. Trans. B 29 (1998) 205). The primary solid
    grows from r = 0 (a plate, or the axis of a cylinder or centre of a sphere) across a volume element of
    ``length``, half the dendrite arm spacing, as ``temperature`` falls; the liquid beyond it is kept uniform (a
    diffusivity ``liquid_diffusivity`` far above the solid's), the interface follows the tie-line of local
    equilibrium at the current temperature, and nothing leaves the element.

    ``start`` is (fraction solid, solid and liquid independent mole fractions) at t = 0, for instance
    :func:`equilibrium_start` a little below the liquidus. ``solid_diffusivity`` is a constant matrix or a callable
    of (x) or (x, t); by default the solid's own interdiffusion matrix at the current temperature. The solid's
    ``cells`` are graded ``grading``:1 toward the interface. Only the primary solid forms: a run that should form a
    second solid (a eutectic) is beyond it. When the liquid runs out the run stops and returns what it has
    (``finished``). With no diffusion in the solid it follows the Scheil-Gulliver path, with fast diffusion the
    lever rule.
    """
    f0, xs0, xl0 = start
    xs0, xl0 = np.atleast_1d(np.asarray(xs0, dtype=float)), np.atleast_1d(np.asarray(xl0, dtype=float))
    if not 0 < f0 < 1:
        raise ValueError("the starting fraction solid must lie strictly between 0 and 1")
    if length <= 0 or t_end <= 0:
        raise ValueError("length and t_end must be positive")
    if not callable(temperature):
        raise ValueError("solidification needs a falling temperature: give a TemperatureSchedule")
    p = {"planar": 0, "cylindrical": 1, "spherical": 2}[geometry]
    s0 = length * f0 ** (1 / (p + 1))
    if solid_diffusivity is None:
        def solid_diffusivity(x, t):
            return solid.interdiffusion_matrix(x, temperature(t))
    m = xs0.size
    layers = [Layer(np.tile(xs0, (cells, 1)), solid_diffusivity, "solid", graded_faces(cells, grading, "right")),
              Layer(np.tile(xl0, (liquid_cells, 1)), np.eye(m) * liquid_diffusivity, "liquid")]
    cond = [LocalEquilibrium(solid, liquid, temperature, xs0, xl0)]
    finished = False
    try:
        res = solve_layers([0.0, s0, length], layers, cond, t_end, dt, save_times=save_times, tol=tol,
                           geometry=geometry, tolerance=tolerance)
    except (PhaseDissolvedError, ConvergenceError) as err:
        # the liquid running out ends solidification; any other failure is passed on
        if "liquid" not in str(err):
            raise
        res, finished = err.result, True
    widths = cell_measures(np.moveaxis(res.faces[1], 0, -1), geometry)
    liquid_x = np.einsum("it,tik->tk", widths, res.x[1]) / widths.sum(axis=0)[:, None]
    return SolidificationResult(res.times, temperature(res.times), (res.interfaces[:, 0] / length) ** (p + 1),
                                liquid_x, res.interface_left[:, 0], list(res.faces[0]), list(res.x[0]), finished)
