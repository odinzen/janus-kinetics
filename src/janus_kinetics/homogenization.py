"""Multiphase diffusion by the homogenization model.

Larsson and Hoglund, Calphad 33 (2009) 495. Each volume element is assumed to be
in local equilibrium at its overall composition, so it carries one set of
chemical potentials however many phases it holds. The flux of every element is
driven by that potential gradient and carried by an average of the phases'
kinetic factors Gamma^phi_k = M^phi_k u^phi_k (their eqs. (2)-(5)):

    J_k = -(1/V_S) Gamma*_k dmu_k/dz,     J'_k = J_k - u_k sum_j J_j

The second form is the number-fixed (volume-fixed at equal partial volumes)
frame of their eq. (3). Gamma* is one of the Wiener or Hashin-Shtrikman bounds,
eqs. (6)-(11). Local equilibrium comes from :func:`pycalphad.equilibrium` on a
grid of compositions, either computed once and interpolated linearly
(:class:`LocalEquilibriumTable`, binaries and ternaries) or filled on demand as
in their section 2.5 (:class:`OnDemandEquilibriumTable`, any number of
elements, since a complete table becomes impractical beyond three). Their small ideal flux
term against oscillations in multiphase regions (section 2.4, eq. 13) is the
``ideal_flux`` option of :func:`solve_homogenized`.

Only substitutional phases are handled, so u-fractions equal mole fractions.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pycalphad import Database, equilibrium, variables as v
from scipy.interpolate import LinearNDInterpolator

from .errors import ConvergenceError
from .phase import SolutionPhase
from .schedule import Temperature, temperature_at
from .thermo import R
from .solver import _EXPONENT, Boundary, DiffusionResult, Geometry, _output_schedule, cell_measures

Averaging = Literal["wiener_upper", "wiener_lower", "hashin_shtrikman_upper", "hashin_shtrikman_lower"]


def average_kinetic_factor(gamma: np.ndarray, fraction: np.ndarray, how: Averaging) -> np.ndarray:
    """Gamma* from per-phase Gamma (P, n) and volume fractions (P,), shape (n,).

    Larsson and Hoglund (2009): Wiener upper and lower bounds, eqs. (6)-(7), and the
    Hashin-Shtrikman bounds of eqs. (8)-(11), where the reference phase alpha is the
    one with the lowest (lower bound) or highest (upper bound) Gamma_k present.
    """
    gamma = np.asarray(gamma, dtype=float)
    f = np.asarray(fraction, dtype=float)[:, None]
    if how == "wiener_upper":
        return (f * gamma).sum(axis=0)
    if how == "wiener_lower":
        return 1.0 / (f / gamma).sum(axis=0)
    if how not in ("hashin_shtrikman_upper", "hashin_shtrikman_lower"):
        raise ValueError(f"unknown averaging {how!r}")
    pick = np.argmax if how == "hashin_shtrikman_upper" else np.argmin
    cols = np.arange(gamma.shape[1])
    ref = gamma[pick(gamma, axis=0), cols]
    diff = gamma - ref
    with np.errstate(divide="ignore"):
        inv = np.where(diff == 0.0, np.inf, 1.0 / np.where(diff == 0.0, 1.0, diff))
    a = (f / (inv + 1.0 / (3.0 * ref))).sum(axis=0)  # the reference phase contributes zero
    return ref + a / (1.0 - a / (3.0 * ref))


def _local_values(phases, T, averaging, phase_names, phase_row, fraction_row, x_rows, mu) -> np.ndarray:
    """[mu (n), ln Gamma* (n), phase fractions (P)] at one composition from its equilibrium."""
    present = [(str(ph), f, xp) for ph, f, xp in zip(phase_row, fraction_row, x_rows) if ph and f > 0]
    gam = np.array([phases[ph].mobility(xp, T) * xp for ph, _, xp in present])
    frac = np.array([f for _, f, _ in present])
    fractions = [sum(f for ph, f, _ in present if ph == name) for name in phase_names]
    return np.concatenate([mu, np.log(average_kinetic_factor(gam, frac / frac.sum(), averaging)), fractions])


class LocalEquilibriumTable:
    """Chemical potentials and Gamma* at local equilibrium, tabulated over composition.

    ``phases`` maps pycalphad phase names to :class:`SolutionPhase` objects that
    supply each phase's mobilities; all must list the same elements in the same
    order. ``spacing`` is the grid step in mole fraction (default 0.0025 for a
    binary, 0.02 for a ternary). Compositions outside the tabulated range raise
    instead of being extrapolated, so pure-element ends need a finer spacing.
    """

    def __init__(
        self,
        db: Database,
        phases: dict[str, SolutionPhase],
        T: float,
        averaging: Averaging = "hashin_shtrikman_lower",
        spacing: float | None = None,
        pressure: float = 101325.0,
    ):
        orders = {p.elements for p in phases.values()}
        if len(orders) != 1:
            raise ValueError(f"phases list different elements or orders: {orders}")
        self.elements = orders.pop()
        self.n = len(self.elements)
        self.m = self.n - 1
        if self.m > 2:
            raise ValueError("the tabulated local equilibrium is limited to binaries and ternaries; "
                             "use OnDemandEquilibriumTable for more elements")
        if spacing is None:
            # the ternary grid grows as 1/spacing^2 and pycalphad cost with it
            spacing = 0.0025 if self.m == 1 else 0.02
        self.T, self.spacing, self.averaging = T, spacing, averaging
        self.phase_names = sorted(phases)

        axis = np.arange(spacing, 1.0 - 0.5 * spacing, spacing)
        conds = {v.T: T, v.P: pressure, v.N: 1}
        conds.update({v.X(el): axis for el in self.elements[: self.m]})
        eq = equilibrium(db, list(self.elements) + ["VA"], self.phase_names, conds)
        dims = [f"X_{el}" for el in self.elements[: self.m]]
        comp = [list(eq.component.values).index(el) for el in self.elements]
        phase = eq.Phase.squeeze(["N", "P", "T"]).transpose(*dims, "vertex").values
        np_ = eq.NP.squeeze(["N", "P", "T"]).transpose(*dims, "vertex").values
        xph = eq.X.squeeze(["N", "P", "T"]).transpose(*dims, "vertex", "component").values[..., comp]
        mu = eq.MU.squeeze(["N", "P", "T"]).transpose(*dims, "component").values[..., comp]

        points, rows = [], []
        for idx in np.ndindex(mu.shape[:-1]):
            x_ind = np.array([axis[i] for i in idx])
            if x_ind.sum() >= 1.0 - 0.5 * spacing or not np.all(np.isfinite(mu[idx])):
                continue
            points.append(x_ind)
            rows.append(_local_values(phases, T, averaging, self.phase_names, phase[idx], np_[idx], xph[idx], mu[idx]))
        self.points = np.array(points)
        values = np.array(rows)
        if self.m == 1:
            order = np.argsort(self.points[:, 0])
            xp, vals = self.points[order, 0], values[order]
            self._interp = lambda x: np.column_stack([np.interp(x[:, 0], xp, vals[:, c]) for c in range(vals.shape[1])])
        else:
            self._interp = LinearNDInterpolator(self.points, values)
        self.lower, self.upper = self.points.min(axis=0), self.points.max(axis=0)
        self.max_sum = float(self.points.sum(axis=1).max())

    def _values(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        tol = 1e-12
        inside = np.all(x >= self.lower - tol, axis=1) & np.all(x <= self.upper + tol, axis=1) \
            & (x.sum(axis=1) <= self.max_sum + tol)
        out = self._interp(x)
        if not np.all(inside) or not np.all(np.isfinite(out)):
            bad = x[~inside | ~np.all(np.isfinite(out), axis=1)][0]
            raise ValueError(f"composition {bad} is outside the tabulated range {self.lower}..{self.upper}; "
                             "rebuild the table with a smaller spacing to reach it")
        return out

    def chemical_potentials(self, x) -> np.ndarray:
        """mu_k at local equilibrium, J/mol, shape (M, n); x holds independent fractions (M, n-1)."""
        return self._values(x)[:, : self.n]

    def kinetic_factor(self, x) -> np.ndarray:
        """Gamma*_k in m^2 mol / (J s), shape (M, n)."""
        return np.exp(self._values(x)[:, self.n : 2 * self.n])

    def phase_fractions(self, x) -> dict[str, np.ndarray]:
        """Volume (mole) fraction of each phase at local equilibrium, keyed by phase name."""
        vals = self._values(x)[:, 2 * self.n :]
        return {name: vals[:, i] for i, name in enumerate(self.phase_names)}

    def dmu_dx(self, x) -> np.ndarray:
        """Slopes of the interpolated potentials, shape (M, n, n-1).

        Central differences over a quarter of the grid step stay inside one linear
        piece away from grid lines, so they return that piece's slope.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        h = 0.25 * self.spacing
        cols = []
        for j in range(self.m):
            e = np.zeros(self.m)
            e[j] = h
            lo = np.clip(x - e, self.lower, None)
            hi = np.clip(x + e, None, self.upper)
            cols.append((self.chemical_potentials(hi) - self.chemical_potentials(lo)) / (hi[:, j] - lo[:, j])[:, None])
        return np.stack(cols, axis=-1)

    def admissible(self, x: np.ndarray) -> bool:
        return bool(np.all(x >= self.lower) and np.all(x <= self.upper) and np.all(x.sum(axis=-1) <= self.max_sum))


class OnDemandEquilibriumTable:
    """Local equilibrium computed only where a simulation goes, for any number of elements.

    Larsson and Hoglund (2009, section 2.5): composition space is divided into a
    grid of step ``spacing``, equilibrium is calculated at a grid node only when a
    composition near it is first requested, and node results are stored under the
    node's integer index (their hash table) for reuse. Between nodes, values are
    interpolated linearly on the Kuhn simplex of the surrounding grid cell, so the
    potentials are continuous and :meth:`dmu_dx` is the exact slope of that
    simplex; in a binary and on grid lines this equals the full table of
    :class:`LocalEquilibriumTable`. Only one temperature is held, as in the paper,
    where stored data are dropped when T changes. ``calculations`` counts the
    equilibria made. Nodes need every mole fraction, the dependent one included,
    to be at least one step from zero.
    """

    def __init__(
        self,
        db: Database,
        phases: dict[str, SolutionPhase],
        T: float,
        averaging: Averaging = "hashin_shtrikman_lower",
        spacing: float = 0.01,
        pressure: float = 101325.0,
    ):
        orders = {p.elements for p in phases.values()}
        if len(orders) != 1:
            raise ValueError(f"phases list different elements or orders: {orders}")
        if not 0 < spacing < 0.5:
            raise ValueError("spacing is a mole-fraction step in (0, 0.5)")
        self.elements = orders.pop()
        self.n = len(self.elements)
        self.m = self.n - 1
        self.db, self.phases, self.T = db, phases, float(T)
        self.spacing, self.averaging, self.pressure = spacing, averaging, pressure
        self.phase_names = sorted(phases)
        self._nodes: dict[tuple[int, ...], np.ndarray] = {}
        self._last = None
        self.calculations = 0
        self._top = int(np.floor(1.0 / spacing + 1e-9)) - 1  # largest index sum keeping the dependent >= one step

    def _valid(self, idx) -> bool:
        return min(idx) >= 1 and sum(idx) <= self._top

    def _node(self, idx: tuple[int, ...]) -> np.ndarray:
        stored = self._nodes.get(idx)
        if stored is not None:
            return stored
        if not self._valid(idx):
            raise ValueError(f"grid node {np.array(idx) * self.spacing} is outside the composition range; "
                             "use a smaller spacing to reach compositions this close to a pure element")
        x_node = np.array(idx) * self.spacing
        conds = {v.T: self.T, v.P: self.pressure, v.N: 1}
        conds.update({v.X(el): float(xi) for el, xi in zip(self.elements[: self.m], x_node)})
        eq = equilibrium(self.db, list(self.elements) + ["VA"], self.phase_names, conds)
        self.calculations += 1
        comp = [list(eq.component.values).index(el) for el in self.elements]
        squeeze = [d for d in eq.MU.dims if d not in ("component",)]
        mu = eq.MU.squeeze(squeeze).values[comp]
        if not np.all(np.isfinite(mu)):
            raise ValueError(f"equilibrium failed at grid node {x_node}")
        phase = eq.Phase.values.reshape(-1)
        fraction = eq.NP.values.reshape(-1)
        x_rows = eq.X.values.reshape(-1, len(eq.component))[:, comp]
        values = _local_values(self.phases, self.T, self.averaging, self.phase_names, phase, fraction, x_rows, mu)
        self._nodes[idx] = values
        return values

    def _simplices(self, x):
        """Kuhn-simplex vertices (M, m+1, m), barycentric weights (M, m+1) and the order in
        which coordinates step up (M, m) for each composition."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        u = x / self.spacing
        base = np.floor(u + 1e-12).astype(int)
        r = np.clip(u - base, 0.0, 1.0)
        order = np.argsort(-r, axis=1, kind="stable")
        rs = np.take_along_axis(r, order, axis=1)
        steps = np.zeros((len(x), self.m + 1, self.m), dtype=int)
        rows = np.arange(len(x))
        for k in range(self.m):
            steps[:, k + 1] = steps[:, k]
            steps[rows, k + 1, order[:, k]] += 1
        vertices = base[:, None, :] + steps
        weights = np.concatenate([1.0 - rs[:, :1], rs[:, :-1] - rs[:, 1:], rs[:, -1:]], axis=1)
        return vertices, weights, order

    def _gather(self, x, slopes: bool = False):
        """The simplices of :meth:`_simplices` plus the node values at their vertices (M, m+1, K).

        Values need only the vertices with weight; slopes need the whole simplex.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        # the solver asks for potentials, slopes and Gamma* at the same compositions in turn
        key = (x.tobytes(), slopes)
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        vertices, weights, order = self._simplices(x)
        if not slopes:
            # a composition on a grid line or node gives some vertices zero weight; they may
            # lie outside the grid, so stand the first vertex in for them
            vertices = np.where((weights == 0.0)[:, :, None], vertices[:, :1, :], vertices)
        keys, inverse = np.unique(vertices.reshape(-1, self.m), axis=0, return_inverse=True)
        table = np.array([self._node(tuple(int(i) for i in key)) for key in keys])
        result = vertices, weights, order, table[inverse.reshape(-1)].reshape(len(x), self.m + 1, -1)
        self._last = (key, result)
        return result

    def _values(self, x) -> np.ndarray:
        _, weights, _, node_values = self._gather(x)
        return np.einsum("mv,mvk->mk", weights, node_values)

    def chemical_potentials(self, x) -> np.ndarray:
        """mu_k at local equilibrium, J/mol, shape (M, n); x holds independent fractions (M, n-1)."""
        return self._values(x)[:, : self.n]

    def kinetic_factor(self, x) -> np.ndarray:
        """Gamma*_k in m^2 mol / (J s), shape (M, n), from ln Gamma* interpolated linearly."""
        return np.exp(self._values(x)[:, self.n : 2 * self.n])

    def phase_fractions(self, x) -> dict[str, np.ndarray]:
        """Mole fraction of each phase at local equilibrium, keyed by phase name."""
        vals = self._values(x)[:, 2 * self.n :]
        return {name: vals[:, i] for i, name in enumerate(self.phase_names)}

    def dmu_dx(self, x) -> np.ndarray:
        """Exact slopes of the interpolated potentials on each point's simplex, shape (M, n, n-1).

        On a grid line the simplex on the upper side is used.
        """
        _, _, order, node_values = self._gather(x, slopes=True)
        mu = node_values[:, :, : self.n]
        g = np.empty((len(mu), self.n, self.m))
        rows = np.arange(len(mu))
        for k in range(self.m):
            g[rows, :, order[:, k]] = (mu[:, k + 1] - mu[:, k]) / self.spacing
        return g

    def admissible(self, x: np.ndarray) -> bool:
        vertices, _, _ = self._simplices(x)
        base = vertices[:, 0]
        return bool(np.all(base >= 1) and np.all(vertices.sum(axis=2) <= self._top))


class TemperatureTable:
    """Local-equilibrium tables at several temperatures, for anneals along a schedule.

    Holds one :class:`LocalEquilibriumTable` per temperature in ``temperatures``
    (binaries only, to keep the cost to one pycalphad grid per temperature). At an
    intermediate T, chemical potentials are interpolated linearly in T and ln Gamma*
    linearly in 1/T, which is exact for Arrhenius mobilities. Space the
    temperatures so that phase boundaries move little between neighbours.
    """

    def __init__(self, db: Database, phases: dict[str, SolutionPhase], temperatures, **table_options):
        temps = np.sort(np.asarray(temperatures, dtype=float))
        if temps.size < 2:
            raise ValueError("give at least two temperatures; use LocalEquilibriumTable for one")
        if len(next(iter(phases.values())).elements) != 2:
            raise ValueError("temperature tables are limited to binaries")
        self.temperatures = temps
        self.tables = [LocalEquilibriumTable(db, phases, float(T), **table_options) for T in temps]

    def at(self, T: float) -> _TableAtTemperature:
        temps = self.temperatures
        if not temps[0] - 1e-9 <= T <= temps[-1] + 1e-9:
            raise ValueError(f"T = {T:g} K is outside the tabulated {temps[0]:g}-{temps[-1]:g} K")
        k = int(np.clip(np.searchsorted(temps, T) - 1, 0, temps.size - 2))
        lo, hi = temps[k], temps[k + 1]
        w = (T - lo) / (hi - lo)
        w_inv = (1 / T - 1 / lo) / (1 / hi - 1 / lo)
        return _TableAtTemperature(self.tables[k], self.tables[k + 1], w, w_inv)


class _TableAtTemperature:
    """Two neighbouring tables blended to one temperature; same interface as a table."""

    def __init__(self, lo: LocalEquilibriumTable, hi: LocalEquilibriumTable, w: float, w_inv: float):
        self.lo, self.hi, self.w, self.w_inv = lo, hi, w, w_inv

    def chemical_potentials(self, x):
        return (1 - self.w) * self.lo.chemical_potentials(x) + self.w * self.hi.chemical_potentials(x)

    def kinetic_factor(self, x):
        return np.exp((1 - self.w_inv) * np.log(self.lo.kinetic_factor(x)) + self.w_inv * np.log(self.hi.kinetic_factor(x)))

    def dmu_dx(self, x):
        return (1 - self.w) * self.lo.dmu_dx(x) + self.w * self.hi.dmu_dx(x)

    def phase_fractions(self, x):
        a, b = self.lo.phase_fractions(x), self.hi.phase_fractions(x)
        return {k: (1 - self.w) * a[k] + self.w * b[k] for k in a}

    def admissible(self, x) -> bool:
        return self.lo.admissible(x) and self.hi.admissible(x)


def solve_homogenized(
    z_faces: np.ndarray,
    x0: np.ndarray,
    table: LocalEquilibriumTable | OnDemandEquilibriumTable | TemperatureTable,
    t_end: float,
    dt: float,
    left: Boundary = Boundary.no_flux(),
    right: Boundary = Boundary.no_flux(),
    save_times=None,
    tol: float = 1e-10,
    max_iter: int = 60,
    temperature: Temperature | None = None,
    ideal_flux: float = 0.0,
    geometry: Geometry = "planar",
) -> DiffusionResult:
    """Integrate multiphase diffusion with the homogenization model in a plate (default), a long
    cylinder or a sphere (``geometry``, radii and centre as for :func:`solve_planar`).

    Backward Euler on cell compositions. Each iteration linearizes the potentials
    in the composition change (Newton) and holds Gamma* and the frame correction
    at the current iterate (Picard). Face fluxes use the potential difference
    between neighbouring cells, so transport across a phase boundary is driven by
    the true potential jump rather than by a diffusivity at a mean composition.
    With a :class:`TemperatureTable`, ``temperature`` (a constant or a schedule)
    sets the table used for each step.

    ``ideal_flux`` is epsilon of Larsson and Hoglund's eq. 13, a flux
    -epsilon (Gamma*_k / u_k) RT du_k/dz added to every element. In a multiphase
    region a composition ripple need not change the potentials, so nothing damps
    it; this term does. They suggest 0.01-0.05, as small as removes oscillations,
    since it also adds numerical diffusion. It equals driving the flux with
    mu_k + epsilon RT ln u_k, which is how it enters here.
    """
    if ideal_flux < 0:
        raise ValueError("ideal_flux (epsilon) must be non-negative")
    if isinstance(table, TemperatureTable):
        if temperature is None:
            raise ValueError("a TemperatureTable needs a temperature or schedule")
        table_at = lambda t: table.at(temperature_at(temperature, t))
        temperature_of = lambda t: temperature_at(temperature, t)
    else:
        if temperature is not None:
            raise ValueError("a single-temperature table cannot follow a schedule; build a TemperatureTable")
        table_at = lambda t: table
        temperature_of = lambda t: table.T
    z_faces = np.asarray(z_faces, dtype=float)
    if geometry not in _EXPONENT:
        raise ValueError(f"geometry must be one of {sorted(_EXPONENT)}, got {geometry!r}")
    if geometry != "planar":
        if z_faces[0] < 0:
            raise ValueError("radii cannot be negative")
        if z_faces[0] == 0 and left.kind != "no_flux":
            raise ValueError("nothing can be held at r = 0; the centre takes a no-flux boundary")
    x = np.asarray(x0, dtype=float)
    x = x[:, None] if x.ndim == 1 else x.copy()
    n_cells, m = x.shape
    n = m + 1
    widths = cell_measures(z_faces, geometry)
    centers = 0.5 * (z_faces[1:] + z_faces[:-1])
    # a face of area r^p over a distance h conducts as a planar face over h / r^p
    p = _EXPONENT[geometry]
    h_inner = np.diff(centers) / z_faces[1:-1] ** p
    area_left = z_faces[0] ** p  # 1 for a plate, whatever z_faces[0]; 0 at a centre, which passes no flux
    h_end = {"left": (centers[0] - z_faces[0]) / area_left if area_left > 0 else np.inf,
             "right": (z_faces[-1] - centers[-1]) / z_faces[-1] ** p}
    ends = {}
    for side, b in (("left", left), ("right", right)):
        if b.kind == "fixed":
            if callable(b.composition) or b.held is not None:
                raise ValueError("held-subset, potential and time-dependent boundaries are supported by solve_planar only")
            if b.composition.size != m:
                raise ValueError(f"a fixed boundary needs the {m} independent mole fractions")
            ends[side] = b.composition.reshape(m)
    a_idx = np.arange(m)

    def operator(xb_face, gamma_face, h):
        """A = P diag(Gamma/h) with P_kj = delta_kj - x_k, shape (..., m, n)."""
        p = np.eye(n)[:m] - xb_face[..., :, None]
        return p * (gamma_face / h[..., None])[..., None, :]

    def step(x_old, dt_step, t_new):
        tab = table_at(t_new)
        xk = x_old.copy()
        for it in range(1, max_iter + 1):
            mu = tab.chemical_potentials(xk)
            g = tab.dmu_dx(xk)
            gam = tab.kinetic_factor(xk)
            if ideal_flux:
                rt = ideal_flux * R * temperature_of(t_new)
                full = np.column_stack([xk, 1.0 - xk.sum(axis=1)])
                mu = mu + rt * np.log(full)
                dln = np.zeros((n_cells, n, m))
                dln[:, a_idx, a_idx] = 1.0 / xk
                dln[:, -1, :] = -1.0 / full[:, -1:]
                g = g + rt * dln
            rows, cols, vals = [], [], []
            rhs = -(widths / dt_step)[:, None] * (xk - x_old)

            def put(r, c, blocks):
                rr = (r[:, None, None] * m + a_idx[None, :, None]).repeat(m, axis=2)
                cc = (c[:, None, None] * m + a_idx[None, None, :]).repeat(m, axis=1)
                rows.append(rr.ravel()), cols.append(cc.ravel()), vals.append(blocks.ravel())

            cells = np.arange(n_cells)
            put(cells, cells, np.broadcast_to(np.eye(m), (n_cells, m, m)) * (widths / dt_step)[:, None, None])
            if n_cells > 1:
                i, j = cells[:-1], cells[1:]
                a = operator(0.5 * (xk[i] + xk[j]), 0.5 * (gam[i] + gam[j]), h_inner)
                f0 = -np.einsum("fkn,fn->fk", a, mu[j] - mu[i])
                ag_i, ag_j = a @ g[i], a @ g[j]
                # flux F = f0 + A g_i d_i - A g_j d_j leaves cell i and enters cell j
                put(i, i, ag_i), put(i, j, -ag_j), put(j, i, -ag_i), put(j, j, ag_j)
                rhs[i] -= f0
                rhs[j] += f0
            for side, xb in ends.items():
                mub, gb = tab.chemical_potentials(xb[None])[0], tab.kinetic_factor(xb[None])[0]
                if ideal_flux:
                    mub = mub + ideal_flux * R * temperature_of(t_new) * np.log(np.append(xb, 1.0 - xb.sum()))
                c = 0 if side == "left" else n_cells - 1
                a = operator(xb, 0.5 * (gb + gam[c]), np.array(h_end[side]))
                ag = a @ g[c]
                put(np.array([c]), np.array([c]), ag[None])
                rhs[c] -= a @ (mu[c] - mub)
            # rhs is the residual of the discrete balance at xk; once it is below tol in
            # mole-fraction units the step has converged. This also ends Newton cycling
            # between two pieces of a piecewise-linear potential, where delta never shrinks.
            if it > 1 and np.max(np.abs(rhs) / (widths / dt_step)[:, None]) < tol:
                return xk, it
            mat = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(n_cells * m, n_cells * m))
            delta = spla.spsolve(mat, rhs.ravel()).reshape(n_cells, m)
            full_step = True
            for _ in range(30):  # keep iterates inside the table
                if tab.admissible(xk + delta):
                    break
                delta = 0.5 * delta
                full_step = False
            else:
                raise ConvergenceError("the composition left the tabulated range during a step; "
                                       "rebuild the table to cover it or reduce dt")
            xk = xk + delta
            # a damped step is small by construction, so it cannot signal convergence
            if full_step and np.max(np.abs(delta)) < tol:
                return xk, it
        raise ConvergenceError(f"homogenization step did not converge (last change {np.max(np.abs(delta)):.3g}); reduce dt")

    times, saved, iters, t = [0.0], [x.copy()], [], 0.0
    for target, steps in _output_schedule(t_end, dt, save_times):
        dt_step = (target - t) / steps
        for i in range(steps):
            x, k = step(x, dt_step, t + (i + 1) * dt_step)
            iters.append(k)
        t = target
        times.append(t)
        saved.append(x.copy())
    return DiffusionResult(np.array(times), z_faces, np.array(saved), iters, None, geometry)
