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
grid of compositions computed once and interpolated linearly. Larsson and
Hoglund (section 2.5) fill such a grid on demand instead, because a complete
table becomes impractical for four or more elements; for the binaries and
ternaries handled here the complete table is affordable. Their small ideal flux
term against oscillations in multiphase regions (section 2.4, eq. 13) is not
included.

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
from .solver import Boundary, DiffusionResult, _output_schedule

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
            raise ValueError("the tabulated local equilibrium is limited to binaries and ternaries")
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

        points, mus, log_gammas, fractions = [], [], [], []
        for idx in np.ndindex(mu.shape[:-1]):
            x_ind = np.array([axis[i] for i in idx])
            if x_ind.sum() >= 1.0 - 0.5 * spacing or not np.all(np.isfinite(mu[idx])):
                continue
            present = [(str(ph), f, xp) for ph, f, xp in zip(phase[idx], np_[idx], xph[idx]) if ph and f > 0]
            gam = np.array([phases[ph].mobility(xp, T) * xp for ph, _, xp in present])
            frac = np.array([f for _, f, _ in present])
            points.append(x_ind)
            mus.append(mu[idx])
            log_gammas.append(np.log(average_kinetic_factor(gam, frac / frac.sum(), averaging)))
            fractions.append([sum(f for ph, f, _ in present if ph == name) for name in self.phase_names])
        self.points = np.array(points)
        values = np.hstack([np.array(mus), np.array(log_gammas), np.array(fractions)])
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
    table: LocalEquilibriumTable | TemperatureTable,
    t_end: float,
    dt: float,
    left: Boundary = Boundary.no_flux(),
    right: Boundary = Boundary.no_flux(),
    save_times=None,
    tol: float = 1e-10,
    max_iter: int = 60,
    temperature: Temperature | None = None,
) -> DiffusionResult:
    """Integrate planar multiphase diffusion with the homogenization model.

    Backward Euler on cell compositions. Each iteration linearizes the potentials
    in the composition change (Newton) and holds Gamma* and the frame correction
    at the current iterate (Picard). Face fluxes use the potential difference
    between neighbouring cells, so transport across a phase boundary is driven by
    the true potential jump rather than by a diffusivity at a mean composition.
    With a :class:`TemperatureTable`, ``temperature`` (a constant or a schedule)
    sets the table used for each step.
    """
    if isinstance(table, TemperatureTable):
        if temperature is None:
            raise ValueError("a TemperatureTable needs a temperature or schedule")
        table_at = lambda t: table.at(temperature_at(temperature, t))
    else:
        if temperature is not None:
            raise ValueError("a single-temperature table cannot follow a schedule; build a TemperatureTable")
        table_at = lambda t: table
    z_faces = np.asarray(z_faces, dtype=float)
    x = np.asarray(x0, dtype=float)
    x = x[:, None] if x.ndim == 1 else x.copy()
    n_cells, m = x.shape
    n = m + 1
    widths = np.diff(z_faces)
    centers = 0.5 * (z_faces[1:] + z_faces[:-1])
    h_inner = np.diff(centers)
    h_end = {"left": centers[0] - z_faces[0], "right": z_faces[-1] - centers[-1]}
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
                c = 0 if side == "left" else n_cells - 1
                a = operator(xb, 0.5 * (gb + gam[c]), np.array(h_end[side]))
                ag = a @ g[c]
                put(np.array([c]), np.array([c]), ag[None])
                rhs[c] -= a @ (mu[c] - mub)
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
    return DiffusionResult(np.array(times), z_faces, np.array(saved), iters)
