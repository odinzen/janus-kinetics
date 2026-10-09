"""Implicit finite-volume solver for multicomponent interdiffusion in a plate, cylinder or sphere.

Solves dx_k/dt = r^-p d/dr ( r^p sum_j D_kj(x) dx_j/dr ) for the n-1 independent mole
fractions in the number-fixed frame with constant molar volume (Andersson and
Agren, J. Appl. Phys. 72 (1992) 1350). Time stepping is backward Euler; the
composition dependence of D is handled by Picard iteration, with D evaluated at
cell faces from the arithmetic mean of the neighbouring cells. The face-flux
form conserves the discrete inventory sum_i w_i x_i under no-flux boundaries,
whatever the Picard iterate, up to floating-point round-off. The exponent p is 0 for a
plate, 1 for a long cylinder and 2 for a sphere, the one-variable geometries DICTRA
offers (Borgenstam et al., J. Phase Equilib. 21 (2000) 269); cells then hold volume
(r_out^(p+1) - r_in^(p+1)) / (p+1) and faces carry area r^p, per unit angle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

from .errors import ConvergenceError
from .schedule import temperature_at, with_time

Diffusivity = Callable[[np.ndarray], np.ndarray] | np.ndarray
LatticeVelocity = Callable[[np.ndarray, np.ndarray], np.ndarray]
Geometry = Literal["planar", "cylindrical", "spherical"]
_EXPONENT = {"planar": 0, "cylindrical": 1, "spherical": 2}


def cell_measures(z_faces: np.ndarray, geometry: Geometry = "planar") -> np.ndarray:
    """Cell volumes per unit area (plate), per radian and unit length (cylinder) or per steradian
    (sphere): (r_out^(p+1) - r_in^(p+1)) / (p+1)."""
    p = _EXPONENT[geometry]
    z = np.asarray(z_faces, dtype=float)
    return (z[1:] ** (p + 1) - z[:-1] ** (p + 1)) / (p + 1)


@dataclass(frozen=True)
class Boundary:
    """Planar boundary condition: zero flux, or some or all components held at the face.

    Build it with :meth:`no_flux`, :meth:`fixed` or :meth:`potential`. Compositions
    are the n-1 independent fractions in the order the phase lists them. With
    ``held`` naming a subset of components, only those are held and the others
    pass no flux through the face.
    """

    kind: Literal["no_flux", "fixed"]
    composition: np.ndarray | Callable | None = None
    held: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.kind not in ("no_flux", "fixed"):
            raise ValueError(f"boundary kind must be 'no_flux' or 'fixed', got {self.kind!r}")
        if self.kind == "no_flux":
            return
        if self.composition is None:
            raise ValueError("a fixed boundary needs a composition")
        if not callable(self.composition):
            comp = np.atleast_1d(np.asarray(self.composition, dtype=float))
            if np.any(comp < 0) or np.any(comp > 1) or comp.sum() > 1:
                raise ValueError(f"boundary composition {comp} is not a set of mole fractions")
            object.__setattr__(self, "composition", comp)
        if self.held is not None:
            object.__setattr__(self, "held", tuple(int(i) for i in self.held))

    @classmethod
    def no_flux(cls) -> Boundary:
        """Zero flux of every element through the face."""
        return cls("no_flux")

    @classmethod
    def fixed(cls, composition, components=None) -> Boundary:
        """Composition held at the face (independent fractions), constant or a function of time.

        ``components`` lists the indices held; the rest pass no flux. A callable
        composition takes (t) or (t, x_cell), x_cell being the adjacent cell.
        """
        return cls("fixed", composition, None if components is None else tuple(components))

    @classmethod
    def potential(cls, phase, potentials: dict, temperature) -> Boundary:
        """Hold the chemical potentials of some components at the face, as a gas atmosphere does.

        ``potentials`` maps element names to mu in J/mol, constant or a function of
        time. Each step the held components take the composition whose potentials
        match, with the other components at the adjacent cell's composition; those
        pass no flux. Works with :class:`SolutionPhase` and :class:`InterstitialPhase`.
        """
        independent = list(getattr(phase, "independent", None) or phase.elements[:-1])
        full = list(getattr(phase, "components", None) or phase.elements)
        names = [k.upper() for k in potentials]
        idx = [independent.index(k) for k in names]
        rows = [full.index(k) for k in names]
        targets = list(potentials.values())
        dmu = getattr(phase, "dmu_du", None) or phase.dmu_dx

        def composition(t, x_cell):
            temp = temperature_at(temperature, t)
            goal = np.array([temperature_at(m, t) for m in targets])
            x = np.array(x_cell, dtype=float)
            s = np.log(np.clip(x[idx], 1e-12, None))  # Newton in ln x: mu ~ RT ln x when dilute
            for _ in range(60):
                x[idx] = np.exp(s)
                res = phase.chemical_potentials(x, temp)[rows] - goal
                jac = dmu(x, temp, check=False) if dmu.__name__ == "dmu_du" else dmu(x, temp)
                step = np.linalg.solve(jac[rows][:, idx] * x[idx], -res)
                step = np.clip(step, -2.0, 2.0)
                s = s + step
                if np.max(np.abs(step)) < 1e-12:
                    x[idx] = np.exp(s)
                    return x
            raise ConvergenceError(f"no composition gives potentials {dict(zip(names, goal))} at T = {temp:g} K")

        return cls("fixed", composition, tuple(idx))

    def face(self, t: float, x_cell: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
        """Face composition and the mask of held components at time t."""
        comp = self.composition
        if callable(comp):
            comp = np.atleast_1d(np.asarray(with_time(comp, 1)(t, x_cell), dtype=float))
        if comp.size != m:
            raise ValueError(f"a fixed boundary needs the {m} independent mole fractions, got {comp.size}")
        mask = np.ones(m, dtype=bool) if self.held is None else np.isin(np.arange(m), self.held)
        return comp, mask


@dataclass
class DiffusionResult:
    """Saved profiles: ``x[t_index, cell, component]`` for the independent components."""

    times: np.ndarray
    z_faces: np.ndarray
    x: np.ndarray
    picard_iterations: list[int] = field(default_factory=list)
    markers: np.ndarray | None = None  # marker positions [t_index, marker] when tracked
    geometry: Geometry = "planar"

    @property
    def z(self) -> np.ndarray:
        return 0.5 * (self.z_faces[1:] + self.z_faces[:-1])

    @property
    def widths(self) -> np.ndarray:
        """Cell measures: widths for a plate, volumes per unit angle for a cylinder or sphere."""
        return cell_measures(self.z_faces, self.geometry)

    def inventory(self) -> np.ndarray:
        """Integral of each mole fraction over the domain (r^p dr) at every saved time, shape (n_t, n-1)."""
        return np.einsum("i,tik->tk", self.widths, self.x)


def uniform_faces(length: float, n_cells: int, origin: float = 0.0) -> np.ndarray:
    """Face positions (m) of ``n_cells`` equal cells spanning [origin, origin + length]."""
    return origin + np.linspace(0.0, length, n_cells + 1)


def _as_callable(diffusivity: Diffusivity) -> tuple[Callable[[np.ndarray, float], np.ndarray], bool]:
    """Diffusivity as a function of (x, t), and whether it is a constant matrix."""
    if callable(diffusivity):
        return with_time(diffusivity, 1), False
    d = np.asarray(diffusivity, dtype=float)
    d = d.reshape(1, 1) if d.ndim == 0 else d
    return (lambda x, t: np.broadcast_to(d, x.shape[:-1] + d.shape)), True


class _BlockTridiagonal:
    """Matrix of (m, m) blocks on three block diagonals, the shape every 1D finite-volume
    operator here has: diag[i], lower[i] at block (i+1, i) and upper[i] at block (i, i+1).

    Solved as a banded system; that avoids building a general sparse matrix and is
    several times faster at the sizes these solvers use.
    """

    def __init__(self, diag: np.ndarray, lower: np.ndarray, upper: np.ndarray):
        self.diag, self.lower, self.upper = diag, lower, upper
        self.n, self.m = diag.shape[0], diag.shape[1]

    def matvec(self, x: np.ndarray) -> np.ndarray:
        xb = x.reshape(self.n, self.m)
        out = np.einsum("iab,ib->ia", self.diag, xb)
        out[1:] += np.einsum("iab,ib->ia", self.lower, xb[:-1])
        out[:-1] += np.einsum("iab,ib->ia", self.upper, xb[1:])
        return out.ravel()

    def __matmul__(self, x):
        return self.matvec(x)

    _layout_cache: dict = {}

    @classmethod
    def _layout(cls, n: int, m: int):
        """Row and column of every block entry, in the order diag, lower, upper (fixed per shape)."""
        key = (n, m)
        if key not in cls._layout_cache:
            a, b = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
            pairs = [(np.arange(n), np.arange(n)), (np.arange(1, n), np.arange(n - 1)), (np.arange(n - 1), np.arange(1, n))]
            rows = np.concatenate([(r[:, None, None] * m + a).ravel() for r, _ in pairs])
            cols = np.concatenate([(c[:, None, None] * m + b).ravel() for _, c in pairs])
            cls._layout_cache[key] = (rows, cols)
        return cls._layout_cache[key]

    def _entries(self):
        rows, cols = self._layout(self.n, self.m)
        vals = np.concatenate([self.diag.ravel(), self.lower.ravel(), self.upper.ravel()])
        return rows, cols, vals

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        band = 2 * self.m - 1
        rows, cols, vals = self._entries()
        ab = np.zeros((2 * band + 1, self.n * self.m))
        ab[band + rows - cols, cols] = vals  # every (row, col) occurs once
        return solve_banded((band, band), ab, rhs)

    def to_csc(self):
        rows, cols, vals = self._entries()
        size = self.n * self.m
        return sp.csc_matrix((vals, (rows, cols)), shape=(size, size))


class _Assembler:
    """Builds the backward-Euler matrix for given face diffusivities."""

    def __init__(self, z_faces: np.ndarray, m: int, geometry: Geometry = "planar"):
        self.m = m
        self.n_cells = len(z_faces) - 1
        self.widths = cell_measures(z_faces, geometry)
        p = _EXPONENT[geometry]
        self.area_inner = z_faces[1:-1] ** p
        self.area_left, self.area_right = z_faces[0] ** p, z_faces[-1] ** p
        centers = 0.5 * (z_faces[1:] + z_faces[:-1])
        self.h_inner = np.diff(centers)
        self.h_left = centers[0] - z_faces[0]
        self.h_right = z_faces[-1] - centers[-1]
        a, b = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        self._a, self._b = a.ravel(), b.ravel()

    def _blocks(self, cells_r, cells_c, blocks):
        rows = (cells_r[:, None] * self.m + self._a).ravel()
        cols = (cells_c[:, None] * self.m + self._b).ravel()
        return rows, cols, blocks.reshape(len(cells_r), -1).ravel()

    def matrix(self, dt: float, c_inner: np.ndarray, c_left: np.ndarray | None, c_right: np.ndarray | None):
        """c_* are face conductances D/h, shape (faces, m, m)."""
        n, m = self.n_cells, self.m
        diag = np.broadcast_to(np.eye(m), (n, m, m)) * (self.widths / dt)[:, None, None]
        diag[:-1] += c_inner
        diag[1:] += c_inner
        if c_left is not None:
            diag[0] += c_left
        if c_right is not None:
            diag[-1] += c_right
        return _BlockTridiagonal(diag, -c_inner.copy(), -c_inner.copy())


def _output_schedule(t_end: float, dt: float, save_times) -> list[tuple[float, int]]:
    """Pair each save time with the number of equal sub-steps that reach it."""
    targets = np.unique(np.append(np.asarray(save_times if save_times is not None else [], dtype=float), t_end))
    if targets[0] <= 0 or targets[-1] > t_end:
        raise ValueError("save times must lie in (0, t_end]")
    schedule, t = [], 0.0
    for target in targets:
        steps = max(1, int(np.ceil((target - t) / dt - 1e-9)))
        schedule.append((target, steps))
        t = target
    return schedule


def _march(x, advance, t_end: float, dt: float, save_times, tolerance: float | None, scale: float = 1.0,
           on_step=None, on_save=None, norm=None):
    """Step ``x`` from t = 0 through the save times to ``t_end``; returns (times, saved states, iterations).

    ``advance(x, step, t_new)`` returns the new state and its iteration count, or raises
    ConvergenceError. Without a tolerance each interval between save times is cut into equal
    steps no longer than ``dt``. With one, backward Euler's local error, about
    (step^2 / 2) d^2x/dt^2, is estimated from the change in dx/dt between consecutive steps
    (largest over the cells, divided by ``scale``); a step above ``tolerance``, or one that does
    not converge, is halved and redone, and one under an eighth of it is followed by one twice
    as long. Steps are dt / 2^k, starting at dt / 1024. ``on_step(x, t_old, t_new)`` runs after
    each step and ``on_save(t)`` at each save time. ``norm(d)``, if given, replaces the largest
    |d| / scale as the size of a change d in the state.
    """
    if tolerance is not None and tolerance <= 0:
        raise ValueError("tolerance must be positive")
    times, saved, iters = [0.0], [x.copy()], []
    t = 0.0
    h = dt / 1024  # the adaptive step, always dt / 2^k
    previous = None  # (dx/dt, step) of the last accepted adaptive step
    for target, steps in _output_schedule(t_end, dt, save_times):
        if tolerance is None:
            step = (target - t) / steps
            for i in range(steps):
                t_new = t + (i + 1) * step
                x, k = advance(x, step, t_new)
                iters.append(k)
                if on_step:
                    on_step(x, t_new - step, t_new)
        while tolerance is not None and t < target:
            step = min(h, target - t)
            t_new = target if step == target - t else t + step
            try:
                x_new, k = advance(x, step, t_new)
            except ConvergenceError:
                x_new = None
            error = 0.0
            if x_new is not None and previous is not None and norm is None:
                error = (np.max(np.abs((x_new - x) / step - previous[0])) * step**2
                         / (step + previous[1]) / scale)
            elif x_new is not None and previous is not None:
                error = norm(((x_new - x) / step - previous[0]) * step**2 / (step + previous[1]))
            if x_new is None or error > tolerance:
                while h >= step:
                    h /= 2
                if h < dt * 2.0**-60:
                    raise ConvergenceError(f"no step meets tolerance {tolerance:g} at t={t:g} s")
                continue
            previous = ((x_new - x) / step, step)
            x, t = x_new, t_new
            iters.append(k)
            if on_step:
                on_step(x, t - step, t)
            if error < tolerance / 8:
                h = min(2 * h, dt)
        t = target
        times.append(t)
        saved.append(x.copy())
        if on_save:
            on_save(t)
    return times, saved, iters


def solve_planar(
    z_faces: np.ndarray,
    x0: np.ndarray,
    diffusivity: Diffusivity,
    t_end: float,
    dt: float,
    left: Boundary = Boundary.no_flux(),
    right: Boundary = Boundary.no_flux(),
    save_times=None,
    picard_tol: float = 1e-10,
    max_picard: int = 50,
    markers=None,
    lattice_velocity: LatticeVelocity | None = None,
    geometry: Geometry = "planar",
    tolerance: float | None = None,
) -> DiffusionResult:
    """Integrate interdiffusion from ``x0`` to ``t_end`` in a plate (default), a long cylinder or a sphere.

    Parameters
    ----------
    z_faces : cell face positions (m), strictly increasing, length N+1.
    x0 : initial independent mole fractions, shape (N, n-1) or (N,) for a binary.
    diffusivity : constant (n-1, n-1) matrix in m^2/s, or a callable mapping
        compositions (M, n-1) to matrices (M, n-1, n-1), such as
        ``lambda x: phase.interdiffusion_matrix(x, T)``. A callable of (x, t) is
        evaluated at the end of each step, which runs a non-isothermal anneal;
        ``phase.interdiffusion_function(schedule)`` builds one.
    dt : maximum time step (s); steps are shortened to land on ``save_times``.
    picard_tol : convergence threshold on the max change in the variables between
        Picard iterates, relative to their largest initial magnitude when that
        exceeds 1 (concentrations in mol/m^3, say); absolute for mole fractions.
    markers : initial positions (m) of inert Kirkendall markers to follow.
    lattice_velocity : callable (x, dx_dz) or (x, dx_dz, t) -> v giving the lattice
        velocity at the interior faces, such as ``lambda x, g: phase.lattice_velocity(x, g, T)``.
        Required with ``markers``.
    geometry : "planar", "cylindrical" or "spherical"; for the curved ones ``z_faces`` are
        radii, and a first face at r = 0 is the axis or centre, where no flux passes.
        Markers are followed in the planar geometry only.
    tolerance : with it the step adapts, ``dt`` becoming the largest step. Backward Euler
        errs by about (step^2 / 2) d^2x/dt^2 per step; the second derivative is taken from
        the change in dx/dt between consecutive steps, and a step whose estimate exceeds
        ``tolerance`` (in the units of ``picard_tol``) is halved and redone, as is one whose
        Picard iteration fails. A step whose estimate is under an eighth of ``tolerance``
        is followed by one twice as long. Steps are dt / 2^k, so a constant D needs few
        factorizations; the first is dt / 1024.

    Returns profiles at t=0 and at every save time (always including ``t_end``).
    """
    z_faces = np.asarray(z_faces, dtype=float)
    if np.any(np.diff(z_faces) <= 0):
        raise ValueError("z_faces must be strictly increasing")
    if geometry not in _EXPONENT:
        raise ValueError(f"geometry must be one of {sorted(_EXPONENT)}, got {geometry!r}")
    if geometry != "planar":
        if z_faces[0] < 0:
            raise ValueError("radii cannot be negative")
        if z_faces[0] == 0 and left.kind != "no_flux":
            raise ValueError("nothing can be held at r = 0; the centre takes a no-flux boundary")
        if markers is not None:
            raise ValueError("markers are followed in the planar geometry only")
    x = np.asarray(x0, dtype=float)
    x = x[:, None] if x.ndim == 1 else x.copy()
    n_cells, m = x.shape
    if n_cells != len(z_faces) - 1:
        raise ValueError("x0 needs one row per cell")
    dfun, constant = _as_callable(diffusivity)
    asm = _Assembler(z_faces, m, geometry)
    scale = max(1.0, float(np.max(np.abs(x))))

    for b in (left, right):
        if b.kind == "fixed" and not callable(b.composition) and b.composition.size != m:
            raise ValueError(f"a fixed boundary needs the {m} independent mole fractions, got {b.composition.size}")

    def boundary(b: Boundary, h: float, t: float, x_cell: np.ndarray):
        """Face conductance and held composition. Components that are not held get zero
        flux, so their face values are eliminated: the held ones see the Schur complement
        D_hh - D_hf D_ff^-1 D_fh of the diffusivity matrix."""
        if b.kind == "no_flux":
            return None, None
        xb, held = b.face(t, x_cell, m)
        d = dfun(np.where(held, xb, x_cell)[None], t)[0]
        if held.all():
            return d / h, xb
        free = ~held
        schur = d[np.ix_(held, held)] - d[np.ix_(held, free)] @ np.linalg.solve(d[np.ix_(free, free)], d[np.ix_(free, held)])
        c = np.zeros((m, m))
        c[np.ix_(held, held)] = schur / h
        return c, np.where(held, xb, 0.0)

    def step_matrix(xk: np.ndarray, step: float, t: float):
        faces = 0.5 * (xk[1:] + xk[:-1])
        c_inner = dfun(faces, t) * (asm.area_inner / asm.h_inner)[:, None, None]
        c_left, xl = boundary(left, asm.h_left / asm.area_left if asm.area_left > 0 else np.inf, t, xk[0])
        c_right, xr = boundary(right, asm.h_right / asm.area_right, t, xk[-1])
        rhs_bc = np.zeros_like(xk)
        if c_left is not None:
            rhs_bc[0] += c_left @ xl
        if c_right is not None:
            rhs_bc[-1] += c_right @ xr
        return asm.matrix(step, c_inner, c_left, c_right), rhs_bc

    cache: dict[float, tuple] = {}
    # held compositions that change with time or with the adjacent cell need the Picard path
    moving_boundary_values = any(b.kind == "fixed" and (callable(b.composition) or b.held is not None)
                                 for b in (left, right))

    def advance(x_old: np.ndarray, step: float, t_new: float) -> tuple[np.ndarray, int]:
        # Each solve is for the increment from the current iterate, so solver round-off
        # scales with the (small) increment and the inventory drifts far less than it
        # would if the full composition were solved for.
        rhs = ((asm.widths / step)[:, None] * x_old).ravel()
        if constant and not moving_boundary_values:
            # A constant D gives the same matrix every step, so factorize once per step size.
            if step not in cache:
                a, rhs_bc = step_matrix(x_old, step, t_new)
                cache[step] = (a, spla.splu(a.to_csc()), rhs_bc.ravel())
            a, lu, rhs_bc = cache[step]
            xv = x_old.ravel()
            return (xv + lu.solve(rhs + rhs_bc - a @ xv)).reshape(n_cells, m), 1
        xk = x_old.ravel()
        for k in range(1, max_picard + 1):
            a, rhs_bc = step_matrix(xk.reshape(n_cells, m), step, t_new)
            delta = a.solve(rhs + rhs_bc.ravel() - a @ xk)
            xk = xk + delta
            change = np.max(np.abs(delta)) / scale if k > 1 else np.inf
            if change < picard_tol:
                return xk.reshape(n_cells, m), k
        raise ConvergenceError(
            f"Picard iteration did not reach {picard_tol:g} in {max_picard} iterations "
            f"at t={t_new:g} s (last change {change:.3g}); reduce dt"
        )

    tracker = None
    if markers is not None:
        if lattice_velocity is None:
            raise ValueError("tracking markers needs lattice_velocity")
        tracker = _MarkerTracker(np.atleast_1d(np.asarray(markers, dtype=float)), z_faces, with_time(lattice_velocity, 2))

    marker_track = [tracker.z.copy()] if tracker else None

    def on_step(x_new, t_old, t_new):
        if tracker:
            tracker.advance(x_new, t_old, t_new)
        if len(cache) > 64:  # adaptive steps shortened to land on save times each add a factorization
            cache.clear()

    times, saved, iters = _march(x, advance, t_end, dt, save_times, tolerance, scale, on_step,
                                 (lambda t: marker_track.append(tracker.z.copy())) if tracker else None)
    return DiffusionResult(np.array(times), z_faces, np.array(saved), iters,
                           np.array(marker_track) if tracker else None, geometry)


class _MarkerTracker:
    """Moves inert markers with the lattice, dz/dt = v(z, t).

    Integrated in tau = sqrt(t) as dz/dtau = 2 tau v by the trapezoid rule. In a
    diffusion couple v falls as 1/sqrt(t) and markers move parabolically (van Loo,
    Prog. Solid State Chem. 20 (1990) 47, eq. (5)), so 2 tau v is constant and the
    rule is exact there; integrating in t instead would carry an O(sqrt(dt/t))
    error from the singular start. The first interval uses its end value alone for
    the same reason.
    """

    def __init__(self, z0: np.ndarray, z_faces: np.ndarray, velocity: LatticeVelocity):
        self.z = z0.copy()
        self.velocity = velocity
        self.face_z = z_faces[1:-1]
        centers = 0.5 * (z_faces[1:] + z_faces[:-1])
        self.h = np.diff(centers)
        self.g_old = None

    def _g(self, x: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        faces = 0.5 * (x[1:] + x[:-1])
        grad = np.diff(x, axis=0) / self.h[:, None]
        v = self.velocity(faces, grad, t)
        return 2.0 * np.sqrt(t) * np.interp(z, self.face_z, v)

    def advance(self, x: np.ndarray, t_old: float, t_new: float) -> None:
        dtau = np.sqrt(t_new) - np.sqrt(t_old)
        g_new = self._g(x, self.z, t_new)
        g_old = g_new if self.g_old is None else self.g_old
        z_pred = self.z + dtau * 0.5 * (g_old + g_new)
        g_new = self._g(x, z_pred, t_new)  # one corrector pass at the predicted position
        g_old = g_new if self.g_old is None else self.g_old
        self.z = self.z + dtau * 0.5 * (g_old + g_new)
        self.g_old = self._g(x, self.z, t_new)
