"""Implicit finite-volume solver for planar multicomponent interdiffusion.

Solves dx_k/dt = d/dz ( sum_j D_kj(x) dx_j/dz ) for the n-1 independent mole
fractions in the number-fixed frame with constant molar volume (Andersson and
Agren, J. Appl. Phys. 72 (1992) 1350). Time stepping is backward Euler; the
composition dependence of D is handled by Picard iteration, with D evaluated at
cell faces from the arithmetic mean of the neighbouring cells. The face-flux
form conserves the discrete inventory sum_i w_i x_i under no-flux boundaries,
whatever the Picard iterate, up to floating-point round-off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .errors import ConvergenceError
from .schedule import with_time

Diffusivity = Callable[[np.ndarray], np.ndarray] | np.ndarray
LatticeVelocity = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Boundary:
    """Planar boundary condition, either zero flux or a fixed composition held at the face."""

    kind: Literal["no_flux", "fixed"]
    composition: np.ndarray | None = None

    @classmethod
    def no_flux(cls) -> Boundary:
        return cls("no_flux")

    @classmethod
    def fixed(cls, composition) -> Boundary:
        return cls("fixed", np.atleast_1d(np.asarray(composition, dtype=float)))


@dataclass
class DiffusionResult:
    """Saved profiles: ``x[t_index, cell, component]`` for the independent components."""

    times: np.ndarray
    z_faces: np.ndarray
    x: np.ndarray
    picard_iterations: list[int] = field(default_factory=list)
    markers: np.ndarray | None = None  # marker positions [t_index, marker] when tracked

    @property
    def z(self) -> np.ndarray:
        return 0.5 * (self.z_faces[1:] + self.z_faces[:-1])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.z_faces)

    def inventory(self) -> np.ndarray:
        """Integral of each mole fraction over the domain at every saved time, shape (n_t, n-1)."""
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


class _Assembler:
    """Builds the sparse backward-Euler matrix for given face diffusivities."""

    def __init__(self, z_faces: np.ndarray, m: int):
        self.m = m
        self.n_cells = len(z_faces) - 1
        self.widths = np.diff(z_faces)
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
        left, right = np.arange(n - 1), np.arange(1, n)
        parts = [
            self._blocks(left, left, c_inner),
            self._blocks(right, right, c_inner),
            self._blocks(left, right, -c_inner),
            self._blocks(right, left, -c_inner),
        ]
        if c_left is not None:
            parts.append(self._blocks(np.array([0]), np.array([0]), c_left[None]))
        if c_right is not None:
            parts.append(self._blocks(np.array([n - 1]), np.array([n - 1]), c_right[None]))
        rows, cols, vals = (np.concatenate(p) for p in zip(*parts))
        mass = np.repeat(self.widths / dt, m)
        idx = np.arange(n * m)
        rows, cols, vals = np.concatenate([rows, idx]), np.concatenate([cols, idx]), np.concatenate([vals, mass])
        return sp.csc_matrix((vals, (rows, cols)), shape=(n * m, n * m))


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
) -> DiffusionResult:
    """Integrate planar interdiffusion from ``x0`` to ``t_end``.

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
    picard_tol : convergence threshold on the max change in mole fraction
        between Picard iterates.
    markers : initial positions (m) of inert Kirkendall markers to follow.
    lattice_velocity : callable (x, dx_dz) or (x, dx_dz, t) -> v giving the lattice
        velocity at the interior faces, such as ``lambda x, g: phase.lattice_velocity(x, g, T)``.
        Required with ``markers``.

    Returns profiles at t=0 and at every save time (always including ``t_end``).
    """
    z_faces = np.asarray(z_faces, dtype=float)
    if np.any(np.diff(z_faces) <= 0):
        raise ValueError("z_faces must be strictly increasing")
    x = np.asarray(x0, dtype=float)
    x = x[:, None] if x.ndim == 1 else x.copy()
    n_cells, m = x.shape
    if n_cells != len(z_faces) - 1:
        raise ValueError("x0 needs one row per cell")
    dfun, constant = _as_callable(diffusivity)
    asm = _Assembler(z_faces, m)

    def boundary(b: Boundary, h: float, t: float):
        if b.kind == "no_flux":
            return None, None
        xb = b.composition.reshape(m)
        return dfun(xb[None], t)[0] / h, xb

    def step_matrix(xk: np.ndarray, step: float, t: float):
        faces = 0.5 * (xk[1:] + xk[:-1])
        c_inner = dfun(faces, t) / asm.h_inner[:, None, None]
        c_left, xl = boundary(left, asm.h_left, t)
        c_right, xr = boundary(right, asm.h_right, t)
        rhs_bc = np.zeros_like(xk)
        if c_left is not None:
            rhs_bc[0] += c_left @ xl
        if c_right is not None:
            rhs_bc[-1] += c_right @ xr
        return asm.matrix(step, c_inner, c_left, c_right), rhs_bc

    cache: dict[float, tuple] = {}

    def advance(x_old: np.ndarray, step: float, t_new: float) -> tuple[np.ndarray, int]:
        # Each solve is for the increment from the current iterate, so solver round-off
        # scales with the (small) increment and the inventory drifts far less than it
        # would if the full composition were solved for.
        rhs = ((asm.widths / step)[:, None] * x_old).ravel()
        if constant:
            # A constant D gives the same matrix every step, so factorize once per step size.
            if step not in cache:
                a, rhs_bc = step_matrix(x_old, step, t_new)
                cache[step] = (a, spla.splu(a), rhs_bc.ravel())
            a, lu, rhs_bc = cache[step]
            xv = x_old.ravel()
            return (xv + lu.solve(rhs + rhs_bc - a @ xv)).reshape(n_cells, m), 1
        xk = x_old.ravel()
        for k in range(1, max_picard + 1):
            a, rhs_bc = step_matrix(xk.reshape(n_cells, m), step, t_new)
            delta = spla.spsolve(a, rhs + rhs_bc.ravel() - a @ xk)
            xk = xk + delta
            change = np.max(np.abs(delta)) if k > 1 else np.inf
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

    times, saved, iters = [0.0], [x.copy()], []
    marker_track = [tracker.z.copy()] if tracker else None
    t = 0.0
    for target, steps in _output_schedule(t_end, dt, save_times):
        step = (target - t) / steps
        for i in range(steps):
            t_new = t + (i + 1) * step
            x, k = advance(x, step, t_new)
            iters.append(k)
            if tracker:
                tracker.advance(x, t_new - step, t_new)
        t = target
        times.append(t)
        saved.append(x.copy())
        if tracker:
            marker_track.append(tracker.z.copy())
    return DiffusionResult(np.array(times), z_faces, np.array(saved), iters,
                           np.array(marker_track) if tracker else None)


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
