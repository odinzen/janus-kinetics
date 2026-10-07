"""Planar sharp-interface diffusion between two phases.

Phase L occupies 0 < z < S(t) and phase R occupies S(t) < z < length. Each phase
has its own grid, stretched uniformly as the interface moves, and is solved with
the same backward-Euler finite-volume scheme as :func:`solve_planar` plus the
flux carried by the moving faces. The interface sits at the compositions given by
an interface condition, and moves at the velocity that conserves every element:

    w (x_R^I - x_L^I) = J_R^I - J_L^I        (one equation per independent element)

which is eq. (8) of Zener, J. Appl. Phys. 20 (1949) 950, for any pair of phases.
With local equilibrium the interface compositions are not known in advance when
there are three or more elements: the tie-line is the one whose compositions make
the chemical potentials equal and the flux balances hold together, which gives
2(n-1) + 1 equations for x_L^I, x_R^I and w.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .errors import ConvergenceError
from .phase import SolutionPhase
from .solver import Boundary, Diffusivity, _as_callable, _Assembler, _output_schedule
from .thermo import R, dmu_dx, reduced_hessian


class PhaseDissolvedError(RuntimeError):
    """The interface reached an outer boundary, so one phase has vanished."""


@dataclass
class _Region:
    """One phase layer on faces z_faces (N+1), compositions x (N, m)."""

    dfun: Callable[[np.ndarray], np.ndarray]
    constant: bool
    xi: np.ndarray  # face positions mapped onto [0, 1]
    x: np.ndarray

    def faces(self, start: float, end: float) -> np.ndarray:
        return start + (end - start) * self.xi


@dataclass(frozen=True)
class _End:
    """A region end: no flux through a fixed face, or a held composition at a face moving at w."""

    composition: np.ndarray | None = None
    velocity: float = 0.0


def _region_step(region: _Region, z_old, z_new, dt, left: _End, right: _End, picard_tol, max_picard):
    """Advance one region by one step; return new compositions and the diffusive end fluxes.

    Cell balance (dz_new x_new - dz_old x_old)/dt + F_out - F_in = 0 with face flux
    F = J - v x_up, where v is the face velocity and x_up the composition on the side
    the material comes from relative to the face. With this face term a uniform
    composition stays uniform as the cells stretch.
    """
    x_old = region.x
    n, m = x_old.shape
    asm = _Assembler(z_new, m)
    v_inner = ((z_new - z_old) / dt)[1:-1]
    i, j = np.arange(n - 1), np.arange(1, n)
    up = np.where(v_inner > 0, j, i)
    a = np.arange(m)
    adv = sp.csc_matrix((
        np.concatenate([np.repeat(-v_inner, m), np.repeat(v_inner, m)]),
        (np.concatenate([(i[:, None] * m + a).ravel(), (j[:, None] * m + a).ravel()]),
         np.concatenate([(up[:, None] * m + a).ravel()] * 2)),
    ), shape=(n * m, n * m))

    rhs = (np.diff(z_old) / dt)[:, None] * x_old
    conductance, cells = {}, {"left": 0, "right": n - 1}
    for side, end, h, sign in (("left", left, asm.h_left, -1.0), ("right", right, asm.h_right, 1.0)):
        if end.composition is None:
            conductance[side] = None
            continue
        c = region.dfun(end.composition[None])[0] / h
        conductance[side] = c
        # face flux F = -c (x_cell - x_b) - v x_b on the left end, -c (x_b - x_cell) - v x_b on the right
        rhs[cells[side]] += c @ end.composition + sign * end.velocity * end.composition

    xk = x_old.copy()
    for k in range(1, max_picard + 1):
        c_inner = region.dfun(0.5 * (xk[1:] + xk[:-1])) / asm.h_inner[:, None, None]
        mat = asm.matrix(dt, c_inner, conductance["left"], conductance["right"]) + adv
        x_next = spla.spsolve(mat, rhs.ravel()).reshape(n, m)
        change = np.max(np.abs(x_next - xk))
        xk = x_next
        if region.constant or (k > 1 and change < picard_tol):
            break
    else:
        raise ConvergenceError(f"Picard iteration in a phase region did not converge (last change {change:.3g})")

    flux = {}
    if left.composition is not None:
        flux["left"] = -conductance["left"] @ (xk[0] - left.composition)
    if right.composition is not None:
        flux["right"] = -conductance["right"] @ (right.composition - xk[-1])
    return xk, flux


class FixedInterface:
    """Interface compositions held constant; only the velocity is solved for.

    Exact for a binary under local equilibrium at constant temperature, where the
    tie-line is unique, and the setting of Zener's analysis. With more elements a
    fixed pair of compositions cannot satisfy every flux balance at once.
    """

    def __init__(self, x_left, x_right):
        self.x_left = np.atleast_1d(np.asarray(x_left, dtype=float))
        self.x_right = np.atleast_1d(np.asarray(x_right, dtype=float))
        if self.x_left.size != 1:
            raise ValueError("FixedInterface is for binaries; use LocalEquilibrium with more elements")

    def initial(self) -> np.ndarray:
        return np.array([])

    def unpack(self, u):
        return self.x_left, self.x_right

    def residual(self, u):
        return np.array([])

    def jacobian(self, u):
        return np.zeros((0, 0))


class LocalEquilibrium:
    """Equal chemical potentials of every element across the interface.

    ``x_left`` and ``x_right`` are starting compositions for the tie-line (n-1
    independent fractions each, the dependent element last as in the phases), for
    example from :func:`pycalphad.equilibrium`. Both phases must list the same
    elements in the same order.
    """

    def __init__(self, left: SolutionPhase, right: SolutionPhase, T: float, x_left, x_right):
        if left.elements != right.elements:
            raise ValueError(f"element order differs: {left.elements} vs {right.elements}")
        self.left, self.right, self.T = left, right, T
        self._u = np.concatenate([np.atleast_1d(x_left), np.atleast_1d(x_right)]).astype(float)
        self.m = len(left.elements) - 1

    def initial(self) -> np.ndarray:
        return self._u.copy()

    def unpack(self, u):
        return u[: self.m], u[self.m : 2 * self.m]

    def _mu_and_derivative(self, phase: SolutionPhase, x):
        full = phase.lattice.full_composition(x)
        mu = phase.gibbs.chemical_potentials(full, self.T)
        # no stability check here: Newton iterates may pass near a spinodal on the way to the tie-line
        return mu, dmu_dx(full, reduced_hessian(phase.gibbs.hessian(full, self.T)))

    def residual(self, u):
        xl, xr = self.unpack(u)
        return (self._mu_and_derivative(self.left, xl)[0] - self._mu_and_derivative(self.right, xr)[0]) / (R * self.T)

    def jacobian(self, u):
        xl, xr = self.unpack(u)
        rt = R * self.T
        return np.hstack([self._mu_and_derivative(self.left, xl)[1] / rt, -self._mu_and_derivative(self.right, xr)[1] / rt])

    def tie_line(self, tol: float = 1e-12, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Solve the binary tie-line, where equal potentials alone fix both compositions."""
        if self.m != 1:
            raise ValueError("a tie-line is fixed by equal potentials alone only in a binary")
        u = self._u.copy()
        for _ in range(max_iter):
            step = np.linalg.solve(self.jacobian(u), -self.residual(u))
            u = u + step
            if np.max(np.abs(step)) < tol:
                return self.unpack(u)
        raise ConvergenceError("binary tie-line did not converge")


@dataclass
class MovingBoundaryResult:
    times: np.ndarray
    interface: np.ndarray  # S at each saved time
    z_left: np.ndarray  # faces [t, N_L+1]
    x_left: np.ndarray  # [t, N_L, m]
    z_right: np.ndarray
    x_right: np.ndarray
    interface_left: np.ndarray  # interface compositions [t, m]
    interface_right: np.ndarray

    def inventory(self) -> np.ndarray:
        """Integral of each independent mole fraction over both phases, shape (n_t, m)."""
        left = np.einsum("ti,tik->tk", np.diff(self.z_left, axis=1), self.x_left)
        right = np.einsum("ti,tik->tk", np.diff(self.z_right, axis=1), self.x_right)
        return left + right


def solve_moving_boundary(
    length: float,
    interface: float,
    x0_left,
    x0_right,
    d_left: Diffusivity,
    d_right: Diffusivity,
    condition: FixedInterface | LocalEquilibrium,
    t_end: float,
    dt: float,
    left: Boundary = Boundary.no_flux(),
    right: Boundary = Boundary.no_flux(),
    save_times=None,
    tol: float = 1e-10,
    picard_tol: float = 1e-10,
    max_newton: int = 30,
    max_picard: int = 50,
) -> MovingBoundaryResult:
    """Integrate two-phase planar diffusion with a moving interface.

    ``x0_left`` (N_L, n-1) and ``x0_right`` (N_R, n-1) are the initial profiles on
    uniform grids over [0, interface] and [interface, length]; a binary may pass
    1D arrays. ``d_left``/``d_right`` are constant matrices or callables like those
    of :func:`solve_planar`, for example ``lambda x: alpha.interdiffusion_matrix(x, T)``.
    ``left``/``right`` are the conditions at z = 0 and z = length.
    """
    def region(x0, d):
        x0 = np.asarray(x0, dtype=float)
        x0 = x0[:, None] if x0.ndim == 1 else x0.copy()
        dfun, constant = _as_callable(d)
        return _Region(dfun, constant, np.linspace(0.0, 1.0, len(x0) + 1), x0)

    reg_l, reg_r = region(x0_left, d_left), region(x0_right, d_right)
    m = reg_l.x.shape[1]
    if not 0.0 < interface < length:
        raise ValueError("the interface must lie inside (0, length)")

    def outer(b: Boundary) -> _End:
        return _End() if b.kind == "no_flux" else _End(b.composition.reshape(m), 0.0)

    end_0, end_l = outer(left), outer(right)
    u_iface = condition.initial()
    s, w_prev = interface, 0.0

    def attempt(u_iface, w, s_old, step):
        s_new = s_old + w * step
        if not 0.0 < s_new < length:
            raise PhaseDissolvedError(f"interface left the domain (S = {s_new:g} m); one phase has vanished")
        xl_i, xr_i = condition.unpack(u_iface)
        xl, fl = _region_step(reg_l, reg_l.faces(0.0, s_old), reg_l.faces(0.0, s_new), step,
                              end_0, _End(xl_i, w), picard_tol, max_picard)
        xr, fr = _region_step(reg_r, reg_r.faces(s_old, length), reg_r.faces(s_new, length), step,
                              _End(xr_i, w), end_l, picard_tol, max_picard)
        return s_new, xl, xr, w * (xr_i - xl_i) - (fr["left"] - fl["right"])

    def scale(s_old, step):
        # velocity in cells per step, so all Newton unknowns are of order one
        h = min(s_old / len(reg_l.x), (length - s_old) / len(reg_r.x))
        return h / step

    def solve_step(u_iface, w_guess, s_old, step):
        ws = scale(s_old, step)
        y = np.append(u_iface, w_guess / ws)
        n_if = len(u_iface)

        def full_residual(y):
            s_new, xl, xr, flux_res = attempt(y[:n_if], y[-1] * ws, s_old, step)
            return np.concatenate([condition.residual(y[:n_if]), flux_res / ws]), (s_new, xl, xr)

        def admissible(y):
            xl, xr = condition.unpack(y[:n_if])
            return all(np.all(v > 0) and v.sum() < 1 for v in (np.atleast_1d(xl), np.atleast_1d(xr)))

        def jacobian(y, res):
            jac = np.zeros((len(y), len(y)))
            jac[: len(res) - m, :n_if] = condition.jacobian(y[:n_if])
            for j in range(len(y)):
                hj = 1e-7 * max(1.0, abs(y[j]))
                yp = y.copy()
                yp[j] += hj
                jac[len(res) - m :, j] = (full_residual(yp)[0][len(res) - m :] - res[len(res) - m :]) / hj
            return jac

        # Chord Newton: each finite-difference column costs two region solves, so the
        # Jacobian is rebuilt only when the residual stops falling fast.
        jac, last = None, np.inf
        for _ in range(max_newton):
            res, state = full_residual(y)
            size = np.max(np.abs(res))
            if jac is None or size > 0.25 * last:
                jac = jacobian(y, res)
            last = size
            delta = np.linalg.solve(jac, -res)
            for _ in range(30):  # damp steps that would leave the composition simplex
                if admissible(y + delta):
                    break
                delta = 0.5 * delta
            y = y + delta
            if np.max(np.abs(delta)) < tol and np.max(np.abs(res)) < 1e3 * tol:
                res, state = full_residual(y)
                return y[:n_if], y[-1] * ws, state
        raise ConvergenceError(f"interface conditions did not converge at S = {s_old:g} m (residual {np.max(np.abs(res)):.3g})")

    def snapshot():
        xl_i, xr_i = condition.unpack(u_iface)
        return (s, reg_l.faces(0.0, s), reg_l.x.copy(), reg_r.faces(s, length), reg_r.x.copy(), xl_i.copy(), xr_i.copy())

    saved, times, t = [snapshot()], [0.0], 0.0
    for target, steps in _output_schedule(t_end, dt, save_times):
        step = (target - t) / steps
        for _ in range(steps):
            u_iface, w_prev, (s, reg_l.x, reg_r.x) = solve_step(u_iface, w_prev, s, step)
        t = target
        times.append(t)
        saved.append(snapshot())
    cols = list(zip(*saved))
    return MovingBoundaryResult(np.array(times), np.array(cols[0]), np.array(cols[1]), np.array(cols[2]),
                                np.array(cols[3]), np.array(cols[4]), np.array(cols[5]), np.array(cols[6]))
