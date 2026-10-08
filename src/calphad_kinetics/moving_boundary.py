"""Sharp-interface diffusion between phase layers in a plate, a long cylinder or a sphere.

Phase L occupies 0 < z < S(t) and phase R occupies S(t) < z < length. Each phase
has its own grid, stretched uniformly as the interface moves, and is solved with
the same backward-Euler finite-volume scheme as :func:`solve_planar` plus the
flux carried by the moving faces. The interface sits at the compositions given by
an interface condition, and moves at the velocity that conserves every element:

    w (x_R^I - x_L^I) = J_R^I - J_L^I        (one equation per independent element)

the generalization of eq. (8) of Zener, J. Appl. Phys. 20 (1949) 950, to
diffusion on both sides of the interface.
With local equilibrium the interface compositions are not known in advance when
there are three or more elements: the tie-line is the one whose compositions make
the chemical potentials equal and the flux balances hold together, which gives
2(n-1) + 1 equations for x_L^I, x_R^I and w.

In a cylinder or sphere the positions are radii and the layers concentric shells,
as in DICTRA (Borgenstam et al., J. Phase Equilib. 21 (2000) 269). Each moving face
then sweeps the volume (r_new^(p+1) - r_old^(p+1)) / (p+1) per unit angle, and its
advective term uses that swept volume divided by the distance moved as the face
area, so a uniform composition stays uniform on a stretching grid. End fluxes are
returned per unit of that area, and the interface balance keeps its planar form.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .errors import ConvergenceError
from .phase import SolutionPhase
from .schedule import Temperature, temperature_at
from .solver import _EXPONENT, Boundary, Diffusivity, Geometry, _as_callable, _Assembler, _output_schedule, cell_measures
from .thermo import R, dmu_dx, reduced_hessian


class PhaseDissolvedError(RuntimeError):
    """The interface reached an outer boundary, so one phase has vanished."""


@dataclass
class _Region:
    """One phase layer on faces z_faces (N+1), compositions x (N, m)."""

    dfun: Callable[[np.ndarray, float], np.ndarray]
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


def _swept_area(z_old, z_new, p: int):
    """Volume swept per unit angle by faces moving from z_old to z_new, divided by the distance moved."""
    z_old, z_new = np.asarray(z_old, dtype=float), np.asarray(z_new, dtype=float)
    dz = z_new - z_old
    with np.errstate(divide="ignore", invalid="ignore"):
        swept = (z_new ** (p + 1) - z_old ** (p + 1)) / ((p + 1) * dz)
    return np.where(np.abs(dz) > 1e-14 * np.maximum(np.abs(z_new), 1e-300), swept, z_new**p)


def _region_step(region: _Region, z_old, z_new, dt, t, left: _End, right: _End, picard_tol, max_picard,
                 geometry: Geometry = "planar"):
    """Advance one region by one step; return new compositions and the diffusive end fluxes.

    Cell balance (dz_new x_new - dz_old x_old)/dt + F_out - F_in = 0 with face flux
    F = J - v x_up, where v is the face velocity and x_up the composition on the side
    the material comes from relative to the face. With this face term a uniform
    composition stays uniform as the cells stretch.
    """
    x_old = region.x
    n, m = x_old.shape
    p = _EXPONENT[geometry]
    asm = _Assembler(z_new, m, geometry)
    area = _swept_area(z_old, z_new, p)
    v_inner = ((z_new - z_old) / dt)[1:-1]
    # moving-face term -v x_up: from the right cell when the face moves right, else the left one
    eye = np.eye(m)
    right_up = (v_inner > 0)[:, None, None]
    vb = (v_inner * area[1:-1])[:, None, None] * eye
    adv_diag = np.zeros((n, m, m))
    adv_diag[:-1] -= np.where(right_up, 0.0, vb)  # cell i loses -v x_i through its right face
    adv_diag[1:] += np.where(right_up, vb, 0.0)  # cell j gains -v x_j through its left face
    adv_upper = -np.where(right_up, vb, 0.0)
    adv_lower = np.where(right_up, 0.0, vb)

    rhs = (cell_measures(z_old, geometry) / dt)[:, None] * x_old
    conductance, cells, end_area = {}, {"left": 0, "right": n - 1}, {"left": area[0], "right": area[-1]}
    for side, end, h, a_j, sign in (("left", left, asm.h_left, asm.area_left, -1.0),
                                    ("right", right, asm.h_right, asm.area_right, 1.0)):
        if end.composition is None:
            conductance[side] = None
            continue
        c = region.dfun(end.composition[None], t)[0] * a_j / h
        conductance[side] = c
        # face flux F = -c (x_cell - x_b) - v x_b on the left end, -c (x_b - x_cell) - v x_b on the right
        rhs[cells[side]] += c @ end.composition + sign * end.velocity * end_area[side] * end.composition

    xk = x_old.copy()
    if max_picard == 0:  # a single linear solve with D at the old profile, for Jacobian estimates
        max_picard, linear = 1, True
    else:
        linear = False
    for k in range(1, max_picard + 1):
        c_inner = region.dfun(0.5 * (xk[1:] + xk[:-1]), t) * (asm.area_inner / asm.h_inner)[:, None, None]
        mat = asm.matrix(dt, c_inner, conductance["left"], conductance["right"])
        mat.diag += adv_diag
        mat.upper += adv_upper
        mat.lower += adv_lower
        x_next = mat.solve(rhs.ravel()).reshape(n, m)
        change = np.max(np.abs(x_next - xk))
        xk = x_next
        if region.constant or linear or (k > 1 and change < picard_tol):
            break
    else:
        raise ConvergenceError(f"Picard iteration in a phase region did not converge (last change {change:.3g})")

    flux = {}  # per unit of the swept area, so the interface balance reads as in a plate
    if left.composition is not None:
        flux["left"] = -conductance["left"] @ (xk[0] - left.composition) / end_area["left"]
    if right.composition is not None:
        flux["right"] = -conductance["right"] @ (right.composition - xk[-1]) / end_area["right"]
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

    def prepare(self, t: float) -> None:
        pass

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
    example from :func:`pycalphad.equilibrium`. Newton follows the tie-line
    nearest the start, so when a phase has a miscibility gap the starting
    compositions decide which branch is found. Both phases must list the same
    elements in the same order. ``temperature`` is a constant or a
    :class:`TemperatureSchedule`; the tie-line follows it during an anneal.
    """

    def __init__(self, left: SolutionPhase, right: SolutionPhase, temperature: Temperature, x_left, x_right):
        if left.elements != right.elements:
            raise ValueError(f"element order differs: {left.elements} vs {right.elements}")
        self.left, self.right, self.temperature = left, right, temperature
        self.T = temperature_at(temperature, 0.0)
        self._u = np.concatenate([np.atleast_1d(x_left), np.atleast_1d(x_right)]).astype(float)
        self.m = len(left.elements) - 1

    def initial(self) -> np.ndarray:
        return self._u.copy()

    def prepare(self, t: float) -> None:
        """Set the temperature for the step ending at time t."""
        self.T = temperature_at(self.temperature, t)

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
            step *= min(1.0, 0.05 / max(np.max(np.abs(step)), 1e-300))  # at most 0.05 in mole fraction per step
            while np.any(u + step <= 0) or np.any(u + step >= 1):
                step *= 0.5
            u = u + step
            if np.max(np.abs(step)) < tol:
                return self.unpack(u)
        raise ConvergenceError("binary tie-line did not converge; start from the compositions pycalphad.equilibrium "
                               "gives for the intended two-phase field")


@dataclass
class Layer:
    """One phase layer of a planar stack.

    ``x0`` is the initial profile on a uniform grid across the layer, shape (N, n-1)
    or (N,) for a binary; ``diffusivity`` is a constant matrix or a callable of
    (x) or (x, t), as for :func:`solve_planar`.
    """

    x0: np.ndarray
    diffusivity: Diffusivity
    name: str = ""


@dataclass
class StackResult:
    """Saved state of a layered run; index [t] is a saved time, [j] a layer, [i] an interface."""

    times: np.ndarray
    interfaces: np.ndarray  # interface positions [t, i]
    faces: list  # per layer, face positions [t, N_j + 1]
    x: list  # per layer, compositions [t, N_j, m]
    interface_left: np.ndarray  # composition on the left side of each interface [t, i, m]
    interface_right: np.ndarray
    names: tuple = ()
    geometry: Geometry = "planar"

    def thickness(self) -> np.ndarray:
        """Layer thicknesses [t, j] in m."""
        return np.stack([f[:, -1] - f[:, 0] for f in self.faces], axis=1)

    def inventory(self) -> np.ndarray:
        """Integral of each independent mole fraction over the whole stack (r^p dr), shape (n_t, m)."""
        return sum(np.einsum("ti,tik->tk", cell_measures(f.T, self.geometry).T, x) for f, x in zip(self.faces, self.x))

    def profile(self, index: int = -1) -> tuple[np.ndarray, np.ndarray]:
        """Cell centres (m) and compositions through the whole stack at one saved time."""
        z = np.concatenate([0.5 * (f[index, 1:] + f[index, :-1]) for f in self.faces])
        return z, np.concatenate([x[index] for x in self.x])


def solve_layers(
    positions,
    layers: list[Layer],
    conditions: list,
    t_end: float,
    dt: float,
    left: Boundary = Boundary.no_flux(),
    right: Boundary = Boundary.no_flux(),
    save_times=None,
    tol: float = 1e-10,
    picard_tol: float = 1e-10,
    max_newton: int = 30,
    max_picard: int = 50,
    geometry: Geometry = "planar",
) -> StackResult:
    """Integrate diffusion through a stack of phase layers with moving interfaces, in a plate (default)
    or in concentric shells of a long cylinder or a sphere (``geometry``), positions then being radii.

    ``positions`` are the layer boundaries from 0 to the stack thickness (len(layers)
    + 1 values); the inner ones are the initial interfaces. ``conditions[i]`` holds
    interface i, between layers i and i+1, at fixed compositions or in local
    equilibrium. Every interface moves to conserve every element (the flux balance
    of the module docstring), and all interface velocities and operating
    tie-lines are solved together each step. A reaction layer must be present from
    the start, if only a few nanometres thick. If a layer is consumed,
    :class:`PhaseDissolvedError` is raised; it and any :class:`ConvergenceError`
    carry the results up to the last saved time as ``.result``.
    """
    pos = np.asarray(positions, dtype=float)
    n_layers = len(layers)
    if len(pos) != n_layers + 1 or len(conditions) != n_layers - 1:
        raise ValueError("need len(layers) + 1 positions and len(layers) - 1 interface conditions")
    if np.any(np.diff(pos) <= 0) or pos[0] != 0.0:
        raise ValueError("positions must start at 0 and increase")
    if geometry not in _EXPONENT:
        raise ValueError(f"geometry must be one of {sorted(_EXPONENT)}, got {geometry!r}")
    if geometry != "planar" and left.kind != "no_flux":
        raise ValueError("nothing can be held at r = 0; the centre takes a no-flux boundary")

    regions = []
    for layer in layers:
        x0 = np.asarray(layer.x0, dtype=float)
        x0 = x0[:, None] if x0.ndim == 1 else x0.copy()
        dfun, constant = _as_callable(layer.diffusivity)
        regions.append(_Region(dfun, constant, np.linspace(0.0, 1.0, len(x0) + 1), x0))
    m = regions[0].x.shape[1]
    length = pos[-1]
    n_if = n_layers - 1

    def outer(b: Boundary) -> _End:
        if b.kind == "fixed" and (callable(b.composition) or b.held is not None):
            raise ValueError("held-subset, potential and time-dependent boundaries are supported by solve_planar only")
        return _End() if b.kind == "no_flux" else _End(b.composition.reshape(m), 0.0)

    end_0, end_l = outer(left), outer(right)
    sizes = [len(c.initial()) for c in conditions]
    extra = [len(c.residual(c.initial())) for c in conditions]  # equilibrium equations per interface
    offsets = np.concatenate([[0], np.cumsum([k + 1 for k in sizes])]).astype(int)
    row_offsets = np.concatenate([[0], np.cumsum([e + m for e in extra])]).astype(int)
    flux_rows = np.concatenate([np.arange(row_offsets[i] + extra[i], row_offsets[i + 1]) for i in range(n_if)])
    s = pos[1:-1].copy()
    u_ifaces = [c.initial() for c in conditions]
    w_prev = np.zeros(n_if)

    def bounds(s_vec):
        return np.concatenate([[0.0], s_vec, [length]])

    def layer_step(j, b_old, b_new, iface_x, w, step, t_new, linear=False):
        lo = end_0 if j == 0 else _End(np.atleast_1d(iface_x[j - 1][1]), w[j - 1])
        hi = end_l if j == n_layers - 1 else _End(np.atleast_1d(iface_x[j][0]), w[j])
        reg = regions[j]
        return _region_step(reg, reg.faces(b_old[j], b_old[j + 1]), reg.faces(b_new[j], b_new[j + 1]),
                            step, t_new, lo, hi, picard_tol, 0 if linear else max_picard, geometry)

    saved_jacobian = [None]

    def solve_step(s_old, step, t_new):
        for c in conditions:
            c.prepare(t_new)
        b_old = bounds(s_old)
        widths = np.diff(b_old) / np.array([len(r.x) for r in regions])
        ws = np.minimum(widths[:-1], widths[1:]) / step  # velocity in cells per step keeps unknowns near one
        y = np.concatenate([np.append(u_ifaces[i], w_prev[i] / ws[i]) for i in range(n_if)])

        def unpack(y):
            us = [y[offsets[i]:offsets[i] + sizes[i]] for i in range(n_if)]
            w = np.array([y[offsets[i] + sizes[i]] for i in range(n_if)]) * ws
            return us, w

        def evaluate(y, base=None, changed=None, linear=False):
            """Layer results and residual; with ``base`` only the layers in ``changed`` are re-solved,
            and with ``linear`` each by one linear solve (enough for a Jacobian estimate)."""
            us, w = unpack(y)
            s_new = s_old + w * step
            b_new = bounds(s_new)
            if np.any(np.diff(b_new) <= 0):
                bad = int(np.argmin(np.diff(b_new)))
                raise PhaseDissolvedError(f"layer {bad} ({layers[bad].name or 'unnamed'}) was consumed near t = {t_new:g} s")
            iface_x = [conditions[i].unpack(us[i]) for i in range(n_if)]
            out = list(base) if base is not None else [None] * n_layers
            for j in (changed if base is not None else range(n_layers)):
                out[j] = layer_step(j, b_old, b_new, iface_x, w, step, t_new, linear)
            res = []
            for i in range(n_if):
                xl_i, xr_i = (np.atleast_1d(v) for v in iface_x[i])
                flux_bal = w[i] * (xr_i - xl_i) - (out[i + 1][1]["left"] - out[i][1]["right"])
                res.append(np.concatenate([conditions[i].residual(us[i]), flux_bal / ws[i]]))
            return np.concatenate(res), out, s_new

        def admissible(y):
            us, _ = unpack(y)
            for i in range(n_if):
                for v in conditions[i].unpack(us[i]):
                    v = np.atleast_1d(v)
                    if np.any(v <= 0) or v.sum() >= 1:
                        return False
            return True

        def jacobian(y, res, out):
            # differences of linear layer solves against a linear base, so the linearization
            # error cancels; Newton still converges on the full residual
            res, out, _ = evaluate(y, linear=True)
            jac = np.zeros((len(y), len(y)))
            us, _ = unpack(y)
            for i in range(n_if):
                if extra[i]:
                    jac[row_offsets[i]:row_offsets[i] + extra[i], offsets[i]:offsets[i] + sizes[i]] = conditions[i].jacobian(us[i])
            for i in range(n_if):
                for k in range(offsets[i], offsets[i + 1]):
                    hk = 1e-7 * max(1.0, abs(y[k]))
                    yp = y.copy()
                    yp[k] += hk
                    # the unknowns of interface i reach only layers i and i+1
                    rp, _, _ = evaluate(yp, base=out, changed=(i, i + 1), linear=True)
                    jac[flux_rows, k] = (rp[flux_rows] - res[flux_rows]) / hk
            return jac

        # Chord Newton: the Jacobian, kept from the previous step since interface conditions
        # change slowly, is rebuilt only when the residual stops falling fast.
        jac, last, fresh = saved_jacobian[0], np.inf, False
        for _ in range(max_newton):
            res, out, s_new = evaluate(y)
            size = np.max(np.abs(res))
            if jac is None or size > 0.25 * last:
                jac, fresh = jacobian(y, res, out), True
            last = size
            delta = np.linalg.solve(jac, -res)
            # interface compositions move at most 0.05 per iteration; larger jumps
            # land far from the tie-line and can stall the layer solves
            comp = np.concatenate([np.arange(offsets[i], offsets[i] + sizes[i]) for i in range(n_if)]).astype(int)
            if comp.size:
                delta *= min(1.0, 0.05 / max(np.max(np.abs(delta[comp])), 1e-300))
            for _ in range(30):  # damp steps that would leave the composition simplex
                if admissible(y + delta):
                    break
                delta = 0.5 * delta
            y = y + delta
            if np.max(np.abs(delta)) < tol and np.max(np.abs(res)) < tol:
                res, out, s_new = evaluate(y)
                # A reused Jacobian converges only linearly, which would leave a small flux
                # imbalance each step that adds up to a mass error; a few cheap chord steps
                # take the residual down to round-off.
                for _ in range(5):
                    if np.max(np.abs(res)) < 1e-3 * tol:
                        break
                    y = y + np.linalg.solve(jac, -res)
                    res, out, s_new = evaluate(y)
                us, w = unpack(y)
                saved_jacobian[0] = jac
                return [u.copy() for u in us], w, s_new, [o[0] for o in out]
        if not fresh:
            saved_jacobian[0] = None
            return solve_step(s_old, step, t_new)  # retry once with a Jacobian built for this step
        raise ConvergenceError(f"interface conditions did not converge near t = {t_new:g} s "
                               f"(residual {np.max(np.abs(res)):.3g}); reduce dt")

    def snapshot():
        b = bounds(s)
        iface_x = [conditions[i].unpack(u_ifaces[i]) for i in range(n_if)]
        return (s.copy(), [regions[j].faces(b[j], b[j + 1]) for j in range(n_layers)],
                [r.x.copy() for r in regions], [np.atleast_1d(a).copy() for a, _ in iface_x],
                [np.atleast_1d(c).copy() for _, c in iface_x])

    def collect(saved, times):
        k = len(saved)
        return StackResult(
            np.array(times), np.array([v[0] for v in saved]).reshape(k, n_if),
            [np.array([v[1][j] for v in saved]) for j in range(n_layers)],
            [np.array([v[2][j] for v in saved]) for j in range(n_layers)],
            np.array([v[3] for v in saved]).reshape(k, n_if, m),
            np.array([v[4] for v in saved]).reshape(k, n_if, m),
            tuple(layer.name for layer in layers), geometry)

    initial = np.diff(pos)
    saved, times, t = [snapshot()], [0.0], 0.0
    try:
        for target, steps in _output_schedule(t_end, dt, save_times):
            step = (target - t) / steps
            for k in range(steps):
                try:
                    u_ifaces, w_prev, s, xs = solve_step(s, step, t + (k + 1) * step)
                except ConvergenceError as err:
                    # a layer a few cells from vanishing makes the interface Newton fail
                    # before any thickness reaches zero; report that as what it is
                    left_frac = np.diff(bounds(s)) / initial
                    j = int(np.argmin(left_frac))
                    if left_frac[j] < 0.02:
                        raise PhaseDissolvedError(
                            f"layer {j} ({layers[j].name or 'unnamed'}) was nearly consumed "
                            f"({left_frac[j]:.2%} of its initial thickness) near t = {t + (k + 1) * step:g} s") from err
                    raise
                for reg, xj in zip(regions, xs):
                    reg.x = xj
            t = target
            times.append(t)
            saved.append(snapshot())
    except (PhaseDissolvedError, ConvergenceError) as err:
        err.result = collect(saved, times)
        raise
    return collect(saved, times)


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
    geometry: Geometry = "planar"

    def inventory(self) -> np.ndarray:
        """Integral of each independent mole fraction over both phases (r^p dr), shape (n_t, m)."""
        left = np.einsum("ti,tik->tk", cell_measures(self.z_left.T, self.geometry).T, self.x_left)
        right = np.einsum("ti,tik->tk", cell_measures(self.z_right.T, self.geometry).T, self.x_right)
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
    geometry: Geometry = "planar",
) -> MovingBoundaryResult:
    """Integrate two-phase diffusion with a moving interface, as two layers of :func:`solve_layers`; in a
    cylinder or sphere (``geometry``) the left phase is the core and the right one the surrounding shell.

    ``x0_left`` (N_L, n-1) and ``x0_right`` (N_R, n-1) are the initial profiles on
    uniform grids over [0, interface] and [interface, length]; a binary may pass
    1D arrays. ``d_left``/``d_right`` are constant matrices or callables like those
    of :func:`solve_planar`, for example ``lambda x: alpha.interdiffusion_matrix(x, T)``.
    ``left``/``right`` are the conditions at z = 0 and z = length.
    """
    if not 0.0 < interface < length:
        raise ValueError("the interface must lie inside (0, length)")
    res = solve_layers([0.0, interface, length], [Layer(x0_left, d_left), Layer(x0_right, d_right)], [condition],
                       t_end, dt, left, right, save_times, tol, picard_tol, max_newton, max_picard, geometry)
    return MovingBoundaryResult(res.times, res.interfaces[:, 0], res.faces[0], res.x[0], res.faces[1], res.x[1],
                                res.interface_left[:, 0], res.interface_right[:, 0], geometry)
