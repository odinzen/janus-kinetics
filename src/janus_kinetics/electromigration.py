"""Stress build-up by electromigration in a confined metal line, after Korhonen, Borgesen, Tu and Li,
J. Appl. Phys. 73 (1993) 3790, with the threshold of Blech, J. Appl. Phys. 47 (1976) 1203.

The electron wind drives atoms along the line; where the flux diverges (a blocking grain, a via, the
line end) atoms are deposited or removed, and in a line held by a rigid passivation that changes
the stress. With vacancies in equilibrium, Korhonen et al.'s eq. (8) is a diffusion equation for
the stress sigma along the line,

    d sigma / dt = d/dx [ kappa (d sigma / dx + G) ],   kappa = D_a B Omega / kT,   G = E q* / Omega,

with D_a the effective diffusivity of the line (eq. 1, delta D_GB / d for grain-boundary transport),
B the effective modulus, Omega the atomic volume and E q* = rho j |Z*| e the electron-wind force on an
atom. The x axis points along the electron flow, so atoms leave x = 0. A blocked end passes no flux,
d sigma/dx + G = 0, and the steady state of a line blocked at both ends is Blech's linear profile
(eq. 12), sigma = G L (1/2 - x/L), whose end-to-end difference G L is the Delta sigma of Blech's eq. (6):
no electromigration damage occurs below the current density-length product (jL)_c = Omega Delta sigma_max
/ (|Z*| e rho). Blech's paper has an erratum (J. Appl. Phys. 48 (1977) 2648) that has not been read here;
only his equations and his own check of the threshold are used.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import erfc

from .errors import ConvergenceError
from .solver import _march

ELEMENTARY_CHARGE = 1.602176634e-19  # C
K_B = 1.380649e-23  # J/K


def wind_stress_gradient(resistivity: float, current_density: float, effective_charge: float,
                         atomic_volume: float) -> float:
    """G = E q* / Omega = rho j |Z*| e / Omega in Pa/m (Korhonen et al. p. 3792), from rho in ohm m,
    j in A/m^2 and Omega in m^3."""
    return resistivity * current_density * abs(effective_charge) * ELEMENTARY_CHARGE / atomic_volume


def stress_diffusivity(diffusivity: float, modulus: float, atomic_volume: float, temperature: float) -> float:
    """kappa = D_a B Omega / kT in m^2/s (Korhonen et al. eq. 8)."""
    return diffusivity * modulus * atomic_volume / (K_B * temperature)


def blech_product(stress_difference: float, effective_charge: float, resistivity: float,
                  atomic_volume: float) -> float:
    """Threshold current density times length (A/m), Blech's eq. (6): (jL)_c = Omega Delta sigma /
    (|Z*| e rho), for the largest stress difference Delta sigma (Pa) the line sustains between its ends."""
    return atomic_volume * stress_difference / (abs(effective_charge) * ELEMENTARY_CHARGE * resistivity)


def drift_velocity(current_density: float, length: float, diffusivity: float, temperature: float,
                   effective_charge: float, resistivity: float, atomic_volume: float,
                   stress_difference: float) -> float:
    """Net drift velocity (m/s) of a line of the given length, Blech's eqs. (1) and (3)-(4): the wind
    drives atoms at v = (D/kT) e |Z*| rho j and the stress difference between the ends drives them back
    at (D/kT) Omega Delta sigma / l; below the threshold of eq. (5) nothing moves."""
    mobility = diffusivity / (K_B * temperature)
    forward = mobility * ELEMENTARY_CHARGE * abs(effective_charge) * resistivity * current_density
    back = mobility * atomic_volume * stress_difference / length
    return max(forward - back, 0.0)


def steady_stress(x, length: float, gradient: float):
    """Blech's steady state of a line blocked at both ends, Korhonen et al. eq. (12)."""
    return gradient * length * (0.5 - np.asarray(x, dtype=float) / length)


def semi_infinite_stress(x, t, kappa: float, gradient: float):
    """Stress in a semi-infinite line blocked at x = 0 from t = 0, Korhonen et al. eq. (9); at the
    block it is 2 G sqrt(kappa t / pi), eq. (10). ``x`` and ``t`` broadcast; the line starts stress-free."""
    x, t = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
    s = np.sqrt(4 * kappa * np.maximum(t, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = gradient * (s / np.sqrt(np.pi) * np.exp(-(x / s) ** 2) - x * erfc(x / s))
    return np.where(t > 0, sigma, 0.0)


def finite_line_stress(x, t: float, length: float, kappa: float, gradient: float, terms: int = 400):
    """Stress in a line blocked at both ends, Korhonen et al. eq. (13), summed over ``terms`` modes
    m_n = (2n + 1) pi."""
    xi = np.asarray(x, dtype=float) / length
    tau = kappa * t / length**2
    m = (2 * np.arange(terms) + 1) * np.pi
    series = np.sum(np.exp(-np.outer(m * m, np.ones_like(xi)) * tau) * np.cos(np.outer(m, xi)) / (m * m)[:, None],
                    axis=0)
    return gradient * length * (0.5 - xi - 4 * series)


def void_line_stress(x, t: float, length: float, kappa: float, gradient: float, width: float, terms: int = 400):
    """Stress in a line with a void at x = 0 and a block at x = L, Korhonen et al. eqs. (15)-(16), with
    alpha = L / w and beta_n the roots of beta tan(beta) = alpha. At the void the flux follows their
    eq. (14), so d sigma/dx + G = sigma / w there. The leading term printed as -G xi is taken as -G L xi:
    it must carry a length, and only -G x vanishes with the series at t = 0 and remains as the steady
    state they describe (compressive except at the void, where the stress is zero)."""
    xi = np.asarray(x, dtype=float) / length
    tau = kappa * t / length**2
    alpha = length / width
    beta = np.array([brentq(lambda b: b * np.tan(b) - alpha, n * np.pi + 1e-12, n * np.pi + np.pi / 2 - 1e-12)
                     for n in range(terms)])
    weight = ((np.cos(beta) + beta * np.sin(beta) - 1 + alpha * (np.sin(beta) - beta * np.cos(beta)) / beta)
              / ((alpha + alpha**2 + beta**2) * beta))
    modes = (beta[:, None] * np.cos(np.outer(beta, xi)) + alpha * np.sin(np.outer(beta, xi))) * np.exp(-(beta**2)[:, None] * tau)
    return -gradient * length * xi + 2 * gradient * length * (weight[:, None] * modes).sum(axis=0)


@dataclass(frozen=True)
class LineEnd:
    """Condition at an end of the line: ``blocked`` (no atomic flux), a ``void`` from which atoms flow
    as Korhonen et al.'s eq. (14) gives (``width`` w, the line width), or a ``reservoir`` held at
    ``stress`` (a bond pad or a wide line, say)."""

    kind: str = "blocked"
    width: float | None = None
    stress: float = 0.0

    def __post_init__(self):
        if self.kind not in ("blocked", "void", "reservoir"):
            raise ValueError(f"a line end is 'blocked', 'void' or 'reservoir', not {self.kind!r}")
        if self.kind == "void" and (self.width is None or self.width <= 0):
            raise ValueError("a void end needs the line width (m)")


@dataclass
class StressResult:
    times: np.ndarray
    x: np.ndarray  # cell centres (m)
    stress: np.ndarray  # [t, cell] in Pa
    steps: int


def solve_line_stress(
    length: float,
    n_cells: int,
    gradient: float,
    kappa: float | Callable[[np.ndarray], np.ndarray],
    t_end: float,
    dt: float,
    left: LineEnd = LineEnd(),
    right: LineEnd = LineEnd(),
    save_times: Sequence[float] | None = None,
    initial: np.ndarray | float = 0.0,
    yield_stress: float | None = None,
    tolerance: float | None = None,
    picard_tol: float = 1e-10,
    max_picard: int = 50,
) -> StressResult:
    """Integrate Korhonen et al.'s eq. (8) along a line of ``length`` (m) by backward Euler on ``n_cells``
    equal cells. ``kappa`` is constant or a function of the stress, such as their eq. (18) with the
    activation volumes neglected, kappa(sigma) = kappa_0 exp(Omega sigma / kT); then each step iterates
    until the stress changes by less than ``picard_tol`` times its largest value. ``tolerance`` (Pa) adapts the steps as in :func:`solve_planar`.

    ``yield_stress`` caps |sigma| by ideal plasticity (their section III, eq. 11): a yielded zone holds
    the yield stress and exchanges no flux with the rest of the line, and grows as the stress beside it
    reaches yield. Loading is taken as monotonic: a yielded cell does not unload (after a current
    reversal, say)."""
    if length <= 0 or n_cells < 2:
        raise ValueError("need a positive length and at least two cells")
    if yield_stress is not None and yield_stress <= 0:
        raise ValueError("the yield stress must be positive")
    h = length / n_cells
    x = (np.arange(n_cells) + 0.5) * h
    sigma = np.broadcast_to(np.asarray(initial, dtype=float), (n_cells,)).copy()
    kfun = kappa if callable(kappa) else (lambda s, k=float(kappa): np.full(np.shape(s), k))

    def end_terms(end: LineEnd, s_cell: float, k: float, outward: float):
        """(a, b) with the stress flux F through the end face = a sigma_cell + b, F = kappa (dsigma/dx + G)
        taken in the +x direction. A void end follows dsigma/dx + G = sigma_face / w (their eq. 14), with
        the face value eliminated over the half cell."""
        if end.kind == "blocked":
            return 0.0, 0.0
        if end.kind == "reservoir":
            # dsigma/dx = outward (s_face - s_cell)/(h/2)
            return -outward * 2 * k / h, k * (outward * 2 * end.stress / h + gradient)
        w = end.width
        # Atoms enter from the void towards the line's interior when it is in tension, so F = -outward
        # kappa s_f / w; with the half-cell gradient, (s_c - s_f)/(h/2) + G = s_f / w at a left end
        # and (s_f - s_c)/(h/2) + G = -s_f / w at a right end.
        if outward < 0:
            s_f = lambda s: (2 * s / h + gradient) / (2 / h + 1 / w)
        else:
            s_f = lambda s: (2 * s / h - gradient) / (2 / h + 1 / w)
        f1, f0 = -outward * k * s_f(1.0) / w, -outward * k * s_f(0.0) / w
        return f1 - f0, f0

    def linear_step(s_old, s, step, held):
        """Backward-Euler step with kappa at the stress s; cells in ``held`` (+1 tension, -1 compression)
        stay at the yield stress and no flux enters, leaves or crosses a yielded zone."""
        k_cell = kfun(s)
        k_face = 0.5 * (k_cell[1:] + k_cell[:-1])
        k_face = np.where((held[1:] == 0) & (held[:-1] == 0), k_face, 0.0)
        ab = np.zeros((3, n_cells))
        rhs = h / step * s_old
        ab[1] = h / step
        c = k_face / h
        ab[1, :-1] += c
        ab[1, 1:] += c
        ab[0, 1:] = -c
        ab[2, :-1] = -c
        # h ds/dt = F_right - F_left, inner F = kappa ((s_{i+1} - s_i)/h + G)
        rhs[:-1] += k_face * gradient
        rhs[1:] -= k_face * gradient
        a_l, b_l = end_terms(left, s[0], k_cell[0], -1.0) if held[0] == 0 else (0.0, 0.0)
        a_r, b_r = end_terms(right, s[-1], k_cell[-1], +1.0) if held[-1] == 0 else (0.0, 0.0)
        # cell 0 loses the flux through its left face, cell n-1 gains that through its right one
        ab[1, 0] += a_l
        rhs[0] -= b_l
        ab[1, -1] -= a_r
        rhs[-1] += b_r
        yielded = held != 0
        ab[1, yielded] = 1.0
        if yield_stress is not None:
            rhs[yielded] = held[yielded] * yield_stress
        return solve_banded((1, 1), ab, rhs), k_cell

    def advance(s_old, step, t_new):
        # Ideal plasticity (section III, eq. 11): a yielded zone holds the yield stress and the plastic
        # flow cancels the transport through it, so it exchanges no flux with the line; its edge then
        # moves as the elastic stress beside it reaches yield. Cells join the zone when a step would
        # carry them past yield and stay in it: unloading after yield is not modelled.
        held = np.zeros(n_cells, dtype=int)
        if yield_stress is not None:
            held[s_old >= yield_stress * (1 - 1e-12)] = 1
            held[s_old <= -yield_stress * (1 - 1e-12)] = -1
        iterations = 0
        for _ in range(n_cells + 1):  # the yielded set only grows
            s = s_old.copy()
            for it in range(1, max_picard + 1):
                new, k_cell = linear_step(s_old, s, step, held)
                change = np.max(np.abs(new - s)) / max(np.max(np.abs(new)), 1.0)
                s = new
                if not callable(kappa) or change < picard_tol:
                    break
            else:
                raise ConvergenceError(f"stress iteration did not converge near t = {t_new:g} s; reduce dt")
            iterations += it
            if yield_stress is None:
                return s, iterations
            update = held.copy()
            update[(held == 0) & (s > yield_stress)] = 1
            update[(held == 0) & (s < -yield_stress)] = -1
            if np.array_equal(update, held):
                return s, iterations
            held = update
        raise ConvergenceError(f"the yielded zone did not settle near t = {t_new:g} s; reduce dt")

    times, saved, iters = _march(sigma, advance, t_end, dt, save_times, tolerance, scale=1.0)
    return StressResult(np.array(times), x, np.array(saved), len(iters))
