"""Linear-parabolic growth of a thermal oxide, after Deal and Grove, J. Appl. Phys. 36 (1965) 3770.

The oxidant crosses the gas boundary layer (flux h (C* - C_0), eq. 1), diffuses through the oxide
(D_eff (C_0 - C_i) / x_0, eq. 4) and reacts at the inner interface (k C_i, eq. 5). In steady state the
three fluxes are equal, and with N_1 oxidant molecules per volume of oxide the thickness grows as
eq. (9), dx_0/dt = B / (A + 2 x_0), with

    A = 2 D_eff (1/k + 1/h),   B = 2 D_eff C* / N_1          (eqs. 12a, 12b)

so that x_0^2 + A x_0 = B (t + tau), tau = (x_i^2 + A x_i) / B for an oxide x_i present at t = 0 (eqs.
11-12c). B is the parabolic and B/A = (kh / (k + h)) C* / N_1 the linear rate constant (p. 3773). The
model holds after an initial transient of order x_0^2 / 2 D_eff (appendix); for dry oxygen Deal and Grove
fold the faster first ~230 A into x_i.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .schedule import Temperature, temperature_at

Value = float | Callable[[float], float]


def deal_grove_constants(diffusivity: float, surface_rate: float, gas_transport: float, solubility: float,
                         sites: float) -> tuple[float, float]:
    """(A in m, B in m^2/s) from D_eff (m^2/s), the interface reaction constant k and gas-phase
    transport coefficient h (m/s), the oxidant solubility C* and the oxidant molecules per volume of
    oxide N_1 (any one unit for both), eqs. (12a)-(12b)."""
    if min(diffusivity, surface_rate, gas_transport, solubility, sites) <= 0:
        raise ValueError("D_eff, k, h, C* and N_1 must be positive")
    return 2 * diffusivity * (1 / surface_rate + 1 / gas_transport), 2 * diffusivity * solubility / sites


def time_shift(a: float, b: float, initial: float = 0.0) -> float:
    """tau = (x_i^2 + A x_i) / B, eq. (12c)."""
    return (initial**2 + a * initial) / b


def oxide_thickness(t, a: float, b: float, initial: float = 0.0):
    """Oxide thickness at time t, eq. (13), from x_0^2 + A x_0 = B (t + tau). Written as
    2 B s / (A + sqrt(A^2 + 4 B s)), s = t + tau, which keeps its precision when A^2 >> B s."""
    if a < 0 or b <= 0:
        raise ValueError("need A >= 0 and B > 0")
    s = np.asarray(t, dtype=float) + time_shift(a, b, initial)
    return 2 * b * s / (a + np.sqrt(a * a + 4 * b * s))


def oxidation_time(thickness, a: float, b: float, initial: float = 0.0):
    """Time to grow from ``initial`` to ``thickness``, eq. (12) solved for t."""
    x = np.asarray(thickness, dtype=float)
    return (x * x + a * x) / b - time_shift(a, b, initial)


def fit_linear_parabolic(times: Sequence[float], thickness: Sequence[float], shift: float = 0.0) -> tuple[float, float]:
    """(A, B) from measured thickness against time as Deal and Grove extract them (their Figs. 4 and 7): eq.
    (12) rearranged as x_0 = B (t + tau) / x_0 - A is a straight line in (t + tau)/x_0, of slope B and
    intercept -A, fitted by least squares."""
    t = np.asarray(times, dtype=float)
    x = np.asarray(thickness, dtype=float)
    if t.shape != x.shape or t.size < 2 or np.any(x <= 0):
        raise ValueError("need at least two positive thicknesses, one per time")
    slope, intercept = np.polyfit((t + shift) / x, x, 1)
    return -intercept, slope


def grow_oxide(t_end: float, a: Value, b: Value, temperature: Temperature = 0.0, initial: float = 0.0,
               save_times: Sequence[float] | None = None, rtol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """(times, thickness) of an oxide growing by eq. (9), dx/dt = B / (A + 2 x), with A and B constants or
    functions of T along ``temperature`` (a constant or a schedule), from ``initial`` at t = 0. Eq. (9) is
    the instantaneous law, so it carries over to an anneal; at constant T it reproduces eq. (13)."""
    at = (lambda t: a(temperature_at(temperature, t))) if callable(a) else (lambda t: a)
    bt = (lambda t: b(temperature_at(temperature, t))) if callable(b) else (lambda t: b)
    save = np.unique(np.append(np.asarray([] if save_times is None else save_times, dtype=float), t_end))
    if save[0] < 0 or save[-1] > t_end:
        raise ValueError("save times must lie in [0, t_end]")
    sol = solve_ivp(lambda t, x: bt(t) / (at(t) + 2 * x), (0.0, t_end), [initial], method="LSODA", t_eval=save,
                    rtol=rtol, atol=rtol * max(initial, np.sqrt(bt(0.0) * max(t_end, 1e-300)), 1e-300))
    if not sol.success:
        raise RuntimeError(f"oxide growth integration failed: {sol.message}")
    return sol.t, sol.y[0]
