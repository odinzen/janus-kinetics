"""Two compound layers competing in a thin-film couple, after Gosele and Tu, J. Appl. Phys. 53 (1982) 3252.

Compounds A_beta B and A_gamma B (beta, gamma the A atoms per B atom) grow between saturated phases
A_alpha B and A_delta B, alpha > beta > gamma > delta. The A flux through each layer is limited both by
diffusion (interdiffusivity D) and by the reactions at its two interfaces (barriers kappa, combined as
1/kappa_eff = 1/kappa_1 + 1/kappa_2, eq. 7):

    j = Delta C_eq kappa_eff / (1 + x kappa_eff / D)                     (eqs. 6, 17-18)

and each layer gains from its own flux and loses to its neighbour's (eqs. 15-16),

    dx_beta/dt  = G_beta j_beta - G_beta,gamma j_gamma,
    dx_gamma/dt = G_gamma j_gamma - G_gamma,beta j_beta,

with the composition factors of eqs. (19)-(21) for a common atomic volume Omega_0. Below its changeover
thickness x* = D / kappa_eff (eq. 11) a layer grows linearly, above it parabolically. Both layers grow
only while r = j_beta / j_gamma lies between r_1 = G_beta,gamma / G_beta and r_2 = G_gamma / G_gamma,beta
(eqs. 24-29); otherwise one shrinks, and a layer of zero thickness that would shrink stays absent. So a
second compound may not form until the first has reached a critical thickness (eq. 34), which is how
interfacial barriers suppress phases in thin films while thick couples grow them all. Nucleation is
outside the model, as in the paper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class CompoundLayer:
    """A compound layer: ``ratio`` A atoms per B atom, equilibrium A concentration drop ``delta_c`` across it
    (atoms per m^3), interdiffusivity ``diffusivity`` (m^2/s) and the reaction barriers of its two interfaces
    (m/s), combined by eq. (7)."""

    ratio: float
    delta_c: float
    diffusivity: float
    barriers: tuple[float, float]

    @property
    def barrier(self) -> float:
        """kappa_eff, eq. (7)."""
        k1, k2 = self.barriers
        return 1.0 / (1.0 / k1 + 1.0 / k2)

    @property
    def changeover(self) -> float:
        """x* = D / kappa_eff, eq. (11)."""
        return self.diffusivity / self.barrier

    def flux(self, thickness):
        """A flux across the layer (atoms m^-2 s^-1), eqs. (6), (17)-(18)."""
        k = self.barrier
        return self.delta_c * k / (1.0 + np.asarray(thickness, dtype=float) * k / self.diffusivity)


def composition_factors(alpha: float, beta: float, gamma: float, delta: float, atomic_volume: float):
    """(G_beta, G_gamma, G_beta,gamma, G_gamma,beta), eqs. (10) and (19)-(21); alpha may be infinite and delta
    zero, the pure components of section IV."""
    if not alpha > beta > gamma > delta >= 0:
        raise ValueError("need alpha > beta > gamma > delta >= 0 (A atoms per B atom)")
    inv = (lambda d: 0.0 if np.isinf(d) else 1.0 / d)
    g_beta = atomic_volume * (1 + beta) ** 2 * (inv(alpha - beta) + 1 / (beta - gamma))
    g_gamma = atomic_volume * (1 + gamma) ** 2 * (1 / (beta - gamma) + 1 / (gamma - delta))
    g_cross = atomic_volume * (1 + beta) * (1 + gamma) / (beta - gamma)
    return g_beta, g_gamma, g_cross, g_cross


def growth_window(alpha: float, beta: float, gamma: float, delta: float) -> tuple[float, float]:
    """(r_1, r_2): both layers grow when r_1 < j_beta / j_gamma < r_2, eqs. (27)-(28)."""
    if not alpha > beta > gamma > delta >= 0:
        raise ValueError("need alpha > beta > gamma > delta >= 0 (A atoms per B atom)")
    r1 = (1 + gamma) / (1 + beta) * (1.0 if np.isinf(alpha) else (alpha - beta) / (alpha - gamma))
    r2 = (1 + gamma) * (beta - delta) / ((1 + beta) * (gamma - delta))
    return r1, r2


def critical_thickness(beta_layer: CompoundLayer, gamma_layer: CompoundLayer, alpha: float, delta: float) -> float:
    """Thickness the gamma layer must reach before an interface-controlled beta layer can grow beside it, eq. (34):
    x_crit = r_1 Delta C_gamma D_gamma / (Delta C_beta kappa_beta). Eq. (34) takes the gamma layer as diffusion
    controlled; for a beta layer of zero thickness the exact onset is x_crit - x*_gamma."""
    r1, _ = growth_window(alpha, beta_layer.ratio, gamma_layer.ratio, delta)
    return r1 * gamma_layer.delta_c * gamma_layer.diffusivity / (beta_layer.delta_c * beta_layer.barrier)


@dataclass
class CompetitionResult:
    times: np.ndarray
    beta: np.ndarray  # thickness of the A_beta B layer (m)
    gamma: np.ndarray  # thickness of the A_gamma B layer (m)


def solve_two_layers(beta_layer: CompoundLayer, gamma_layer: CompoundLayer, alpha: float, delta: float,
                     atomic_volume: float, t_end: float, initial: tuple[float, float] = (0.0, 0.0),
                     save_times: Sequence[float] | None = None, rtol: float = 1e-10) -> CompetitionResult:
    """Integrate eqs. (15)-(18) from layer thicknesses ``initial`` (m). A layer of zero thickness whose rate would
    be negative stays at zero (the phase is kinetically suppressed); a shrinking layer stops at zero."""
    g_b, g_g, g_bg, g_gb = composition_factors(alpha, beta_layer.ratio, gamma_layer.ratio, delta, atomic_volume)
    if min(initial) < 0:
        raise ValueError("layer thicknesses cannot be negative")
    save = np.unique(np.append(np.asarray([] if save_times is None else save_times, dtype=float), t_end))

    def rhs(t, x):
        xb, xg = max(x[0], 0.0), max(x[1], 0.0)
        jb, jg = beta_layer.flux(xb), gamma_layer.flux(xg)
        rate = np.array([g_b * jb - g_bg * jg, g_g * jg - g_gb * jb])
        # an absent layer (or one that has just shrunk away) cannot shrink further
        rate[(np.array([xb, xg]) <= 0) & (rate < 0)] = 0.0
        return rate

    # thin films are followed to rtol of a nanometre; a changeover thickness is no scale, being metres for a
    # layer whose reactions are all but blocked
    sol = solve_ivp(rhs, (0.0, t_end), np.asarray(initial, dtype=float), method="LSODA", t_eval=save, rtol=rtol,
                    atol=rtol * 1e-9)
    if not sol.success:
        raise RuntimeError(f"layer competition integration failed: {sol.message}")
    return CompetitionResult(sol.t, np.maximum(sol.y[0], 0.0), np.maximum(sol.y[1], 0.0))
