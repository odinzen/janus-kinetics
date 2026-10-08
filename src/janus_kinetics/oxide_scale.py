"""Parabolic growth of a compact oxide scale on its metal, with the transport model of
Hallstrom, Halvarsson, Hoglund, Jonsson and Agren, Solid State Ionics 240 (2013) 41.

The paper integrates the moving-boundary problem numerically (their eqs. 13-15).
For a scale that stays close to stoichiometry the amount stored when its
composition shifts is negligible next to what crosses it, so each flux is the same
at every depth. Writing the cation flux (their eq. 3) in terms of mu_O through
Gibbs-Duhem, n_M dmu_M + dmu_O = 0, and integrating across a scale of thickness X
from the metal/oxide equilibrium to the gas,

    J_M X = (1/V_m) int sum_s (a_s/a_O) y_Va^s y_M^s M_s / n_M dmu_O     (outward)
    J_O X = (1/V_m) int M_O dmu_O                                          (inward, eq. 5)

with n_M the local metal per anion site. Each mole of metal reaching the gas side
makes 1/n_g moles (per anion) of oxide of the gas-side composition n_g, and each
mole of oxygen reaching the metal side makes one, so the interface balances (eq.
15) give dX/dt = V_m (J_M / n_g + J_O) and X^2 = 2 k t with

    k = (1/n_g) int sum_s (a_s/a_O) y_Va^s y_M^s M_s / n_M dmu_O + int M_O dmu_O
      = (1/2) int (D*_M / n_g + D*_O) d ln p_O2

in which V_m, taken as constant per mole of anion, cancels. With D* from their eqs.
(4) and (7) the second form is Wagner's rate constant, 1/n_g = 3/2 being the charge
ratio of Cr2O3. The neglected storage is of the order of the change in n_M across
the scale (under 2 % for Cr2O3 on Cr). The derivation is this package's; the paper
states only the equations it starts from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycalphad import Database, equilibrium, variables as v

from .oxides import OxideCationDiffusion


@dataclass(frozen=True)
class ScaleGrowth:
    """Parabolic constant ``k`` (m^2/s, X^2 = 2 k t) split into its ``cation`` (scale grown at the
    gas side) and ``anion`` (grown at the metal side) parts, with the oxygen potentials (J/mol)
    at the metal/oxide and oxide/gas interfaces and the range of metal per anion site across the scale."""

    k: float
    cation: float
    anion: float
    mu_inner: float
    mu_outer: float
    n_metal_range: tuple[float, float]

    def thickness(self, t):
        """Scale thickness (m) after time ``t`` (s)."""
        return np.sqrt(2 * self.k * np.asarray(t, dtype=float))

    @property
    def inward_fraction(self) -> float:
        """Share of the scale formed at the metal/oxide interface by oxygen transport."""
        return self.anion / self.k


def metal_oxide_potential(db: Database, T: float, metal: str, metal_phase: str, oxide_phase: str,
                          x_anion: float, anion: str = "O") -> float:
    """Anion chemical potential (J/mol) at which ``metal_phase`` and ``oxide_phase`` coexist,
    from an equilibrium of the two at overall anion fraction ``x_anion`` lying between them."""
    eq = equilibrium(db, [metal.upper(), anion.upper(), "VA"], [metal_phase, oxide_phase],
                     {v.T: T, v.P: 1e5, v.N: 1, v.X(anion.upper()): x_anion})
    names = [str(p) for p in np.ravel(eq.Phase.values)]
    present = {p for p in names if p}
    if present != {metal_phase.upper(), oxide_phase.upper()}:
        raise ValueError(f"at x_{anion} = {x_anion} and {T:g} K the stable phases are {sorted(present)}, "
                         f"not {metal_phase} + {oxide_phase}")
    # An interstitial metal model can formally fill with oxygen beyond the oxide (bcc
    # (Cr)1(O,Va)3 up to x_O = 0.75), and pycalphad then pairs that with the oxide.
    k = list(eq.component.values).index(anion.upper())
    x = eq.X.values.reshape(len(names), -1)
    x_metal, x_oxide = (x[names.index(p.upper()), k] for p in (metal_phase, oxide_phase))
    if not x_metal < x_oxide:
        raise ValueError(f"at x_{anion} = {x_anion} the {metal_phase} found holds x_{anion} = {x_metal:.3g}, "
                         f"not the metal side of {oxide_phase} (x_{anion} = {x_oxide:.3g})")
    return float(dict(zip(eq.component.values, np.ravel(eq.MU.values)))[anion.upper()])


def parabolic_scale_growth(oxide: OxideCationDiffusion, T: float, p_outer: float, metal_phase: str | None = None,
                           p_inner: float | None = None, x_anion: float | None = None,
                           nodes: int = 16) -> ScaleGrowth:
    """Parabolic rate constant of a scale of ``oxide`` between O2 at ``p_outer`` (bar) and either
    the metal phase ``metal_phase`` (local equilibrium at the metal/oxide interface) or a
    fixed inner pressure ``p_inner``.

    The integral over mu_O uses Gauss-Legendre with ``nodes`` points, each an
    equilibrium of the oxide. ``x_anion`` places the metal/oxide equilibrium
    calculation between the two phases; by default it is half the oxide's anion
    fraction. A ``p_inner`` is taken as given: it must lie inside the oxide's
    stability range, since below the metal/oxide equilibrium the integral runs over
    oxide that would not exist and k comes out too large without any error. The
    model assumes a compact, adherent scale with no voids or porosity, a planar
    front, and transport only through the oxide.
    """
    if (metal_phase is None) == (p_inner is None):
        raise ValueError("give exactly one of metal_phase and p_inner")
    mu_outer = oxide.oxygen_potential(T, p_outer)
    gas_side = oxide.state_at_potential(T, mu_outer)
    if metal_phase is not None:
        if x_anion is None:
            x_anion = 0.5 * gas_side["x"][oxide.anion]
        mu_inner = metal_oxide_potential(oxide.db, T, oxide.metal, metal_phase, oxide.phase, x_anion, oxide.anion)
    else:
        mu_inner = oxide.oxygen_potential(T, p_inner)
    if not mu_inner < mu_outer:
        raise ValueError("the gas must be more oxidizing than the metal side of the scale")
    t, w = np.polynomial.legendre.leggauss(nodes)
    half = 0.5 * (mu_outer - mu_inner)
    cation = anion = 0.0
    n_values = [gas_side["n_metal"]]
    for ti, wi in zip(t, w):
        s = oxide.state_at_potential(T, mu_inner + half * (ti + 1))
        n_values.append(s["n_metal"])
        cation += wi * half * oxide._site_sum(T, s["y"]) / (s["n_metal"] * gas_side["n_metal"])
        anion += wi * half * oxide._anion_mobility(T, s["y"])
    return ScaleGrowth(cation + anion, cation, anion, mu_inner, mu_outer, (min(n_values), max(n_values)))
