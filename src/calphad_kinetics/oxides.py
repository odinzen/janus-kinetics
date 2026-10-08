"""Cation diffusion in non-stoichiometric oxides, after Hallstrom, Hoglund and Agren, Acta Mater. 59 (2011) 53.

The metal M jumps into vacant sites of the sublattices it shares with vacancies,
so its flux in the lattice-fixed frame depends on how many of those sites are
empty, which the oxygen potential sets through the defect structure of the
Calphad description. For each jump path s (a sublattice with a_s sites per
formula unit holding M and vacancies), their eqs. (3) and (13) read

    J_M = -sum_s (a_s / a_O) y_Va^s y_M^s M_s (1 / V_m) dmu_M/dz

with a_O the anion sites and V_m the volume per mole of anion sites; magnetite
has two paths with a_s / a_O = 1/2 (eq. 3), hematite one with 1/3 (eq. 13). A
tracer spreads over the metal sites in proportion, which gives eq. (7)

    D*_M = RT sum_s (a_s / a_O) y_Va^s y_M^s M_s / n_M

with n_M the metal atoms per anion site, and the chemical diffusion coefficient
of eq. (9), D_M = (D*_M / RT) n_M dmu_M/dn_M. The site fractions and chemical
potentials come from ``pycalphad.equilibrium`` of the oxide alone at the given
oxygen potential, so the results are only as good as the thermodynamic
description; the paper's mobilities were fitted with Sundman's (1991) Fe-O
description for magnetite and Kjellqvist et al.'s (2008) for hematite. Kjellqvist et
al.'s spinel, which pycalphad's alfeo.tdb carries, reproduces the paper's magnetite
defect fractions (its Fig. 3); Sundman's printed Table 1 does not (one end member
differs in sign).

Hallstrom, Halvarsson, Hoglund, Jonsson and Agren, Solid State Ionics 240 (2013)
41, apply the same scheme to Cr2O3 and add oxygen, which moves by vacancies on
its own sublattice: J_O = -M_O (1 / V_m) dmu_O/dz (their eq. 5), with the
vacancy fraction taken proportional to the interstitial metal, M_O = k(T) y_M^s
(eq. 6), so D*_O = RT M_O (eq. 7). In a fine-grained scale both mobilities are
replaced by the grain-boundary average of their eq. (8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pycalphad import Database, Model, equilibrium, variables as v

from .thermo import R


@dataclass(frozen=True)
class Arrhenius:
    """M(T) = prefactor exp(-activation / RT), the form of Hallstrom et al.'s (2011) eq. (4)."""

    prefactor: float
    activation: float

    def __call__(self, T: float) -> float:
        return self.prefactor * np.exp(-self.activation / (R * T))


def arrhenius_mobility(prefactor: float, activation: float) -> Arrhenius:
    """M(T) = prefactor exp(-activation / RT), the form of Hallstrom et al.'s eq. (4)."""
    return Arrhenius(prefactor, activation)


@dataclass(frozen=True)
class GrainBoundaryMobility:
    """Effective mobility of a polycrystal, Hallstrom et al. (2013) eq. (8):
    M_eff = f M_gb + (1 - f) M_b with f = width / grain_size.

    The boundary mobility keeps the bulk prefactor and takes ``alpha`` times its
    activation energy (their section 4.5.2, with alpha = 1/2 as the rule of thumb
    and a 5 Angstrom boundary width). The average holds when the diffusion fields
    of neighbouring boundaries overlap, a planar front in a scale with grains much
    thinner than the scale; their results are very sensitive to alpha (Fig. 10).
    """

    bulk: Arrhenius
    grain_size: float
    alpha: float = 0.5
    width: float = 5e-10

    def __post_init__(self):
        if not 0 < self.width <= self.grain_size:
            raise ValueError("need 0 < boundary width <= grain size")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")

    def __call__(self, T: float) -> float:
        f = self.width / self.grain_size
        boundary = self.bulk.prefactor * np.exp(-self.alpha * self.bulk.activation / (R * T))
        return f * boundary + (1 - f) * self.bulk(T)


@dataclass(frozen=True)
class JumpPath:
    """Jumps of the metal into vacancies of sublattice ``sublattice`` (index in the phase), with
    mobility ``mobility(T)`` in m^2 mol / (J s)."""

    sublattice: int
    mobility: Callable[[float], float]


@dataclass(frozen=True)
class AnionPath:
    """Anion jumps into anion vacancies, their fraction taken proportional to the metal on
    sublattice ``sublattice``: M_O = mobility(T) y_M (Hallstrom et al. 2013, eq. 6)."""

    sublattice: int
    mobility: Callable[[float], float]


class OxideCationDiffusion:
    """Metal diffusion in an oxide phase by vacancy jumps on the given sublattices, and
    optionally anion diffusion through ``anion_path``.

    ``metal`` and ``anion`` are element names (e.g. "FE" and "O"); every constituent
    of the metal (its ions) counts as the metal on a sublattice. The oxygen
    potential is given as a partial pressure of O2 relative to the database's gas
    standard state (1 bar for the usual R*T*LN(1E-05*P) term), using its GAS phase.
    """

    def __init__(self, db: Database, phase: str, paths: Sequence[JumpPath], metal: str = "FE", anion: str = "O",
                 anion_path: AnionPath | None = None):
        self.db, self.phase = db, phase.upper()
        self.metal, self.anion = metal.upper(), anion.upper()
        self.paths = tuple(paths)
        self.anion_path = anion_path
        self.components = [self.metal, self.anion, "VA"]
        model = Model(db, self.components, self.phase)
        self._names = [(y.sublattice_index, str(y.species.name)) for y in model.site_fractions]
        self._sites = [float(a) for a in db.phases[self.phase].sublattices]
        species = {i: {name for idx, name in self._names if idx == i} for i in range(len(self._sites))}
        anion_subl = [i for i, held in species.items() if held and all(s.startswith(self.anion) for s in held)]
        if len(anion_subl) != 1:
            raise ValueError(f"{self.phase} needs exactly one sublattice holding only {self.anion}")
        self._anion_sites = self._sites[anion_subl[0]]
        for path in self.paths:
            held = species.get(path.sublattice, set())
            if "VA" not in held or not any(s.startswith(self.metal) for s in held):
                raise ValueError(f"sublattice {path.sublattice} of {self.phase} does not hold both {self.metal} and VA")
        if anion_path is not None and not any(self._is_metal(s) for s in species.get(anion_path.sublattice, set())):
            raise ValueError(f"sublattice {anion_path.sublattice} of {self.phase} holds no {self.metal}")

    def _is_metal(self, name: str) -> bool:
        return name == self.metal or name.startswith(self.metal + "+")

    def oxygen_potential(self, T: float, p_o2: float) -> float:
        """mu_O (J/mol) of O2 gas at T and partial pressure p_o2, from the database's GAS phase."""
        if p_o2 <= 0:
            raise ValueError("the O2 partial pressure must be positive")
        eq = equilibrium(self.db, [self.anion], ["GAS"], {v.T: T, v.P: 1e5, v.N: 1})
        return float(eq.MU.values.squeeze()) + 0.5 * R * T * np.log(p_o2)

    def state(self, T: float, p_o2: float) -> dict:
        """Site fractions keyed (sublattice, species), metal per anion site n_M, chemical potentials and mole fractions."""
        return self.state_at_potential(T, self.oxygen_potential(T, p_o2))

    def state_at_potential(self, T: float, mu_o: float) -> dict:
        """As :meth:`state`, at the oxygen chemical potential ``mu_o`` (J/mol) instead of a pressure."""
        eq = equilibrium(self.db, self.components, [self.phase], {v.T: T, v.P: 1e5, v.N: 1, v.MU(self.anion): mu_o})
        y = eq.Y.values.squeeze()
        y = y[0] if y.ndim > 1 else y
        if not np.all(np.isfinite(y[: len(self._names)])):
            raise ValueError(f"{self.phase} has no equilibrium at {T:g} K and mu_O = {mu_o:g} J/mol")
        fractions = dict(zip(self._names, y[: len(self._names)]))
        metal = sum(self._sites[i] * f for (i, name), f in fractions.items() if self._is_metal(name))
        mu = dict(zip(eq.component.values, eq.MU.values.squeeze()))
        x = eq.X.values.squeeze()
        x = dict(zip(eq.component.values, x[0] if x.ndim > 1 else x))
        return {"y": fractions, "n_metal": metal / self._anion_sites, "mu": {k: float(m) for k, m in mu.items()},
                "x": {k: float(f) for k, f in x.items()}}

    def _metal_on(self, sublattice: int, fractions: dict) -> float:
        return sum(f for (i, name), f in fractions.items() if i == sublattice and self._is_metal(name))

    def _site_sum(self, T: float, fractions: dict) -> float:
        total = 0.0
        for path in self.paths:
            total += self._sites[path.sublattice] / self._anion_sites * fractions[(path.sublattice, "VA")] \
                * self._metal_on(path.sublattice, fractions) * path.mobility(T)
        return total

    def _anion_mobility(self, T: float, fractions: dict) -> float:
        if self.anion_path is None:
            return 0.0
        return self.anion_path.mobility(T) * self._metal_on(self.anion_path.sublattice, fractions)

    def tracer_diffusivity(self, T: float, p_o2: float) -> float:
        """D*_M in m^2/s, eq. (7) (eq. (13) for a single path)."""
        s = self.state(T, p_o2)
        return R * T * self._site_sum(T, s["y"]) / s["n_metal"]

    def anion_tracer_diffusivity(self, T: float, p_o2: float) -> float:
        """D*_O = RT M_O in m^2/s, Hallstrom et al. (2013) eq. (7); zero without an anion path."""
        return R * T * self._anion_mobility(T, self.state(T, p_o2)["y"])

    def chemical_diffusivity(self, T: float, p_o2: float, step: float = 0.02) -> float:
        """D_M = (D*_M / RT) n_M dmu_M/dn_M (eq. 9) in m^2/s, the derivative taken along the
        oxygen potential by central differences in ln p_O2 of relative ``step``."""
        lo, mid, hi = (self.state(T, p_o2 * f) for f in (np.exp(-step), 1.0, np.exp(step)))
        dmu = hi["mu"][self.metal] - lo["mu"][self.metal]
        dn = hi["n_metal"] - lo["n_metal"]
        d_star = R * T * self._site_sum(T, mid["y"]) / mid["n_metal"]
        return d_star / (R * T) * mid["n_metal"] * dmu / dn


# Hallstrom, Hoglund and Agren (2011): eqs. (10)-(11) for magnetite, eq. (14) for hematite
HALLSTROM2011_MAGNETITE = (arrhenius_mobility(1.6e-10, 114520.0), arrhenius_mobility(6.4e-10, 121940.0))
HALLSTROM2011_HEMATITE = arrhenius_mobility(6.02e-6, 304290.0)
# Hallstrom et al. (2013): eq. (11) for Cr, eq. (12) for O without its y''_Cr factor
HALLSTROM2013_CHROMIUM = arrhenius_mobility(2.72e-7, 428200.0)
HALLSTROM2013_OXYGEN = arrhenius_mobility(2.47e-10, 351170.0)


def magnetite(db: Database, phase: str = "SPINEL") -> OxideCationDiffusion:
    """Fe in magnetite with Sundman's four-sublattice spinel (Fe)1(Fe,Va)2(Va,Fe)2(O)4: jumps on
    the octahedral (index 1) and extra octahedral (index 2) sublattices, eqs. (3) and (10)-(11)."""
    m2, m3 = HALLSTROM2011_MAGNETITE
    return OxideCationDiffusion(db, phase, [JumpPath(1, m2), JumpPath(2, m3)])


def hematite(db: Database, phase: str = "CORUNDUM") -> OxideCationDiffusion:
    """Fe in hematite with the corundum model (Fe+2,Fe+3)2(Fe+3,Va)1(O-2)3: jumps on sublattice 1,
    eqs. (13)-(14); the mobility describes diffusion along the c axis."""
    return OxideCationDiffusion(db, phase, [JumpPath(1, HALLSTROM2011_HEMATITE)])


def chromia(db: Database, phase: str = "CORUNDUM", grain_size: float | None = None, alpha: float = 0.5,
            width: float = 5e-10) -> OxideCationDiffusion:
    """Cr and O in Cr2O3 with the corundum model (Cr+2,Cr+3)2(Cr+3,Va)1(O-2)3, Hallstrom et al.
    (2013) eqs. (3)-(7) and (11)-(12). Cr jumps on the interstitial sublattice (index 1), O by
    vacancies in proportion to the interstitial Cr. With ``grain_size`` (m) both mobilities
    become the grain-boundary averages of eq. (8); without it they are the bulk values."""
    cr, o = HALLSTROM2013_CHROMIUM, HALLSTROM2013_OXYGEN
    if grain_size is not None:
        cr = GrainBoundaryMobility(cr, grain_size, alpha, width)
        o = GrainBoundaryMobility(o, grain_size, alpha, width)
    return OxideCationDiffusion(db, phase, [JumpPath(1, cr)], metal="CR", anion_path=AnionPath(1, o))
