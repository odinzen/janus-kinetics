"""Multicomponent precipitation with CALPHAD driving forces, after Svoboda, Fischer, Fratzl and
Kozeschnik, Mater. Sci. Eng. A 385 (2004) 166 (part I) and Kozeschnik, Svoboda, Fratzl and Fischer,
ibid. 157 (part II).

Precipitates are held in classes, each with a number density, a radius and the
composition it nucleated with. A class grows by eq. (45) of part I, the
Onsager-extremum result for a precipitate of fixed composition with infinite
interface mobility:

    drho/dt = (F - 2 gamma / rho) / (R T rho) [ sum_i (c_ki - c_0i)^2 / (c_0i D_0i) ]^-1

with F = -sum_i c_ki (mu_ki - mu_0i) the chemical driving force per volume (eq. 37,
no misfit energy), c = x / Omega and D_0i the tracer diffusivity RT M_i of i in the
matrix. With one molar volume Omega for both phases, as part I assumes, the sum
over c becomes (x_ki - x_0i)^2 / (Omega x_0i D_0i) and mu_ki enters only through the
precipitate's molar Gibbs energy G_k = sum_i x_ki mu_ki, so F = (sum_i x_ki mu_0i - G_k) / Omega.

Nuclei form at the composition of largest driving force (the "ortho-composition"
of part II, the parallel tangent to the matrix), at the rate of part I eqs. (40),
(41) and (46)-(48),

    J = N_C Z beta* exp(-Delta G* / kT) exp(-tau / t),  Delta G* = 16 pi gamma^3 / (3 F^2),
    beta* = 4 pi rho*^2 / (a^4 Omega) [ sum_i (c_i* - c_0i)^2 / (c_0i D_0i) ]^-1,  tau = 1 / (2 beta* Z^2),

with rho* = 2 gamma / F and the Zeldovich factor of Perez, Dumont and Acevedo-Reyes,
Acta Mater. 56 (2008) 2119, corrected eq. (4), Z = (Omega / N_A) / (2 pi rho*^2)
sqrt(gamma / kT), since part I takes Z from a source it does not reproduce. Each
step's nuclei enter as a new class slightly above rho* (part II, p. 158). The matrix
composition follows from mass balance (part I, eq. 4).

Scope: one precipitate phase, isothermal, spherical precipitates whose composition
stays as nucleated, no misfit energy, dilute enough that diffusion fields do not
overlap (the assumption behind part I's eq. 22).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from scipy.optimize import brentq

from .phase import SolutionPhase
from .thermo import R

K_B = 1.380649e-23  # J/K
N_A = 6.02214076e23  # 1/mol


class PrecipitateThermo:
    """Gibbs energy of the precipitate phase from pycalphad, with its internal degrees of freedom
    (ordering, sublattice occupancy) at equilibrium for each composition.

    ``components`` are the elements in the matrix's order, dependent element last.
    A ``composition`` fixes the precipitate's mole fractions (a line compound or a
    chosen stoichiometry); without it nuclei take the composition of largest driving force.
    """

    def __init__(self, db: Database, phase: str, components: Sequence[str], composition: Sequence[float] | None = None):
        self.db, self.phase = db, phase.upper()
        self.components = [c.upper() for c in components]
        self.fixed = None if composition is None else np.asarray(composition, dtype=float)
        if self.fixed is not None and (len(self.fixed) != len(self.components) or abs(self.fixed.sum() - 1) > 1e-9):
            raise ValueError("a fixed precipitate composition needs one mole fraction per component, summing to 1")
        self._gm_cache: dict = {}

    def _eq(self, T: float, conditions: dict):
        return equilibrium(self.db, self.components + ["VA"], [self.phase],
                           {v.T: T, v.P: 101325, v.N: 1, **conditions})

    @staticmethod
    def _values(eq, name: str) -> np.ndarray:
        return np.ravel(getattr(eq, name).values)

    def gm(self, x: Sequence[float], T: float) -> float:
        """Molar Gibbs energy (J per mole of atoms) at mole fractions ``x``."""
        x = np.asarray(x, dtype=float)
        key = (round(T, 6),) + tuple(np.round(x, 9))
        if key not in self._gm_cache:
            # elements absent from the precipitate are left out of the calculation, so a phase
            # without them (a pure element, a binary compound in a ternary) can be evaluated
            present = [c for c, xi in zip(self.components, x) if xi > 0]
            eq = equilibrium(self.db, present + ["VA"], [self.phase],
                             {v.T: T, v.P: 101325, v.N: 1,
                              **{v.X(c): xi for c, xi in zip(self.components, x) if c in present[:-1]}})
            g = float(self._values(eq, "GM")[0])
            if not np.isfinite(g):
                raise ValueError(f"{self.phase} has no state at x = {x} and {T:g} K")
            self._gm_cache[key] = g
        return self._gm_cache[key]

    def nucleus(self, mu_matrix: np.ndarray, T: float) -> tuple[np.ndarray, float]:
        """(composition, molar driving force in J/mol) of the precipitate with the largest driving force
        against a matrix with chemical potentials ``mu_matrix``.

        The precipitate is equilibrated at MU_i = mu_0i + d for every component but the
        last, and d is found so that its last potential is mu_0n + d as well. The
        tangent planes are then parallel and the driving force sum_i x_i mu_0i - G is -d.
        """
        mu_matrix = np.asarray(mu_matrix, dtype=float)
        if self.fixed is not None:
            return self.fixed.copy(), float(self.fixed @ mu_matrix - self.gm(self.fixed, T))

        def state(d):
            eq = self._eq(T, {v.MU(c): m + d for c, m in zip(self.components[:-1], mu_matrix[:-1])})
            names = [str(c) for c in eq.component.values]
            mu = dict(zip(names, self._values(eq, "MU")))
            x = dict(zip(names, self._values(eq, "X")))
            return mu[self.components[-1]] - mu_matrix[-1] - d, np.array([x[c] for c in self.components])

        lo, hi = -1000.0, 1000.0
        f_lo, f_hi = state(lo)[0], state(hi)[0]
        for _ in range(40):
            if f_lo > 0 > f_hi:
                break
            lo, hi = (2 * lo, hi) if f_lo <= 0 else (lo, 2 * hi)
            f_lo, f_hi = state(lo)[0], state(hi)[0]
        else:
            raise ValueError(f"no parallel tangent found for {self.phase} at {T:g} K")
        d = brentq(lambda d: state(d)[0], lo, hi, xtol=1e-6)
        return state(d)[1], -d


@dataclass(frozen=True)
class NucleationSettings:
    """Interfacial energy ``gamma`` (J/m^2), molar volume ``molar_volume`` (m^3 per mole of atoms,
    both phases), lattice parameter (m) of the attachment rate eq. (46), and nucleation sites per
    m^3 (default N_A / molar_volume, every lattice site, homogeneous nucleation)."""

    gamma: float
    molar_volume: float
    lattice_parameter: float
    sites: float | None = None

    def __post_init__(self):
        if self.gamma <= 0 or self.molar_volume <= 0:
            raise ValueError("interfacial energy and molar volume must be positive")
        if not 1e-11 < self.lattice_parameter < 1e-8:
            raise ValueError(f"lattice parameter {self.lattice_parameter:g} m is not a lattice parameter "
                             "(about 4e-10 m; give it in metres)")

    @property
    def site_density(self) -> float:
        return N_A / self.molar_volume if self.sites is None else self.sites


def growth_rates(radius, composition, driving_force, x_matrix, tracer, settings: NucleationSettings, T: float):
    """drho/dt (m/s) of classes with radii ``radius`` (k,), mole fractions ``composition`` (k, n) and
    driving forces ``driving_force`` (k,) in J/m^3, part I eq. (45)."""
    radius = np.asarray(radius, dtype=float)
    resistance = (((np.asarray(composition) - x_matrix) ** 2) / (x_matrix * tracer)).sum(axis=-1) / settings.molar_volume
    return (np.asarray(driving_force) - 2 * settings.gamma / radius) / (R * T * radius * resistance)


def nucleation(composition, driving_force, x_matrix, tracer, settings: NucleationSettings, T: float, t: float):
    """(rate in m^-3 s^-1, critical radius in m) for nuclei of mole fractions ``composition`` with
    driving force ``driving_force`` (J/m^3), part I eqs. (40)-(41) and (46)-(48)."""
    if driving_force <= 0 or t <= 0:
        return 0.0, np.inf
    kt = K_B * T
    g, omega = settings.gamma, settings.molar_volume
    rho = 2 * g / driving_force
    barrier = 16 * np.pi * g**3 / (3 * driving_force**2)
    resistance = (((np.asarray(composition) - x_matrix) ** 2) / (x_matrix * tracer)).sum() / omega
    beta = 4 * np.pi * rho**2 / (settings.lattice_parameter**4 * omega) / resistance
    zeldovich = omega / N_A / (2 * np.pi * rho**2) * np.sqrt(g / kt)
    tau = 1 / (2 * beta * zeldovich**2)
    return settings.site_density * zeldovich * beta * np.exp(-barrier / kt) * np.exp(-tau / t), rho


@dataclass
class MulticomponentResult:
    """Saved states: times (s), mean radius (m), number density (m^-3), precipitate volume
    fraction, matrix mole fractions (t, n), mean precipitate mole fractions (t, n, volume-weighted),
    and the class radii, densities and compositions at each saved time."""

    times: np.ndarray
    mean_radius: np.ndarray
    number_density: np.ndarray
    volume_fraction: np.ndarray
    matrix_composition: np.ndarray
    precipitate_composition: np.ndarray
    classes: list
    steps: int


def solve_multicomponent_precipitation(
    matrix: SolutionPhase,
    precipitate: PrecipitateThermo,
    alloy: Sequence[float],
    T: float,
    t_end: float,
    settings: NucleationSettings,
    save_times: Sequence[float],
    t_start: float = 1e-3,
    max_radius_change: float = 0.02,
    max_matrix_change: float = 2e-4,
    max_classes: int = 400,
    min_classes: int = 100,
    smallest_radius: float = 1e-10,
    nucleus_update: float = 1e-3,
    initial: dict | None = None,
) -> MulticomponentResult:
    """Isothermal precipitation from a supersaturated matrix of mole fractions ``alloy``.

    Steps are explicit, limited so no class radius changes by more than
    ``max_radius_change`` (relative) and no matrix fraction by more than
    ``max_matrix_change``, and growing by at most 20 % from step to step. Classes
    that shrink below ``smallest_radius`` dissolve, their solute returning to the
    matrix through the mass balance. Above ``max_classes`` the two classes closest in
    radius are merged, conserving number, volume and solute; the merged class's Gibbs
    energy is the volume-weighted mean of the two, which keeps the total Gibbs energy
    and hence the mean driving force exact. Below ``min_classes`` the class holding
    the most precipitates splits into two of equal number, 1 % apart in radius with
    the volume kept, as part II (p. 162) splits classes to hold their count, since a
    handful of classes cannot carry a coarsening distribution. The nucleus composition is redone
    when the matrix has moved by ``nucleus_update`` in mole fraction since it was
    last found; between updates its driving force follows the matrix potentials,
    which is exact to first order because the nucleus maximises it. ``initial``
    holds precipitates present at ``t_start`` as arrays "number" (m^-3), "radius" (m)
    and "composition" (k, n).
    """
    alloy = np.asarray(alloy, dtype=float)
    if alloy.shape != (len(matrix.elements),) or abs(alloy.sum() - 1) > 1e-9:
        raise ValueError(f"alloy needs one mole fraction per element of {matrix.elements}, summing to 1")
    if [e.upper() for e in matrix.elements] != precipitate.components:
        raise ValueError("matrix and precipitate must list the same elements in the same order")
    omega = settings.molar_volume
    save = np.sort(np.asarray(save_times, dtype=float))
    if save.size == 0 or save[0] < t_start or save[-1] > t_end:
        raise ValueError("save times must lie between t_start and t_end")
    if not 0 < min_classes <= max_classes // 2:
        raise ValueError("need 0 < min_classes <= max_classes / 2, or splitting and merging would fight")

    if initial is None:
        number, radius, comp = np.zeros(0), np.zeros(0), np.zeros((0, alloy.size))
    else:
        number = np.asarray(initial["number"], dtype=float)
        radius = np.asarray(initial["radius"], dtype=float)
        comp = np.atleast_2d(np.asarray(initial["composition"], dtype=float))
        if not number.shape == radius.shape == comp.shape[:1] or comp.shape[1] != alloy.size:
            raise ValueError("initial needs one number, radius and composition row per class")
    gibbs = np.array([precipitate.gm(x, T) for x in comp])
    nucleus_x, nucleus_g, nucleus_at = None, None, None
    t, dt = t_start, t_start
    out = {k: [] for k in ("t", "r", "n", "phi", "x0", "xp", "classes")}
    steps = 0
    next_save = 0

    def matrix_state(number, radius, comp):
        vol = number * 4 * np.pi / 3 * radius**3
        phi = vol.sum()
        x0 = (alloy - vol @ comp) / (1 - phi)
        return phi, x0, vol

    while next_save < save.size:
        phi, x0, vol = matrix_state(number, radius, comp)
        if np.any(x0 <= 0):
            raise ValueError("the precipitates have used up an element of the matrix; reduce the step limits")
        mu0 = np.asarray(matrix.chemical_potentials(x0, T), dtype=float)
        tracer = R * T * np.asarray(matrix.mobility(x0, T), dtype=float)

        if nucleus_at is None or np.abs(x0 - nucleus_at).max() > nucleus_update:
            nucleus_x, _ = precipitate.nucleus(mu0, T)
            nucleus_g = precipitate.gm(nucleus_x, T)
            nucleus_at = x0.copy()
        f_nuc = (nucleus_x @ mu0 - nucleus_g) / omega

        f_cls = (comp @ mu0 - gibbs) / omega
        rate = growth_rates(radius, comp, f_cls, x0, tracer, settings, T) if radius.size else np.zeros(0)
        j, rho_star = nucleation(nucleus_x, f_nuc, x0, tracer, settings, T, t)

        # step size from the growing classes and the matrix depletion they cause
        dt_new = 1.2 * dt
        # a shrinking class whose whole solute content is within the matrix step limit may
        # dissolve inside a step; following it to the end would cost steps without accuracy,
        # since its rate diverges as it vanishes
        content = number * 4 * np.pi / 3 * radius**3 * np.abs(comp - x0).max(axis=-1) if radius.size else radius
        limiting = (radius > 2 * smallest_radius) & ~((rate < 0) & (content < max_matrix_change))
        if np.any(limiting):
            dt_new = min(dt_new, max_radius_change
                         * np.min(radius[limiting] / np.maximum(np.abs(rate[limiting]), 1e-300)))
        dvol = number * 4 * np.pi * radius**2 * np.where(limiting | (rate > 0), rate, 0.0)
        dx0 = -(dvol @ (comp - x0)) / (1 - phi) if radius.size else np.zeros_like(x0)
        if np.abs(dx0).max() > 0:
            dt_new = min(dt_new, max_matrix_change / np.abs(dx0).max())
        if j > 0:
            new_vol = j * 4 * np.pi / 3 * rho_star**3 * np.abs(nucleus_x - x0).max()
            dt_new = min(dt_new, max_matrix_change / max(new_vol, 1e-300))
        dt = min(dt_new, save[next_save] - t) if save[next_save] > t else dt_new
        dt = max(dt, 1e-12)

        radius = radius + rate * dt
        keep = radius > smallest_radius
        number, radius, comp, gibbs = number[keep], radius[keep], comp[keep], gibbs[keep]
        if j * dt > 0 and np.isfinite(rho_star):
            number = np.append(number, j * dt)
            radius = np.append(radius, 1.05 * rho_star)
            comp = np.vstack([comp, nucleus_x])
            gibbs = np.append(gibbs, nucleus_g)
        if radius.size > max_classes:
            number, radius, comp, gibbs = _merge_closest(number, radius, comp, gibbs)
        while 0 < radius.size < min_classes:
            number, radius, comp, gibbs = _split_most_populated(number, radius, comp, gibbs)
        t += dt
        steps += 1

        while next_save < save.size and t >= save[next_save] * (1 - 1e-9):
            phi, x0, vol = matrix_state(number, radius, comp)
            total = number.sum()
            out["t"].append(t)
            out["n"].append(total)
            out["r"].append((number @ radius) / total if total > 0 else 0.0)
            out["phi"].append(phi)
            out["x0"].append(x0)
            out["xp"].append((vol @ comp) / phi if phi > 0 else np.full(alloy.size, np.nan))
            out["classes"].append({"number": number.copy(), "radius": radius.copy(), "composition": comp.copy()})
            next_save += 1

    return MulticomponentResult(np.array(out["t"]), np.array(out["r"]), np.array(out["n"]), np.array(out["phi"]),
                                np.array(out["x0"]), np.array(out["xp"]), out["classes"], steps)


def _split_most_populated(number, radius, comp, gibbs, spread: float = 0.01):
    # Splitting by the widest gap in radius picks the dissolving dust, whose gaps are
    # huge, and leaves the real particles in identical classes that stop coarsening.
    # The split itself widens the distribution: against LSW the coarsening rate comes
    # out 2.3 times too fast with a 20 % split, 1.2 with 5 % and 1.0 with 1 %.
    i = int(np.argmax(number))
    r_small = radius[i] * (1 - spread)
    r_large = (2 * radius[i] ** 3 - r_small**3) ** (1 / 3)
    number = np.concatenate([number[:i], [number[i] / 2, number[i] / 2], number[i + 1:]])
    radius = np.concatenate([radius[:i], [r_small, r_large], radius[i + 1:]])
    comp = np.vstack([comp[:i], comp[i], comp[i], comp[i + 1:]])
    gibbs = np.concatenate([gibbs[:i], [gibbs[i], gibbs[i]], gibbs[i + 1:]])
    return number, radius, comp, gibbs


def _merge_closest(number, radius, comp, gibbs):
    order = np.argsort(radius)
    number, radius, comp, gibbs = number[order], radius[order], comp[order], gibbs[order]
    gap = np.diff(np.log(radius))
    i = int(np.argmin(gap))
    n = number[i] + number[i + 1]
    v1, v2 = number[i] * radius[i] ** 3, number[i + 1] * radius[i + 1] ** 3
    r = ((v1 + v2) / n) ** (1 / 3)
    x = (v1 * comp[i] + v2 * comp[i + 1]) / (v1 + v2)
    g = (v1 * gibbs[i] + v2 * gibbs[i + 1]) / (v1 + v2)
    number = np.concatenate([number[:i], [n], number[i + 2:]])
    radius = np.concatenate([radius[:i], [r], radius[i + 2:]])
    comp = np.vstack([comp[:i], x, comp[i + 2:]])
    gibbs = np.concatenate([gibbs[:i], [g], gibbs[i + 2:]])
    return number, radius, comp, gibbs
