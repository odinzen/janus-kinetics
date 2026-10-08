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


class PhaseMinimizer:
    """Internal equilibrium of one phase at fixed composition: site fractions that minimise its molar
    Gibbs energy, found by Newton's method on the constrained problem for many compositions at once.

    pycalphad's equilibrium does this too, but too slowly to call for every precipitate class at every
    step; here the Gibbs energy, gradient and Hessian are compiled once and each solve starts from the
    previous site fractions. The phase must hold no vacancies, so that composition is linear in the
    site fractions. Chemical potentials follow from the Lagrange multipliers of the composition
    constraints: with x_n dependent, dG/dx_i = -nu_i, mu_n = G + sum_i x_i nu_i and mu_i = mu_n - nu_i.
    """

    def __init__(self, db: Database, phase: str, components: Sequence[str]):
        import symengine as se
        from pycalphad import Model

        self.components = [c.upper() for c in components]
        model = Model(db, self.components, phase.upper())
        self.y_symbols = list(model.site_fractions)
        names = [str(y.species.name) for y in self.y_symbols]
        if "VA" in names:
            raise ValueError(f"{phase} holds vacancies; the minimiser needs a vacancy-free phase")
        gm = model.GM.xreplace({v.P: se.Float(101325.0)})
        args = self.y_symbols + [v.T]
        grad = [gm.diff(y) for y in self.y_symbols]
        hess = [g.diff(y) for g in grad for y in self.y_symbols]
        self._g = se.Lambdify(args, [gm], cse=True)
        self._grad = se.Lambdify(args, grad, cse=True)
        self._hess = se.Lambdify(args, hess, cse=True)
        sites = np.array([float(a) for a in db.phases[phase.upper()].sublattices])
        sub = np.array([y.sublattice_index for y in self.y_symbols])
        ny = len(self.y_symbols)
        self._sublattices = np.zeros((sites.size, ny))
        self._sublattices[sub, np.arange(ny)] = 1.0
        self._composition = np.zeros((len(self.components), ny))
        for j, (s, name) in enumerate(zip(sub, names)):
            self._composition[self.components.index(name), j] = sites[s] / sites.sum()
        self._a = np.vstack([self._sublattices, self._composition[:-1]])

    def disordered(self, x) -> np.ndarray:
        """Site fractions with every sublattice at composition ``x``, a starting point."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.stack([x[:, self.components.index(str(y.species.name))] for y in self.y_symbols], axis=-1)

    def solve(self, x, y0, T: float, tol: float = 1e-11, max_iter: int = 60):
        """(site fractions (k, ny), G (k,) in J/mol, mu (k, n)) at mole fractions ``x`` (k, n), from ``y0``."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.array(np.atleast_2d(y0), dtype=float)
        k, ny = y.shape
        m = self._a.shape[0]
        rhs_c = np.concatenate([np.ones((k, self._sublattices.shape[0])), x[:, :-1]], axis=1)
        a = np.broadcast_to(self._a, (k, m, ny))
        for _ in range(max_iter):
            args = np.concatenate([y, np.full((k, 1), T)], axis=1)
            g = self._grad(args).reshape(k, ny)
            h = self._hess(args).reshape(k, ny, ny)
            kkt = np.zeros((k, ny + m, ny + m))
            kkt[:, :ny, :ny] = h
            kkt[:, :ny, ny:] = np.transpose(a, (0, 2, 1))
            kkt[:, ny:, :ny] = a
            rhs = np.concatenate([-g, rhs_c - np.einsum("kmj,kj->km", a, y)], axis=1)
            sol = np.linalg.solve(kkt, rhs[..., None])[..., 0]
            dy, nu = sol[:, :ny], sol[:, ny:]
            # stay inside the simplex: step at most 90 % of the way to zero on any site fraction
            with np.errstate(divide="ignore", invalid="ignore"):
                limit = np.where(dy < 0, -0.9 * y / dy, np.inf).min(axis=1)
            step = np.minimum(1.0, limit)[:, None]
            y = np.maximum(y + step * dy, 1e-300)
            if np.abs(dy).max() < tol:
                break
        args = np.concatenate([y, np.full((k, 1), T)], axis=1)
        gm = self._g(args).reshape(k)
        nu_c = nu[:, self._sublattices.shape[0]:]
        mu_n = gm + (x[:, :-1] * nu_c).sum(axis=1)
        mu = np.concatenate([mu_n[:, None] - nu_c, mu_n[:, None]], axis=1)
        return y, gm, mu


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

    def site_fractions(self, x: Sequence[float], T: float) -> np.ndarray:
        """Equilibrium site fractions of the phase at mole fractions ``x``, in pycalphad's order, from the
        composition set holding most of the material (a start for :class:`PhaseMinimizer`)."""
        x = np.asarray(x, dtype=float)
        eq = self._eq(T, {v.X(c): xi for c, xi in zip(self.components[:-1], x[:-1])})
        npf = self._values(eq, "NP")
        y = eq.Y.values.reshape(npf.size, -1)
        best = int(np.nanargmax(npf))
        return y[best][np.isfinite(y[best])]

    @property
    def minimizer(self) -> PhaseMinimizer:
        if not hasattr(self, "_minimizer"):
            self._minimizer = PhaseMinimizer(self.db, self.phase, self.components)
        return self._minimizer


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


def evolution_rates(radius, composition, mu_precipitate, x_matrix, mu_matrix, tracer, precipitate_tracer,
                    settings: NucleationSettings, T: float):
    """(drho/dt (k,), dx/dt (k, n)) for classes whose composition evolves, part I eqs. (27)-(39) for
    substitutional elements and infinite interface mobility.

    The paper's reference element 1 is the last element here. Per class the rates solve
    A y = B with y = (drho/dt, dc_i/dt for i < n), c = x / Omega and Delta_i = c_ki - c_0i:

        A_11 = R T rho sum_i Delta_i^2 / (c_0i D_0i)
        A_1i = (R T rho^2 / 3) (Delta_i / (c_0i D_0i) - Delta_n / (c_0n D_0n))
        A_ij = (R T rho^3 / 45) [(1/(c_ki D_ki) + 5/(c_0i D_0i)) delta_ij + 1/(c_kn D_kn) + 5/(c_0n D_0n)]
        B_1 = F - 2 gamma / rho,   B_i = -(rho / 3) (mu_ki - mu_0i - mu_kn + mu_0n)

    The printed eq. (32) has 5/(3 c_01 D_01) in the last term; eq. (25), from which it comes, and the
    interstitial counterpart eq. (33) have 5/(c_01 D_01), as does differentiating the dissipation of
    eqs. (15) and (23), so the 3 is taken as a misprint. D_ki is the tracer diffusivity of i in the
    precipitate; as it goes to zero the composition freezes and eq. (45) is recovered.
    """
    rho = np.asarray(radius, dtype=float)
    omega = settings.molar_volume
    xk = np.asarray(composition, dtype=float)
    k, n = xk.shape
    c0d = np.asarray(x_matrix) * np.asarray(tracer) / omega  # c_0i D_0i
    ckd = xk * np.asarray(precipitate_tracer) / omega  # c_ki D_ki
    delta = (xk - x_matrix) / omega
    rt = R * T
    a = np.zeros((k, n, n))
    a[:, 0, 0] = rt * rho * (delta**2 / c0d).sum(axis=1)
    cross = rt * rho[:, None] ** 2 / 3 * (delta[:, :-1] / c0d[:-1] - (delta[:, -1] / c0d[-1])[:, None])
    a[:, 0, 1:] = cross
    a[:, 1:, 0] = cross
    diag = 1 / ckd[:, :-1] + 5 / c0d[:-1]
    common = 1 / ckd[:, -1] + 5 / c0d[-1]
    a[:, 1:, 1:] = rt * rho[:, None, None] ** 3 / 45 * (diag[:, :, None] * np.eye(n - 1) + common[:, None, None])
    mu_k = np.asarray(mu_precipitate, dtype=float)
    f = (xk @ np.asarray(mu_matrix) - (xk * mu_k).sum(axis=1)) / omega
    b = np.empty((k, n))
    b[:, 0] = f - 2 * settings.gamma / rho
    b[:, 1:] = -rho[:, None] / 3 * (mu_k[:, :-1] - mu_matrix[:-1] - (mu_k[:, -1] - mu_matrix[-1])[:, None])
    y = np.linalg.solve(a, b[..., None])[..., 0]
    dx = np.empty((k, n))
    dx[:, :-1] = omega * y[:, 1:]
    dx[:, -1] = -dx[:, :-1].sum(axis=1)
    return y[:, 0], dx


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
    max_composition_change: float = 1e-3,
    max_classes: int = 400,
    min_classes: int = 100,
    smallest_radius: float = 1e-10,
    nucleus_update: float = 1e-3,
    initial: dict | None = None,
    precipitate_tracer: Sequence[float] | None = None,
) -> MulticomponentResult:
    """Isothermal precipitation from a supersaturated matrix of mole fractions ``alloy``.

    Without ``precipitate_tracer`` each class keeps the composition it nucleated with and
    grows by eq. (45). With it (tracer diffusivities of each element in the precipitate,
    m^2/s) the class compositions evolve by eqs. (27)-(39), see :func:`evolution_rates`,
    and the precipitate's chemical potentials at each class composition come from a
    :class:`PhaseMinimizer` of the precipitate phase.

    Steps are explicit, limited so no class radius changes by more than
    ``max_radius_change`` (relative), no class mole fraction by more than
    ``max_composition_change`` and no matrix fraction by more than
    ``max_matrix_change``, and growing by at most 20 % from step to step. Classes
    that shrink below ``smallest_radius`` dissolve, their solute returning to the
    matrix through the mass balance. Above ``max_classes`` the two classes closest in
    radius are merged, conserving number, volume and solute; the merged class's Gibbs
    energy is the volume-weighted mean of the two, which keeps the total Gibbs energy
    and hence the mean driving force exact (with evolving compositions it is
    recomputed at the merged composition, the particle being taken as uniform). Below
    ``min_classes`` the class holding the most precipitates splits into two of equal
    number, 1 % apart in radius with the volume kept, as part II (p. 162) splits
    classes to hold their count, since a handful of classes cannot carry a coarsening
    distribution. The nucleus composition is redone when the matrix has moved by
    ``nucleus_update`` in mole fraction since it was last found; between updates its
    driving force follows the matrix potentials, which is exact to first order because
    the nucleus maximises it. ``initial`` holds precipitates present at ``t_start`` as
    arrays "number" (m^-3), "radius" (m) and "composition" (k, n).
    """
    alloy = np.asarray(alloy, dtype=float)
    n_el = alloy.size
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
    evolving = precipitate_tracer is not None
    if evolving:
        precipitate_tracer = np.asarray(precipitate_tracer, dtype=float)
        if precipitate_tracer.shape != (n_el,) or np.any(precipitate_tracer <= 0):
            raise ValueError("precipitate_tracer needs one positive diffusivity per element")
        if precipitate.fixed is not None:
            raise ValueError("a precipitate of fixed composition cannot evolve; leave out precipitate_tracer")
        minimizer = precipitate.minimizer

    cls = {"number": np.zeros(0), "radius": np.zeros(0), "comp": np.zeros((0, n_el)), "gibbs": np.zeros(0)}
    if initial is not None:
        cls["number"] = np.asarray(initial["number"], dtype=float)
        cls["radius"] = np.asarray(initial["radius"], dtype=float)
        cls["comp"] = np.atleast_2d(np.asarray(initial["composition"], dtype=float))
        if not cls["number"].shape == cls["radius"].shape == cls["comp"].shape[:1] or cls["comp"].shape[1] != n_el:
            raise ValueError("initial needs one number, radius and composition row per class")
        cls["gibbs"] = np.array([precipitate.gm(x, T) for x in cls["comp"]])
    if evolving:
        cls["sites"] = np.array([precipitate.site_fractions(x, T) for x in cls["comp"]]).reshape(
            len(cls["number"]), len(minimizer.y_symbols))
    nucleus_x = nucleus_g = nucleus_at = nucleus_y = None
    t, dt = t_start, t_start
    out = {k: [] for k in ("t", "r", "n", "phi", "x0", "xp", "classes")}
    steps = 0
    next_save = 0

    def matrix_state(c):
        vol = c["number"] * 4 * np.pi / 3 * c["radius"] ** 3
        phi = vol.sum()
        return phi, (alloy - vol @ c["comp"]) / (1 - phi), vol

    while next_save < save.size:
        phi, x0, vol = matrix_state(cls)
        if np.any(x0 <= 0):
            raise ValueError("the precipitates have used up an element of the matrix; reduce the step limits")
        mu0 = np.asarray(matrix.chemical_potentials(x0, T), dtype=float)
        tracer = R * T * np.asarray(matrix.mobility(x0, T), dtype=float)

        if nucleus_at is None or np.abs(x0 - nucleus_at).max() > nucleus_update:
            nucleus_x, _ = precipitate.nucleus(mu0, T)
            nucleus_g = precipitate.gm(nucleus_x, T)
            nucleus_at = x0.copy()
            if evolving:
                nucleus_y = precipitate.site_fractions(nucleus_x, T)
        f_nuc = (nucleus_x @ mu0 - nucleus_g) / omega

        radius, comp = cls["radius"], cls["comp"]
        dcomp = np.zeros_like(comp)
        if radius.size and evolving:
            cls["sites"], cls["gibbs"], mu_k = minimizer.solve(comp, cls["sites"], T)
            rate, dcomp = evolution_rates(radius, comp, mu_k, x0, mu0, tracer, precipitate_tracer, settings, T)
        elif radius.size:
            rate = growth_rates(radius, comp, (comp @ mu0 - cls["gibbs"]) / omega, x0, tracer, settings, T)
        else:
            rate = np.zeros(0)
        j, rho_star = nucleation(nucleus_x, f_nuc, x0, tracer, settings, T, t)

        # step size from the growing classes and the matrix depletion they cause
        dt_new = 1.2 * dt
        number = cls["number"]
        # a shrinking class whose whole solute content is within the matrix step limit may
        # dissolve inside a step; following it to the end would cost steps without accuracy,
        # since its rate diverges as it vanishes
        content = number * 4 * np.pi / 3 * radius**3 * np.abs(comp - x0).max(axis=-1) if radius.size else radius
        limiting = (radius > 2 * smallest_radius) & ~((rate < 0) & (content < max_matrix_change))
        if np.any(limiting):
            dt_new = min(dt_new, max_radius_change
                         * np.min(radius[limiting] / np.maximum(np.abs(rate[limiting]), 1e-300)))
            if evolving:
                dt_new = min(dt_new, max_composition_change / max(np.abs(dcomp[limiting]).max(), 1e-300))
        dvol = number * 4 * np.pi * radius**2 * np.where(limiting | (rate > 0), rate, 0.0)
        dx0 = -(dvol @ (comp - x0)) / (1 - phi) if radius.size else np.zeros_like(x0)
        if evolving and radius.size:
            dx0 -= (number * 4 * np.pi / 3 * radius**3) @ np.where(limiting[:, None], dcomp, 0.0) / (1 - phi)
        if np.abs(dx0).max() > 0:
            dt_new = min(dt_new, max_matrix_change / np.abs(dx0).max())
        if j > 0:
            new_vol = j * 4 * np.pi / 3 * rho_star**3 * np.abs(nucleus_x - x0).max()
            dt_new = min(dt_new, max_matrix_change / max(new_vol, 1e-300))
        dt = min(dt_new, save[next_save] - t) if save[next_save] > t else dt_new
        dt = max(dt, 1e-12)

        cls["radius"] = radius + rate * dt
        if evolving and radius.size:
            cls["comp"] = np.clip(comp + dcomp * dt, 1e-12, None)
            cls["comp"] /= cls["comp"].sum(axis=1, keepdims=True)
        cls = {key: val[cls["radius"] > smallest_radius] for key, val in cls.items()}
        if j * dt > 0 and np.isfinite(rho_star):
            new = {"number": j * dt, "radius": 1.05 * rho_star, "comp": nucleus_x, "gibbs": nucleus_g}
            if evolving:
                new["sites"] = nucleus_y
            cls = {key: np.concatenate([val, np.atleast_1d(new[key])[None] if val.ndim == 2 else [new[key]]])
                   for key, val in cls.items()}
        if cls["radius"].size > max_classes:
            cls = _merge_closest(cls)
        while 0 < cls["radius"].size < min_classes:
            cls = _split_most_populated(cls)
        t += dt
        steps += 1

        while next_save < save.size and t >= save[next_save] * (1 - 1e-9):
            phi, x0, vol = matrix_state(cls)
            total = cls["number"].sum()
            out["t"].append(t)
            out["n"].append(total)
            out["r"].append((cls["number"] @ cls["radius"]) / total if total > 0 else 0.0)
            out["phi"].append(phi)
            out["x0"].append(x0)
            out["xp"].append((vol @ cls["comp"]) / phi if phi > 0 else np.full(n_el, np.nan))
            out["classes"].append({"number": cls["number"].copy(), "radius": cls["radius"].copy(),
                                   "composition": cls["comp"].copy()})
            next_save += 1

    return MulticomponentResult(np.array(out["t"]), np.array(out["r"]), np.array(out["n"]), np.array(out["phi"]),
                                np.array(out["x0"]), np.array(out["xp"]), out["classes"], steps)


def _split_most_populated(cls: dict, spread: float = 0.01) -> dict:
    # Splitting by the widest gap in radius picks the dissolving dust, whose gaps are
    # huge, and leaves the real particles in identical classes that stop coarsening.
    # The split itself widens the distribution: against LSW the coarsening rate comes
    # out 2.3 times too fast with a 20 % split, 1.2 with 5 % and 1.0 with 1 %.
    i = int(np.argmax(cls["number"]))
    r = cls["radius"][i]
    r_small = r * (1 - spread)
    r_large = (2 * r**3 - r_small**3) ** (1 / 3)
    out = {key: np.insert(val, i, val[i], axis=0) for key, val in cls.items()}
    out["number"][i:i + 2] = cls["number"][i] / 2
    out["radius"][i:i + 2] = (r_small, r_large)
    return out


def _merge_closest(cls: dict) -> dict:
    order = np.argsort(cls["radius"])
    cls = {key: val[order] for key, val in cls.items()}
    number, radius = cls["number"], cls["radius"]
    i = int(np.argmin(np.diff(np.log(radius))))
    v = number[i:i + 2] * radius[i:i + 2] ** 3
    merged = {key: (v @ val[i:i + 2]) / v.sum() for key, val in cls.items()}  # volume-weighted
    merged["number"] = number[i] + number[i + 1]
    merged["radius"] = (v.sum() / merged["number"]) ** (1 / 3)
    return {key: np.concatenate([val[:i], np.asarray(merged[key])[None], val[i + 2:]]) for key, val in cls.items()}
