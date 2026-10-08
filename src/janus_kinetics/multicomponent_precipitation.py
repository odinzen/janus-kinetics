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

Scope: spherical precipitates, one molar volume for every phase, dilute enough that
diffusion fields do not overlap (the assumption behind part I's eq. 22). Several phases,
or one phase at several kinds of site, compete for the matrix through
:func:`solve_competing_precipitation`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from scipy.optimize import brentq

from .phase import SolutionPhase
from .schedule import Temperature, temperature_at
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
    both phases), lattice parameter (m) of the attachment rate eq. (46), nucleation sites per
    m^3 (default N_A / molar_volume, every lattice site, homogeneous nucleation), and the misfit
    energy lambda (J/m^3) stored per volume of precipitate, which part I eq. (37) subtracts from the
    chemical driving force (see :func:`misfit_energy`).

    ``site`` places the nuclei on grain boundaries ("boundary"), edges where three grains meet
    ("edge") or corners where four meet ("corner"), with ``boundary_ratio`` k = gamma_gb / (2 gamma)
    from the grain boundary energy; the barrier is then the fraction :func:`junction_barrier_factor`
    of the homogeneous one. ``sites`` should then count the boundary, edge or corner sites per m^3,
    which :func:`nucleation_sites` gives from the grain size. Nuclei on dislocations keep the bulk
    barrier here, with ``sites`` from the dislocation density.
    """

    gamma: float
    molar_volume: float
    lattice_parameter: float
    sites: float | None = None
    misfit: float = 0.0
    site: Literal["bulk", "boundary", "edge", "corner"] = "bulk"
    boundary_ratio: float = 0.0

    def __post_init__(self):
        if self.gamma <= 0 or self.molar_volume <= 0:
            raise ValueError("interfacial energy and molar volume must be positive")
        if self.misfit < 0:
            raise ValueError("the misfit energy cannot be negative")
        if not 1e-11 < self.lattice_parameter < 1e-8:
            raise ValueError(f"lattice parameter {self.lattice_parameter:g} m is not a lattice parameter "
                             "(about 4e-10 m; give it in metres)")
        junction_barrier_factor(self.site, self.boundary_ratio)  # checks site and k

    @property
    def barrier_factor(self) -> float:
        return junction_barrier_factor(self.site, self.boundary_ratio)

    @property
    def site_density(self) -> float:
        return N_A / self.molar_volume if self.sites is None else self.sites


_CRITICAL_RATIO = {"bulk": np.inf, "boundary": 1.0, "edge": np.sqrt(3) / 2, "corner": np.sqrt(2 / 3)}


def junction_factors(site: str, k: float) -> tuple[float, float, float]:
    """Coefficients (a, b, c) of Clemm and Fisher, Acta Metall. 3 (1955) 70, for a nucleus whose bounding
    spherical caps have radius of curvature r: it removes a r^2 of matrix grain boundary, creates b r^2
    of interface and has volume c r^3. k = gamma_AA / (2 gamma_AB) = cos(theta), the dihedral half angle.
    Two-grain junction (a lens) p. 71; three-grain p. 71-72; four-grain p. 72."""
    if site == "bulk":
        return 0.0, 4 * np.pi, 4 * np.pi / 3
    if site == "boundary":
        return np.pi * (1 - k * k), 4 * np.pi * (1 - k), 2 * np.pi / 3 * (2 - 3 * k + k**3)
    if site == "edge":
        alpha = np.arcsin(1 / (2 * np.sqrt(1 - k * k)))
        beta = np.arccos(k / np.sqrt(3 * (1 - k * k)))
        a = 3 * beta * (1 - k * k) - k * np.sqrt(3 - 4 * k * k)
        b = 12 * (np.pi / 2 - alpha - k * beta)
        c = 2 * (np.pi - 2 * alpha + k * k / 3 * np.sqrt(3 - 4 * k * k) - beta * k * (3 - k * k))
        return a, b, c
    if site == "corner":
        big_k = 4 / 3 * np.sqrt(1.5 - 2 * k * k) - 2 * k / 3
        phi = np.arcsin(big_k / (2 * np.sqrt(1 - k * k)))
        delta = np.arccos(np.clip((np.sqrt(2) - k * np.sqrt(3 - big_k**2)) / (big_k * np.sqrt(1 - k * k)), -1, 1))
        root = np.sqrt(max(1 - k * k - big_k**2 / 4, 0.0)) - big_k / np.sqrt(8)
        a = 3 * (2 * phi * (1 - k * k) - big_k * root)
        b = 24 * (np.pi / 3 - k * phi - delta)
        c = 2 * (4 * (np.pi / 3 - delta) + k * big_k * root - 2 * k * phi * (3 - k * k))
        return a, b, c
    raise ValueError(f"site must be one of {sorted(_CRITICAL_RATIO)}, got {site!r}")


def junction_barrier_factor(site: str, k: float) -> float:
    """Nucleation barrier at a grain junction over the homogeneous one. Clemm and Fisher's
    W* = (4/27) (b gamma_AB - a gamma_AA)^3 / (c^2 Delta f_v^2) with gamma_AA = 2 k gamma_AB, divided by
    16 pi gamma_AB^3 / (3 Delta f_v^2), is (b - 2 k a)^3 / (36 pi c^2). The critical radius of curvature
    is that of the homogeneous nucleus, 2 gamma / F. Beyond the critical k of the paper (1, sqrt(3)/2,
    sqrt(2/3)) there is no barrier and the phase grows from the junction without nucleating."""
    if site not in _CRITICAL_RATIO:
        raise ValueError(f"site must be one of {sorted(_CRITICAL_RATIO)}, got {site!r}")
    if site == "bulk":
        return 1.0
    if not 0 <= k < _CRITICAL_RATIO[site]:
        raise ValueError(f"k = gamma_gb / (2 gamma) must lie in [0, {_CRITICAL_RATIO[site]:.4f}) for a {site} "
                         "nucleus; beyond it there is no nucleation barrier (Clemm and Fisher 1955, p. 72)")
    a, b, c = junction_factors(site, k)
    return (b - 2 * k * a) ** 3 / (36 * np.pi * c * c)


def grain_structure(grain_size: float) -> tuple[float, float, float]:
    """(grain boundary area in m^2, grain edge length in m, number of grain corners) per m^3 for equal
    tetrakaidecahedral grains whose square faces lie ``grain_size`` D apart, Cahn, Acta Metall. 4 (1956)
    449, eq. (1): S = 3.35 / D, L = 8.5 / D^2, C = 12.0 / D^3. The exact values of that solid are used
    (edge D / (2 sqrt 2), 2 / D^3 grains per volume, faces shared by two grains, edges by three and
    corners by four): (3 + 6 sqrt 3) / (4 D), 6 sqrt 2 / D^2 and 12 / D^3."""
    if grain_size <= 0:
        raise ValueError("the grain size must be positive")
    d = grain_size
    return (3 + 6 * np.sqrt(3)) / (4 * d), 6 * np.sqrt(2) / d**2, 12 / d**3


def nucleation_sites(site: str, molar_volume: float, grain_size: float | None = None,
                     dislocation_density: float | None = None) -> float:
    """Nucleation sites per m^3 for :class:`NucleationSettings`: every atom for "bulk", and otherwise the
    atoms on the defect, as part II (p. 162) counts the atoms on a dislocation line. With the atomic
    distance w = (Omega / N_A)^(1/3), a boundary of area S holds S / w^2 atoms in one layer, an edge of
    length L holds L / w, and a corner is taken as one site; the areas, lengths and corners per volume
    come from ``grain_size`` through :func:`grain_structure`. "dislocation" counts
    ``dislocation_density`` (m^-2) / w. Taking one atomic layer, column or site is this package's choice;
    neither paper fixes it for grain junctions."""
    if molar_volume <= 0:
        raise ValueError("the molar volume must be positive")
    atom = molar_volume / N_A
    w = atom ** (1 / 3)
    if site == "bulk":
        return 1 / atom
    if site == "dislocation":
        if dislocation_density is None or dislocation_density <= 0:
            raise ValueError("dislocation sites need a positive dislocation_density (m^-2)")
        return dislocation_density / w
    if site in ("boundary", "edge", "corner"):
        if grain_size is None:
            raise ValueError(f"{site} sites need a grain_size (m)")
        area, length, corners = grain_structure(grain_size)
        return {"boundary": area / w**2, "edge": length / w, "corner": corners}[site]
    raise ValueError(f"site must be bulk, boundary, edge, corner or dislocation, got {site!r}")


def misfit_energy(strain: float, youngs_modulus: float, poisson: float, yield_stress: float | None = None) -> float:
    """Energy per volume of precipitate (J/m^3) from a linear misfit strain eps0 (volume misfit 3 eps0),
    part I Appendix A: elastic lambda0 = eps0^2 E / (1 - nu), eq. (A.1), for equal elastic constants in
    both phases. With a yield stress the matrix around the particle may yield (Fischer and Oberaigner,
    as the appendix gives it): S = sigma_f (1 - nu) / (E eps0), eq. (A.2), and lambda = lambda0 f(S) with
    f(S) = 2 S (1 - ln S) - S^2 for S < 1, eq. (A.3), which exceeds 1 between S = 0.199 and 1 (plastic
    work added to the stored elastic energy)."""
    if youngs_modulus <= 0 or not -1 < poisson < 0.5:
        raise ValueError("need E > 0 and -1 < nu < 0.5")
    elastic = strain**2 * youngs_modulus / (1 - poisson)
    if yield_stress is None or strain == 0:
        return elastic
    s = yield_stress * (1 - poisson) / (youngs_modulus * abs(strain))
    return elastic * (2 * s * (1 - np.log(s)) - s**2) if s < 1 else elastic


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
    f = (xk @ np.asarray(mu_matrix) - (xk * mu_k).sum(axis=1)) / omega - settings.misfit
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
    driving force ``driving_force`` (J/m^3), part I eqs. (40)-(41) and (46)-(48).

    On a grain junction (``settings.site``) only the barrier changes, by Clemm and Fisher's factor;
    the Zeldovich factor and attachment rate keep their homogeneous form, which the paper does not
    address. The nucleus then enters the class array as a sphere of the critical radius of
    curvature, which grows by the same capillarity law; its volume is larger than that of the
    lens or tetrahedron it stands for, a choice of this package."""
    if driving_force <= 0 or t <= 0:
        return 0.0, np.inf
    kt = K_B * T
    g, omega = settings.gamma, settings.molar_volume
    rho = 2 * g / driving_force
    barrier = 16 * np.pi * g**3 / (3 * driving_force**2) * settings.barrier_factor
    resistance = (((np.asarray(composition) - x_matrix) ** 2) / (x_matrix * tracer)).sum() / omega
    beta = 4 * np.pi * rho**2 / (settings.lattice_parameter**4 * omega) / resistance
    zeldovich = omega / N_A / (2 * np.pi * rho**2) * np.sqrt(g / kt)
    tau = 1 / (2 * beta * zeldovich**2)
    return settings.site_density * zeldovich * beta * np.exp(-barrier / kt) * np.exp(-tau / t), rho


@dataclass
class MulticomponentResult:
    """Saved states of one precipitate population: times (s), mean radius (m), number density (m^-3),
    its volume fraction, the matrix mole fractions (t, n), its mean mole fractions (t, n,
    volume-weighted), and the class radii, densities and compositions at each saved time."""

    times: np.ndarray
    mean_radius: np.ndarray
    number_density: np.ndarray
    volume_fraction: np.ndarray
    matrix_composition: np.ndarray
    precipitate_composition: np.ndarray
    classes: list
    steps: int


@dataclass(frozen=True)
class Population:
    """One precipitate phase nucleating at one kind of site, for :func:`solve_competing_precipitation`.
    ``initial`` and ``precipitate_tracer`` are as in :func:`solve_multicomponent_precipitation`. A phase
    may form several populations, in the bulk and on grain boundaries say, as part II (p. 158) creates
    one kinetic object per phase and nucleation site."""

    precipitate: PrecipitateThermo
    settings: NucleationSettings
    initial: dict | None = None
    precipitate_tracer: Sequence[float] | None = None


def solve_multicomponent_precipitation(
    matrix: SolutionPhase,
    precipitate: PrecipitateThermo,
    alloy: Sequence[float],
    T: Temperature,
    t_end: float,
    settings: NucleationSettings,
    save_times: Sequence[float],
    t_start: float = 1e-3,
    max_temperature_change: float = 1.0,
    max_radius_change: float = 0.02,
    max_matrix_change: float = 2e-4,
    max_composition_change: float = 1e-3,
    max_classes: int = 400,
    min_classes: int = 100,
    smallest_radius: float = 1e-10,
    nucleus_update: float = 1e-3,
    min_class_number: float = 1e-6,
    initial: dict | None = None,
    precipitate_tracer: Sequence[float] | None = None,
) -> MulticomponentResult:
    """Precipitation from a supersaturated matrix of mole fractions ``alloy`` at temperature ``T``, a
    constant or a :class:`TemperatureSchedule` (quench and age, ramps, multi-step treatments).
    Along a schedule the steps are also limited to ``max_temperature_change`` (K), and the
    Gibbs energy of every class is recomputed at each new temperature (by the
    :class:`PhaseMinimizer` for a precipitate of variable composition, which must then be
    vacancy-free), as is the nucleus.

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
    the nucleus maximises it. Nuclei open a new class only once more than
    ``min_class_number`` times the precipitates present (and at least one per m^3) have
    formed; until then they are held back. ``initial`` holds precipitates present at
    ``t_start`` as arrays "number" (m^-3), "radius" (m) and "composition" (k, n).
    """
    return solve_competing_precipitation(
        matrix, [Population(precipitate, settings, initial, precipitate_tracer)], alloy, T, t_end, save_times,
        t_start=t_start, max_temperature_change=max_temperature_change, max_radius_change=max_radius_change,
        max_matrix_change=max_matrix_change, max_composition_change=max_composition_change,
        max_classes=max_classes, min_classes=min_classes, smallest_radius=smallest_radius,
        nucleus_update=nucleus_update, min_class_number=min_class_number)[0]


class _PopulationState:
    """Classes, nucleus and rates of one population during a run."""

    def __init__(self, pop: Population, n_el: int, varying: bool, T0: float):
        self.precipitate, self.settings = pop.precipitate, pop.settings
        self.evolving = pop.precipitate_tracer is not None
        self.varying = varying
        if varying or self.evolving:
            self.minimizer = self.precipitate.minimizer if self.precipitate.fixed is None else None
        if self.evolving:
            self.precipitate_tracer = np.asarray(pop.precipitate_tracer, dtype=float)
            if self.precipitate_tracer.shape != (n_el,) or np.any(self.precipitate_tracer <= 0):
                raise ValueError("precipitate_tracer needs one positive diffusivity per element")
            if self.precipitate.fixed is not None:
                raise ValueError("a precipitate of fixed composition cannot evolve; leave out precipitate_tracer")
        cls = {"number": np.zeros(0), "radius": np.zeros(0), "comp": np.zeros((0, n_el)), "gibbs": np.zeros(0)}
        if pop.initial is not None:
            cls["number"] = np.asarray(pop.initial["number"], dtype=float)
            cls["radius"] = np.asarray(pop.initial["radius"], dtype=float)
            cls["comp"] = np.atleast_2d(np.asarray(pop.initial["composition"], dtype=float))
            if not cls["number"].shape == cls["radius"].shape == cls["comp"].shape[:1] or cls["comp"].shape[1] != n_el:
                raise ValueError("initial needs one number, radius and composition row per class")
            cls["gibbs"] = np.array([self.precipitate.gm(x, T0) for x in cls["comp"]])
        self.track_sites = (self.evolving or varying) and self.precipitate.fixed is None
        if self.track_sites:
            cls["sites"] = np.array([self.precipitate.site_fractions(x, T0) for x in cls["comp"]]).reshape(
                len(cls["number"]), len(self.minimizer.y_symbols))
        self.cls = cls
        self.nucleus_x = self.nucleus_g = self.nucleus_at = self.nucleus_y = self.nucleus_t = None
        self.pending = 0.0  # nuclei formed but not yet numerous enough for a class of their own
        self.gibbs_t = T0

    def volumes(self) -> np.ndarray:
        return self.cls["number"] * 4 * np.pi / 3 * self.cls["radius"] ** 3

    def rates(self, x0, mu0, tracer, T: float, t: float, nucleus_update: float):
        """Growth (and composition) rates of the classes and the nucleation rate in the current matrix."""
        cls, omega = self.cls, self.settings.molar_volume
        if self.nucleus_at is None or np.abs(x0 - self.nucleus_at).max() > nucleus_update or abs(T - self.nucleus_t) > 0.5:
            self.nucleus_x, _ = self.precipitate.nucleus(mu0, T)
            self.nucleus_g = self.precipitate.gm(self.nucleus_x, T)
            self.nucleus_at, self.nucleus_t = x0.copy(), T
            if self.track_sites:
                self.nucleus_y = self.precipitate.site_fractions(self.nucleus_x, T)
        if self.varying and not self.evolving and cls["radius"].size and T != self.gibbs_t:
            # a class keeps its composition, but its Gibbs energy follows the temperature
            if self.precipitate.fixed is not None:
                cls["gibbs"] = np.full(cls["radius"].size, self.precipitate.gm(self.precipitate.fixed, T))
            else:
                cls["sites"], cls["gibbs"], _ = self.minimizer.solve(cls["comp"], cls["sites"], T)
        self.gibbs_t = T
        f_nuc = (self.nucleus_x @ mu0 - self.nucleus_g) / omega - self.settings.misfit

        radius, comp = cls["radius"], cls["comp"]
        self.dcomp = np.zeros_like(comp)
        if radius.size and self.evolving:
            cls["sites"], cls["gibbs"], mu_k = self.minimizer.solve(comp, cls["sites"], T)
            self.rate, self.dcomp = evolution_rates(radius, comp, mu_k, x0, mu0, tracer, self.precipitate_tracer,
                                                    self.settings, T)
        elif radius.size:
            self.rate = growth_rates(radius, comp, (comp @ mu0 - cls["gibbs"]) / omega - self.settings.misfit, x0,
                                     tracer, self.settings, T)
        else:
            self.rate = np.zeros(0)
        self.j, self.rho_star = nucleation(self.nucleus_x, f_nuc, x0, tracer, self.settings, T, t)

    def advance(self, dt: float, smallest_radius: float, min_class_number: float, max_classes: int,
                min_classes: int):
        cls = self.cls
        radius, comp = cls["radius"], cls["comp"]
        cls["radius"] = radius + self.rate * dt
        if self.evolving and radius.size:
            cls["comp"] = np.clip(comp + self.dcomp * dt, 1e-12, None)
            cls["comp"] /= cls["comp"].sum(axis=1, keepdims=True)
        cls = {key: val[cls["radius"] > smallest_radius] for key, val in cls.items()}
        self.pending += self.j * dt if np.isfinite(self.rho_star) else 0.0
        # Late in a run the nucleation rate is astronomically small but not zero; a class for every
        # step's 1e-100 nuclei would fill the array with empty classes, whose merging then collapses
        # the real size distribution into one class that can no longer coarsen.
        if np.isfinite(self.rho_star) and self.pending > max(min_class_number * cls["number"].sum(), 1.0):
            new = {"number": self.pending, "radius": 1.05 * self.rho_star, "comp": self.nucleus_x,
                   "gibbs": self.nucleus_g}
            self.pending = 0.0
            if self.track_sites:
                new["sites"] = self.nucleus_y
            cls = {key: np.concatenate([val, np.atleast_1d(new[key])[None] if val.ndim == 2 else [new[key]]])
                   for key, val in cls.items()}
        if cls["radius"].size > max_classes:
            cls = _merge_closest(cls)
        while 0 < cls["radius"].size < min_classes:
            cls = _split_most_populated(cls)
        self.cls = cls


def solve_competing_precipitation(
    matrix: SolutionPhase,
    populations: Sequence[Population],
    alloy: Sequence[float],
    T: Temperature,
    t_end: float,
    save_times: Sequence[float],
    t_start: float = 1e-3,
    max_temperature_change: float = 1.0,
    max_radius_change: float = 0.02,
    max_matrix_change: float = 2e-4,
    max_composition_change: float = 1e-3,
    max_classes: int = 400,
    min_classes: int = 100,
    smallest_radius: float = 1e-10,
    nucleus_update: float = 1e-3,
    min_class_number: float = 1e-6,
) -> list[MulticomponentResult]:
    """Several precipitate populations drawing on one matrix, one result per population in the order
    given. Part I's mass balance, eq. (4), sums over precipitates of every kind, and part II (section
    3.5) runs such competitions to follow precipitation sequences.

    Each population nucleates, grows and coarsens as in :func:`solve_multicomponent_precipitation`,
    whose arguments these are. The step is common to all of them: its matrix-change limits count the
    depletion by every population together, and shrinking classes of all populations share one
    dissolution budget. All populations need the same molar volume, part I's single Omega (p. 167),
    with which volume fractions carry the mass balance.
    """
    alloy = np.asarray(alloy, dtype=float)
    n_el = alloy.size
    if alloy.shape != (len(matrix.elements),) or abs(alloy.sum() - 1) > 1e-9:
        raise ValueError(f"alloy needs one mole fraction per element of {matrix.elements}, summing to 1")
    populations = list(populations)
    if not populations:
        raise ValueError("need at least one precipitate population")
    if any([e.upper() for e in matrix.elements] != p.precipitate.components for p in populations):
        raise ValueError("matrix and precipitate must list the same elements in the same order")
    omega = populations[0].settings.molar_volume
    if any(p.settings.molar_volume != omega for p in populations):
        raise ValueError("all populations need the same molar volume; part I takes one for every phase")
    save = np.sort(np.asarray(save_times, dtype=float))
    if save.size == 0 or save[0] < t_start or save[-1] > t_end:
        raise ValueError("save times must lie between t_start and t_end")
    if not 0 < min_classes <= max_classes // 2:
        raise ValueError("need 0 < min_classes <= max_classes / 2, or splitting and merging would fight")
    varying = callable(T)
    schedule = T  # T is reassigned to the current temperature each step
    temp = (lambda t: temperature_at(schedule, t))
    states = [_PopulationState(p, n_el, varying, temp(t_start)) for p in populations]
    t, dt = t_start, t_start
    out = [{k: [] for k in ("t", "r", "n", "phi", "xp", "classes")} for _ in states]
    out_x0 = []
    steps = 0
    next_save = 0
    previous_dx0 = None

    def matrix_state():
        vols = [s.volumes() for s in states]
        phi = sum(vol.sum() for vol in vols)
        return phi, (alloy - sum(vol @ s.cls["comp"] for vol, s in zip(vols, states))) / (1 - phi), vols

    while next_save < save.size:
        T = temp(t)
        phi, x0, _ = matrix_state()
        if np.any(x0 <= 0):
            raise ValueError("the precipitates have used up an element of the matrix; reduce the step limits")
        mu0 = np.asarray(matrix.chemical_potentials(x0, T), dtype=float)
        tracer = R * T * np.asarray(matrix.mobility(x0, T), dtype=float)
        for s in states:
            s.rates(x0, mu0, tracer, T, t, nucleus_update)

        # step size from the growing classes and the matrix depletion they cause
        dt_new = 1.2 * dt
        # shrinking classes whose solute content together stays within the matrix step limit
        # may dissolve inside a step; following each to the end would cost steps without
        # accuracy, since its rate diverges as it vanishes. The budget is shared, between
        # populations too, or a whole population shrinking at once (on heating) would vanish
        # in one step.
        content = np.concatenate([s.volumes() * np.abs(s.cls["comp"] - x0).max(axis=-1) for s in states])
        rate = np.concatenate([s.rate for s in states])
        free = np.zeros(rate.size, dtype=bool)
        shrinking = np.flatnonzero(rate < 0)
        order = shrinking[np.argsort(content[shrinking])]
        free[order[np.cumsum(content[order]) < max_matrix_change]] = True
        dx0 = np.zeros_like(x0)
        start = 0
        for s in states:
            number, radius, comp = s.cls["number"], s.cls["radius"], s.cls["comp"]
            limiting = (radius > 2 * smallest_radius) & ~free[start:start + radius.size]
            start += radius.size
            if np.any(limiting):
                dt_new = min(dt_new, max_radius_change
                             * np.min(radius[limiting] / np.maximum(np.abs(s.rate[limiting]), 1e-300)))
                if s.evolving:
                    dt_new = min(dt_new, max_composition_change / max(np.abs(s.dcomp[limiting]).max(), 1e-300))
            if radius.size:
                dvol = number * 4 * np.pi * radius**2 * np.where(limiting | (s.rate > 0), s.rate, 0.0)
                dx0 += -(dvol @ (comp - x0)) / (1 - phi)
                if s.evolving:
                    dx0 -= (number * 4 * np.pi / 3 * radius**3) @ np.where(limiting[:, None], s.dcomp, 0.0) / (1 - phi)
        if np.abs(dx0).max() > 0:
            dt_new = min(dt_new, max_matrix_change / np.abs(dx0).max())
        # Near a stationary state a step longer than the matrix's relaxation time overshoots, and
        # the matrix then swings about the fixed point by up to max_matrix_change, which for large
        # particles exceeds the capillary supersaturation that drives them. The matrix rate
        # reversing between steps marks the overshoot. Reversals far below the step limit are
        # left alone, or rounding noise at the fixed point would hold the step near zero.
        if (previous_dx0 is not None and dx0 @ previous_dx0 < 0
                and np.abs(dx0).max() * dt > 1e-3 * max_matrix_change):
            dt_new = min(dt_new, 0.5 * dt)
        previous_dx0 = dx0
        nucleating = [s for s in states if s.j > 0]
        if nucleating:
            new_vol = sum(s.j * 4 * np.pi / 3 * s.rho_star**3 * np.abs(s.nucleus_x - x0).max() for s in nucleating)
            dt_new = min(dt_new, max_matrix_change / max(new_vol, 1e-300))
        dt = min(dt_new, save[next_save] - t) if save[next_save] > t else dt_new
        dt = max(dt, 1e-12)
        if varying:
            for _ in range(60):
                if abs(temp(t + dt) - T) <= max_temperature_change:
                    break
                dt *= 0.5

        for s in states:
            s.advance(dt, smallest_radius, min_class_number, max_classes, min_classes)
        t += dt
        steps += 1

        while next_save < save.size and t >= save[next_save] * (1 - 1e-9):
            _, x0, vols = matrix_state()
            out_x0.append(x0)
            for s, vol, o in zip(states, vols, out):
                phi_k = vol.sum()
                total = s.cls["number"].sum()
                o["t"].append(t)
                o["n"].append(total)
                o["r"].append((s.cls["number"] @ s.cls["radius"]) / total if total > 0 else 0.0)
                o["phi"].append(phi_k)
                o["xp"].append((vol @ s.cls["comp"]) / phi_k if phi_k > 0 else np.full(n_el, np.nan))
                o["classes"].append({"number": s.cls["number"].copy(), "radius": s.cls["radius"].copy(),
                                     "composition": s.cls["comp"].copy()})
            next_save += 1

    return [MulticomponentResult(np.array(o["t"]), np.array(o["r"]), np.array(o["n"]), np.array(o["phi"]),
                                 np.array(out_x0), np.array(o["xp"]), o["classes"], steps) for o in out]


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
