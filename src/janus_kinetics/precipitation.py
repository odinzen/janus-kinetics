"""Precipitation kinetics: classical nucleation, diffusion-controlled growth and coarsening.

Equations from Perez, Dumont and Acevedo-Reyes, Acta Mater. 56 (2008) 2119, with
eqs. (4), (7) and the multicomponent condensation rate as corrected in their
corrigendum, Acta Mater. 57 (2009) 1318. The size distribution is held in fixed
size classes and evolves by the class-to-class fluxes of their "Euler-like"
approach (eq. 24), the Kampmann-Wagner numerical scheme. Nucleation, growth and
coarsening all follow from the nucleation rate (eq. 6) and the growth rate with
the Gibbs-Thomson interface composition (eqs. 9-10); no separate coarsening law
is imposed.

Their "mean radius approach" (section 3.1) is here too, as :func:`solve_mean_radius`.

Scope: spherical precipitates of fixed composition in a binary matrix, isothermal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp

K_B = 1.380649e-23  # J/K


@dataclass(frozen=True)
class DiluteSolution:
    """Matrix/precipitate thermodynamics of a dilute binary, Perez et al. (2008) eq. (9).

    ``x_eq`` is the solute fraction of the matrix in equilibrium with a flat
    precipitate interface, X^i(inf); ``x_p`` the solute fraction of the
    precipitate, X^P = y/(x+y) for A_xB_y; ``v_at_p`` and ``v_at_m`` the mean
    atomic volumes of precipitate and matrix (m^3); ``gamma`` the interface
    energy (J/m^2).

    Eq. (9) written per atom of precipitate reads
        (1 - X^P) ln X_A^i(R) + X^P ln X_B^i(R) = (1 - X^P) ln X_A^i(inf) + X^P ln X_B^i(inf) + R0/R
    with R0 = 2 gamma v_at^P / kT, the length of their eq. (11). The driving
    force below is the one consistent with it: a nucleus of the critical radius
    R* = -2 gamma / dg (from dG/dR = 0 in eq. 1) has exactly the matrix
    composition at its interface.
    """

    x_eq: float
    x_p: float
    v_at_p: float
    v_at_m: float
    gamma: float
    T: float

    def __post_init__(self):
        if not 0.0 < self.x_eq < self.x_p <= 1.0:
            raise ValueError(f"need 0 < x_eq < x_p <= 1 as mole fractions, got x_eq={self.x_eq}, x_p={self.x_p}")
        for name in ("v_at_p", "v_at_m"):
            value = getattr(self, name)
            if not 1e-31 < value < 1e-27:
                raise ValueError(f"{name} = {value:g} m^3 is not an atomic volume (about 1e-29 m^3; "
                                 "a molar volume must be divided by Avogadro's number)")
        if self.gamma <= 0:
            raise ValueError(f"interface energy must be positive, got {self.gamma}")
        if self.T <= 0:
            raise ValueError(f"temperature must be in kelvin, got {self.T}")

    @property
    def alpha(self) -> float:
        """v_at^M / v_at^P, the ratio in Perez et al. eq. (8)."""
        return self.v_at_m / self.v_at_p

    @property
    def capillary_length(self) -> float:
        """R0 = 2 gamma v_at^P / kT in m."""
        return 2.0 * self.gamma * self.v_at_p / (K_B * self.T)

    def _potential(self, x):
        x = np.asarray(x, dtype=float)
        return (1.0 - self.x_p) * np.log1p(-x) + self.x_p * np.log(x)

    def driving_force(self, x) -> np.ndarray:
        """Chemical driving force per unit precipitate volume, J/m^3; negative when supersaturated."""
        return -(K_B * self.T / self.v_at_p) * (self._potential(x) - self._potential(self.x_eq))

    def critical_radius(self, x) -> np.ndarray:
        """R* = -2 gamma / dg; infinite where the matrix is not supersaturated."""
        dg = self.driving_force(x)
        with np.errstate(divide="ignore"):
            return np.where(dg < 0, -2.0 * self.gamma / np.where(dg < 0, dg, -1.0), np.inf)

    @property
    def smallest_radius(self) -> float:
        """Radius below which eq. (9) has no matrix composition below X^P.

        The left side of eq. (9) peaks at X = X^P, so a particle smaller than
        R0 / (pot(X^P) - pot(X^i)) has no equilibrium interface composition.
        """
        gap = float(self._potential(self.x_p) - self._potential(self.x_eq)) if self.x_p < 1.0 else np.inf
        return self.capillary_length / gap if gap > 0 else np.inf

    def interface_composition(self, radius) -> np.ndarray:
        """X^i(R), the matrix solute fraction in equilibrium with a particle of radius R (eq. 9).

        NaN where R is below :attr:`smallest_radius` and no such composition exists.
        """
        radius = np.asarray(radius, dtype=float)
        target = self._potential(self.x_eq) + self.capillary_length / radius
        if self.x_p == 1.0:
            return np.exp(target)
        # bisection in ln x on (0, X^P), where the potential rises monotonically
        lo = np.full_like(radius, np.log(1e-300))
        hi = np.full_like(radius, np.log(self.x_p))
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            below = self._potential(np.exp(mid)) < target
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        x = np.exp(0.5 * (lo + hi))
        return np.where(target <= self._potential(self.x_p), x, np.nan)


def solubility_from_pycalphad(db, components, matrix: str, precipitate: str, solute: str, T: float,
                              x_overall: float, pressure: float = 101325.0) -> tuple[float, float]:
    """(X^i(inf), X^P): solute fractions of matrix and precipitate in two-phase equilibrium.

    From :func:`pycalphad.equilibrium` with only the two phases allowed, at an
    overall solute fraction ``x_overall`` inside the two-phase field; this is the
    thermodynamic input that eqs. (9)-(10) of Perez et al. (2008) take as X^i and X^P.
    """
    from pycalphad import equilibrium, variables as v

    comps = list(components) + (["VA"] if "VA" not in components else [])
    eq = equilibrium(db, comps, [matrix, precipitate], {v.T: T, v.P: pressure, v.N: 1, v.X(solute): x_overall})
    phases = [str(p) for p in np.squeeze(eq.Phase.values)]
    if matrix not in phases or precipitate not in phases:
        raise ValueError(f"x_overall = {x_overall} is not in the {matrix} + {precipitate} field at {T} K (found {phases})")
    x = np.squeeze(eq.X.sel(component=solute).values)
    return float(x[phases.index(matrix)]), float(x[phases.index(precipitate)])


@dataclass(frozen=True)
class Kinetics:
    """Transport and nucleation inputs: solute diffusivity in the matrix (m^2/s),
    lattice parameter ``a`` (m) of the condensation rate, and the number of
    nucleation sites per unit volume N0 (m^-3) of eq. (3).

    The diffusivity can come from a mobility database as D* = RT M of the solute
    (Andersson and Agren, J. Appl. Phys. 72 (1992) 1350, eq. (61)), for example
    ``R * T * phase.mobility(x, T)[solute_index]``.
    """

    diffusivity: float
    lattice_parameter: float
    sites: float

    def __post_init__(self):
        if self.diffusivity < 0:
            raise ValueError(f"diffusivity must not be negative, got {self.diffusivity}")
        if not 1e-11 < self.lattice_parameter < 1e-8:
            raise ValueError(f"lattice parameter {self.lattice_parameter:g} m is not a lattice parameter "
                             "(about 4e-10 m; give it in metres, not angstrom)")
        if self.sites < 0:
            raise ValueError(f"number of nucleation sites must not be negative, got {self.sites}")


def growth_rate(thermo: DiluteSolution, kin: Kinetics, x, radius) -> np.ndarray:
    """dR/dt = (D/R) (X - X^i(R)) / (alpha X^P - X^i(R)), Perez et al. eq. (10)."""
    radius = np.asarray(radius, dtype=float)
    xi = thermo.interface_composition(radius)
    return kin.diffusivity / radius * (x - xi) / (thermo.alpha * thermo.x_p - xi)


def nucleation_rate(thermo: DiluteSolution, kin: Kinetics, x: float, t: float) -> tuple[float, float]:
    """(dN/dt in m^-3 s^-1, nucleation radius R*_kT in m) at matrix fraction x and time t.

    Eq. (6) with the Zeldovich factor Z = v_at^P / (2 pi R*^2) sqrt(gamma/kT)
    (corrected eq. 4), beta* = 4 pi R*^2 D X / a^4 (eq. 5), tau = 4 / (2 pi beta* Z^2),
    and R*_kT = R* + (1/2) sqrt(kT / (pi gamma)) (corrected eq. 7).
    """
    dg = float(thermo.driving_force(x))
    if dg >= 0.0 or t <= 0.0:
        return 0.0, np.inf
    kt = K_B * thermo.T
    g = thermo.gamma
    r_star = -2.0 * g / dg
    barrier = 16.0 * np.pi * g**3 / (3.0 * dg**2)
    zeldovich = thermo.v_at_p / (2.0 * np.pi * r_star**2) * np.sqrt(g / kt)
    beta = 4.0 * np.pi * r_star**2 * kin.diffusivity * x / kin.lattice_parameter**4
    tau = 4.0 / (2.0 * np.pi * beta * zeldovich**2)
    rate = kin.sites * zeldovich * beta * np.exp(-barrier / kt) * np.exp(-tau / t)
    return rate, r_star + 0.5 * np.sqrt(kt / (np.pi * g))


@dataclass
class PrecipitationResult:
    times: np.ndarray
    radius_edges: np.ndarray
    number: np.ndarray  # precipitates per m^3 in each class [t, class]
    matrix_fraction: np.ndarray  # X
    critical_radius: np.ndarray
    steps: int = 0
    rejected: int = 0

    @property
    def radius(self) -> np.ndarray:
        return 0.5 * (self.radius_edges[1:] + self.radius_edges[:-1])

    @property
    def number_density(self) -> np.ndarray:
        """N = sum_i N_i (eq. 19)."""
        return self.number.sum(axis=1)

    @property
    def mean_radius(self) -> np.ndarray:
        """R_mean = sum R_i N_i / sum N_i (eq. 20); NaN before any precipitate exists."""
        n = self.number_density
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.number @ self.radius) / n

    @property
    def volume_fraction(self) -> np.ndarray:
        """f_T = (4/3) pi sum_i N_i R_i^3 (eq. 21)."""
        return 4.0 / 3.0 * np.pi * (self.number @ self.radius**3)


def solve_kwn(
    thermo: DiluteSolution,
    kin: Kinetics,
    x0: float,
    t_end: float,
    radius_edges: np.ndarray,
    number0: np.ndarray | None = None,
    save_times=None,
    dt0: float = 1e-3,
    max_change: float = 0.01,
    growth: float = 1.1,
) -> PrecipitationResult:
    """Evolve the precipitate size distribution from an alloy of solute fraction ``x0``.

    The class fluxes of eq. (24) are taken implicitly (backward Euler), so a step
    is not bound by the explicit limit of eq. (25). The step grows by ``growth``
    after each step and is halved and retried when the critical radius changes by
    more than ``max_change`` (relative), the rule Perez et al. give for their
    Lagrange-like approach (section 3.4.1). The same limit on the matrix fraction
    and on the mean radius is this package's addition; without the latter a dilute
    population, which barely changes the matrix, would take steps so long that the
    implicit fluxes smear the distribution. Particles leaving the smallest
    class through its lower edge have dissolved; their solute returns to the
    matrix through the mass balance of eq. (22). The lower edge must lie above the
    radius where X^i(R) reaches alpha X^P, below which eq. (10) does not hold.
    """
    edges = np.asarray(radius_edges, dtype=float)
    if np.any(np.diff(edges) <= 0) or edges[0] <= 0:
        raise ValueError("radius_edges must be positive and increasing")
    xi0 = float(thermo.interface_composition(edges[0]))
    if not xi0 < thermo.alpha * thermo.x_p:
        cutoff = max(thermo.smallest_radius, thermo.capillary_length / max(
            float(thermo._potential(min(thermo.alpha * thermo.x_p, 1 - 1e-12)) - thermo._potential(thermo.x_eq)), 1e-300))
        raise ValueError(
            f"the lowest class edge {edges[0]:.3g} m is below the cut-off radius {cutoff:.3g} m of eq. (10) "
            f"(capillary length 2 gamma v/kT = {thermo.capillary_length:.3g} m); start the classes above it, "
            "and check the units of gamma, the atomic volumes and T if the cut-off looks unphysical")
    width = np.diff(edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    n_cls = len(centers)
    number = np.zeros(n_cls) if number0 is None else np.asarray(number0, dtype=float).copy()
    a_xp = thermo.alpha * thermo.x_p

    def matrix_fraction(num):
        f = 4.0 / 3.0 * np.pi * (num @ centers**3)
        return (x0 - a_xp * f) / (1.0 - thermo.alpha * f)  # eq. (22) solved for X

    def critical(x):
        return float(thermo.critical_radius(x))

    def mean_radius(num):
        total = num.sum()
        return num @ centers / total if total > 0 else np.nan

    def advance(num, x, t_new, dt):
        v = growth_rate(thermo, kin, x, edges)  # at every class edge
        out_up = np.maximum(v[1:], 0.0) / width  # leaves class i upward through edge i+1
        out_down = np.maximum(-v[:-1], 0.0) / width  # leaves class i downward through edge i
        diag = 1.0 + dt * (out_up + out_down)
        lower = -dt * out_up[:-1]  # class i+1 gains from class i
        upper = -dt * out_down[1:]  # class i gains from class i+1
        mat = sp.diags([lower, diag, upper], [-1, 0, 1], format="csc")
        rhs = num.copy()
        rate, r_nuc = nucleation_rate(thermo, kin, x, t_new)
        if rate > 0.0:
            k = np.searchsorted(edges, r_nuc) - 1
            if not 0 <= k < n_cls:
                raise ValueError(f"nucleation radius {r_nuc:.3g} m lies outside radius_edges")
            rhs[k] += rate * dt  # eq. (23)
        new = np.maximum(spla.spsolve(mat, rhs), 0.0)
        lost = dt * out_up[-1] * new[-1]  # left through the top edge
        if lost > 1e-6 * max(new.sum(), 1e-300):
            raise ValueError("the size distribution reached the largest class; extend radius_edges")
        return new

    targets = np.unique(np.append(np.asarray(save_times if save_times is not None else [], dtype=float), t_end))
    x = matrix_fraction(number)
    times, saved, xs, rstar = [0.0], [number.copy()], [x], [critical(x)]
    t, dt, steps, rejected = 0.0, dt0, 0, 0
    for target in targets:
        while t < target * (1 - 1e-12):
            step = min(dt, target - t)
            new = advance(number, x, t + step, step)
            x_new = matrix_fraction(new)
            r_old, r_new = critical(x), critical(x_new)
            ok = 0.0 < x_new < 1.0 and abs(x_new - x) <= max_change * x
            if np.isfinite(r_old) and np.isfinite(r_new):
                ok = ok and abs(r_new - r_old) <= max_change * r_old
            m_old, m_new = mean_radius(number), mean_radius(new)
            if np.isfinite(m_old) and np.isfinite(m_new):
                ok = ok and abs(m_new - m_old) <= max_change * m_old
            if not ok and step > 1e-12 * max(t, 1.0):
                dt = 0.5 * step
                rejected += 1
                continue
            number, x, t = new, x_new, t + step
            steps += 1
            dt = max(dt, step) * growth if step < dt else step * growth
        times.append(t)
        saved.append(number.copy())
        xs.append(x)
        rstar.append(critical(x))
    return PrecipitationResult(np.array(times), edges, np.array(saved), np.array(xs), np.array(rstar), steps, rejected)


def coarsening_fraction(radius, critical) -> np.ndarray:
    """f_coars = 1 - 1000 (R / R* - 1)^2 for 0.99 R* < R < 1.01 R*, Perez et al. eq. (17), and 0 outside, where the
    paper leaves it undefined. The function drops from 0.9 to 0 at both edges of the band; it is used as printed."""
    u = np.asarray(radius, dtype=float) / np.asarray(critical, dtype=float) - 1.0
    return np.where(np.abs(u) < 0.01, 1.0 - 1000.0 * u**2, 0.0)


@dataclass
class MeanRadiusResult:
    times: np.ndarray
    mean_radius: np.ndarray
    number_density: np.ndarray
    matrix_fraction: np.ndarray
    critical_radius: np.ndarray

    @property
    def volume_fraction(self) -> np.ndarray:
        return 4.0 / 3.0 * np.pi * self.number_density * self.mean_radius**3


def solve_mean_radius(thermo: DiluteSolution, kin: Kinetics, x0: float, t_end: float, number0: float = 0.0,
                      radius0: float | None = None, save_times=None, rtol: float = 1e-6) -> MeanRadiusResult:
    """Number density N and mean radius R of the precipitates alone, the "mean radius approach" of Perez et al.
    (section 3.1). New nuclei of radius R*_kT join at the nucleation rate of eq. (6), pulling R toward their size
    (eq. 12); growth is eq. (10) at R. Coarsening is the LSW rate of eq. (11), d R / dt = (4/27) X^i / (alpha X^P -
    X^i) R0 D / R^2 (R0 over X^P for a compound, as the paper notes), and eq. (14) for N; the two are blended by
    the coarsening fraction of eq. (17), R by eq. (15) and N by eq. (18). The matrix fraction X follows from the mass
    balance, eq. (13). Eq. (18) compares the coarsening and nucleation rates of N, of opposite signs; it is read
    here as comparing their sizes, -dN/dt|coars against dN/dt|nucl. Integrated by LSODA, the paper's adaptive
    Runge-Kutta being unsuited to the stiff start, when 1/N dN/dt is large.

    Starting from ``number0`` particles per m^3 of radius ``radius0`` (none by default)."""
    if number0 > 0 and (radius0 is None or radius0 <= 0):
        raise ValueError("an initial population needs a positive radius0")
    a_xp = thermo.alpha * thermo.x_p
    xi = thermo.x_eq
    r0 = thermo.capillary_length / thermo.x_p
    lsw = 4.0 / 27.0 * xi / (a_xp - xi) * r0 * kin.diffusivity

    def matrix(n, r):
        f = 4.0 / 3.0 * np.pi * r**3 * n
        return (x0 - a_xp * f) / (1.0 - thermo.alpha * f)

    start_r = radius0 if number0 > 0 else nucleation_rate(thermo, kin, x0, 1.0)[1]
    if not np.isfinite(start_r):
        raise ValueError("the alloy is not supersaturated and holds no precipitates to start from")

    @np.errstate(invalid="ignore", divide="ignore")  # trial states of the integrator may leave the matrix range
    def rhs(t, y):
        n, r = max(y[0], 0.0), y[1]
        x = matrix(n, r)
        rate, r_kt = nucleation_rate(thermo, kin, x, t)
        if n <= 0.0:
            return [rate, 0.0]
        grow = float(growth_rate(thermo, kin, x, r))
        if rate > 0.0:
            grow += rate / n * (r_kt - r)
        f = float(coarsening_fraction(r, thermo.critical_radius(x)))
        if f == 0.0:
            return [rate, grow]
        coars_r = lsw / r**2
        coars_n = lsw / r**3 * (r0 * xi / (r * (thermo.x_p - xi)) * (3.0 / (4.0 * np.pi * r**3) - n) - 3.0 * n)
        dn = f * coars_n if -coars_n > rate else rate
        return [dn, f * coars_r + (1.0 - f) * grow]

    targets = np.unique(np.append(np.asarray(save_times if save_times is not None else [], dtype=float), t_end))
    sol = solve_ivp(rhs, (0.0, t_end), [float(number0), float(start_r)], method="LSODA", t_eval=np.concatenate([[0.0], targets]),
                    rtol=rtol, atol=[1e-6, 1e-15])
    if not sol.success:
        raise RuntimeError(f"mean radius integration failed: {sol.message}")
    n, r = sol.y
    x = matrix(n, r)
    return MeanRadiusResult(sol.t, r, n, x, thermo.critical_radius(x))
