"""Precipitation, dissolution and coarsening along a heat treatment, after Myhr and Grong, Acta Mater. 48 (2000) 1605.

A dilute alloy holds spherical particles of one solute-rich phase in fixed size classes. Each step:

    j = j0 exp[-(A0 / RT)^3 / ln(C / C_e)^2] exp(-Q_d / RT)        nucleation, eq. (3)
    v = dr/dt = (C - C_i) / (C_p - C_i) D / r                       growth or dissolution, eq. (4)
    C_i = C_e exp(2 sigma V_m / (r R T)),  C_e = C_s exp(-Q_s / RT)  Gibbs-Thomson and solvus, eqs. (5), (17)
    C = C_0 - (C_p - C) sum_i (4/3) pi r_i^3 N_i                     solute balance, eq. (11)

with D = D_0 exp(-Q_d / RT). The distribution N (particles per volume per radius) moves through the classes by
the implicit upwind control-volume scheme of eqs. (12)-(16) and Table 1, with the wall velocities taken from the
matrix content of the previous step; nuclei enter at r* (1 + 0.05), r* = 2 sigma V_m / (RT ln(C / C_e)) (eq. 6).
The first wall sits at half the first radius, particles crossing it have dissolved, and none may reach the last
class. Concentrations are in any one unit (the paper's wt%), and f = (C_0 - C) / (C_p - C) is the volume fraction.

Time steps are this package's: a step is repeated at half length when it moves ln C by more than ``tolerance``
times ln(C / C_e) (or, near the solvus, times ``1e-3``), lengthened by a quarter after an easy one, and kept below
one class width of travel for growing classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .schedule import Temperature, temperature_at
from .thermo import R


@dataclass(frozen=True)
class HeatTreatableAlloy:
    """The inputs of Myhr and Grong's Table 2: solute content of alloy ``c0`` and precipitate ``cp``, solvus
    prefactor ``cs`` and enthalpy ``qs`` (J/mol, eq. 17), diffusivity prefactor ``d0`` (m^2/s) and activation
    energy ``qd`` (J/mol), nucleation prefactor ``j0`` (1/m^3 s) and barrier parameter ``a0`` (J/mol, eq. 2),
    interfacial energy ``gamma`` (J/m^2) and precipitate molar volume ``molar_volume`` (m^3/mol)."""

    c0: float
    cp: float
    cs: float
    qs: float
    d0: float
    qd: float
    j0: float
    a0: float
    gamma: float
    molar_volume: float
    growth_margin: float = 0.05

    def __post_init__(self):
        if not 0 < self.c0 < self.cp:
            raise ValueError("need 0 < c0 < cp")
        if min(self.cs, self.d0, self.j0, self.a0, self.gamma, self.molar_volume) <= 0 or self.growth_margin < 0:
            raise ValueError("cs, d0, j0, a0, gamma and the molar volume must be positive")

    def solubility(self, T):
        """C_e, eq. (17)."""
        return self.cs * np.exp(-self.qs / (R * np.asarray(T, dtype=float)))

    def diffusivity(self, T):
        return self.d0 * np.exp(-self.qd / (R * np.asarray(T, dtype=float)))

    def equilibrium_fraction(self, T):
        """Lever-rule volume fraction (C_0 - C_e) / (C_p - C_e), zero above the solvus."""
        ce = self.solubility(T)
        return np.maximum((self.c0 - ce) / (self.cp - ce), 0.0)

    def critical_radius(self, c: float, T: float) -> float:
        """r* of eq. (6); infinite when the matrix is not supersaturated."""
        s = np.log(c / self.solubility(T))
        return 2 * self.gamma * self.molar_volume / (R * T * s) if s > 0 else np.inf

    def nucleation_rate(self, c: float, T: float) -> float:
        """j of eq. (3), zero when the matrix is not supersaturated."""
        s = np.log(c / self.solubility(T))
        if s <= 0:
            return 0.0
        return float(self.j0 * np.exp(-((self.a0 / (R * T)) ** 3) / s**2 - self.qd / (R * T)))

    def interface_content(self, radius, T):
        """C_i of eq. (5)."""
        arg = 2 * self.gamma * self.molar_volume / (np.asarray(radius, dtype=float) * R * T)
        return self.solubility(T) * np.exp(np.minimum(arg, 700.0))

    def growth_rate(self, radius, c: float, T: float):
        """dr/dt of eqs. (4)-(5). A particle so small that C_i reaches C_p, below
        2 sigma V_m / (RT ln(C_p / C_e)), has no meaning in eq. (4), whose sign would flip; it dissolves at once."""
        radius = np.asarray(radius, dtype=float)
        ci = self.interface_content(radius, T)
        stable = ci < self.cp
        ratio = np.where(stable, (c - ci) / np.where(stable, self.cp - ci, 1.0), -1e12)
        return ratio * self.diffusivity(T) / radius


def aa6082(gamma: float = 0.2) -> HeatTreatableAlloy:
    """Myhr and Grong's Table 2: Mg in an AA 6082 alloy precipitating as Mg2Si, in wt% Mg."""
    return HeatTreatableAlloy(c0=0.63, cp=63.4, cs=970.0, qs=47175.0, d0=2.2e-4, qd=130000.0, j0=9.66e34,
                              a0=16220.0, gamma=gamma, molar_volume=3.95e-5)


@dataclass
class HeatTreatmentResult:
    """``distribution[k]`` is N (1/m^4) at ``times[k]`` on the class radii ``radii`` with widths ``widths``."""

    radii: np.ndarray
    widths: np.ndarray
    times: np.ndarray
    temperature: np.ndarray
    distribution: np.ndarray
    matrix_content: np.ndarray
    nucleation: np.ndarray
    critical_radius: np.ndarray
    steps: int

    @property
    def number_density(self) -> np.ndarray:
        return self.distribution @ self.widths

    @property
    def mean_radius(self) -> np.ndarray:
        n = self.number_density
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, self.distribution @ (self.widths * self.radii) / n, np.nan)

    @property
    def volume_fraction(self) -> np.ndarray:
        return self.distribution @ (self.widths * 4 * np.pi / 3 * self.radii**3)

    def state(self, k: int = -1) -> np.ndarray:
        """The distribution at save ``k``, to start another treatment from."""
        return self.distribution[k].copy()


def uniform_radii(r_max: float, classes: int = 1000) -> np.ndarray:
    """Class radii i dr, i = 1..classes, as in the paper (first wall at dr / 2)."""
    return r_max / classes * np.arange(1, classes + 1)


def stretched_radii(step: float, uniform_until: float, ratio: float, r_max: float) -> np.ndarray:
    """Radii ``step`` apart up to ``uniform_until``, then each ``ratio`` times the last up to ``r_max``: fine
    classes where particles nucleate and dissolve, few where they have coarsened."""
    if not 0 < step < uniform_until < r_max or ratio <= 1:
        raise ValueError("need 0 < step < uniform_until < r_max and ratio > 1")
    fine = np.arange(1, int(round(uniform_until / step)) + 1) * step
    n = int(np.ceil(np.log(r_max / fine[-1]) / np.log(ratio)))
    return np.concatenate([fine, fine[-1] * ratio ** np.arange(1, n + 1)])


def _widths(radii):
    r = np.asarray(radii, dtype=float)
    return np.diff(np.concatenate([[r[0] / 2], (r[1:] + r[:-1]) / 2, [r[-1] + (r[-1] - r[-2]) / 2]]))


def _scaled(radii, shape, volume_fraction):
    r = np.asarray(radii, dtype=float)
    return shape * volume_fraction / (shape @ (_widths(r) * 4 * np.pi / 3 * r**3))


def lsw_distribution(radii, mean_radius: float, volume_fraction: float) -> np.ndarray:
    """The Lifshitz-Slyozov-Wagner distribution N (1/m^4) with the given mean radius, on the class radii, scaled
    to ``volume_fraction``."""
    u = np.asarray(radii, dtype=float) / mean_radius
    x = np.minimum(u, 1.5 - 1e-12)
    g = np.where(u < 1.5, x**2 * (3 / (3 + x)) ** (7 / 3) * (1.5 / (1.5 - x)) ** (11 / 3) * np.exp(-x / (1.5 - x)), 0.0)
    return _scaled(radii, g, volume_fraction)


def log_normal_distribution(radii, mean_radius: float, volume_fraction: float, spread: float) -> np.ndarray:
    """Myhr and Grong's eq. (18), normal in ln r^2 with standard deviation ``spread`` (their b), placed so that the
    number-mean radius is ``mean_radius`` (median exp(-b^2 / 8) below it) and scaled to ``volume_fraction``."""
    r = np.asarray(radii, dtype=float)
    median = mean_radius * np.exp(-spread**2 / 8)
    shape = np.exp(-((np.log(r**2) - np.log(median**2)) ** 2) / (2 * spread**2)) / r  # d ln r^2 = 2 dr / r
    return _scaled(r, shape, volume_fraction)


def solve_heat_treatment(alloy: HeatTreatableAlloy, temperature: Temperature, t_end: float, radii=None,
                         save_times=None, initial=None, dt: float = 1e-3, tolerance: float = 0.02,
                         courant: float = 1.0, max_steps: int = 2_000_000) -> HeatTreatmentResult:
    """Run the Myhr-Grong model from t = 0 to ``t_end`` (s) along ``temperature`` (K, constant, schedule or
    callable). ``radii`` are the class radii (default :func:`uniform_radii` up to 20 nm); ``initial`` a starting
    distribution N on them (1/m^4), for instance :meth:`HeatTreatmentResult.state` of an earlier treatment.
    Steps are also kept below ``courant`` class widths of travel for growing classes: the implicit upwind
    scheme is stable at any step, but longer steps smear the distribution over many classes.
    Raises RuntimeError if particles reach the largest class."""
    r = uniform_radii(20e-9) if radii is None else np.asarray(radii, dtype=float)
    if r.ndim != 1 or r.size < 3 or r[0] <= 0 or np.any(np.diff(r) <= 0):
        raise ValueError("class radii must be positive and increasing")
    if t_end <= 0 or dt <= 0 or not 0 < tolerance < 1 or courant <= 0:
        raise ValueError("need t_end > 0, dt > 0, 0 < tolerance < 1 and courant > 0")
    width = _widths(r)
    walls = np.concatenate([[r[0] / 2], r[0] / 2 + np.cumsum(width)])
    volume = 4 * np.pi / 3 * r**3 * width
    n = np.zeros_like(r) if initial is None else np.array(initial, dtype=float)
    if n.shape != r.shape or np.any(n < 0):
        raise ValueError("the initial distribution needs one non-negative value per class")
    saves = np.append([] if save_times is None else np.asarray(save_times, dtype=float), t_end)
    if saves.min() < 0 or saves.max() > t_end * (1 + 1e-9):
        raise ValueError("save times must lie in [0, t_end]")
    saves = np.unique(np.minimum(saves, t_end))

    def content(dist):
        v = dist @ volume
        if v >= 1:
            raise RuntimeError("the particles fill the volume; the distribution is not physical")
        return (alloy.c0 - alloy.cp * v) / (1 - v)

    c = content(n)
    if c <= 0:
        raise ValueError("the initial distribution holds more solute than the alloy")

    def advance(dist, c_old, T, h):
        v_wall = alloy.growth_rate(walls, c_old, T)
        ve, vw = v_wall[1:], v_wall[:-1]
        ap = width / h + np.maximum(ve, 0) - np.minimum(vw, 0)
        ab = np.zeros((3, r.size))
        ab[1] = ap
        ab[0, 1:] = np.minimum(ve[:-1], 0)  # -a_E above the diagonal; the last class has no east neighbour
        ab[2, :-1] = -np.maximum(vw[1:], 0)  # -a_W below; nothing enters the first class from below
        rhs = width / h * dist
        j = alloy.nucleation_rate(c_old, T)
        rc = alloy.critical_radius(c_old, T)
        if j > 0:
            k = min(np.searchsorted(walls, rc * (1 + alloy.growth_margin)) - 1, r.size - 1)
            rhs[max(k, 0)] += j  # the equation is per unit time: width / h (N - N0) = fluxes + j
        new = np.maximum(solve_banded((1, 1), ab, rhs), 0.0)
        # less than one particle per cubic metre in a class is nothing; left in, round-off from a dissolved
        # population would go on growing and set the mean radius
        new[new * width < 1.0] = 0.0
        return new, j, rc

    times, temps, dists, contents, rates, crit = [], [], [], [], [], []

    def save(t, dist, cc):
        T = temperature_at(temperature, t)
        times.append(t), temps.append(T), dists.append(dist.copy()), contents.append(cc)
        rates.append(alloy.nucleation_rate(cc, T)), crit.append(alloy.critical_radius(cc, T))

    t, h, steps, k_save = 0.0, dt, 0, 0
    while saves[k_save] <= 0:
        save(0.0, n, c)
        k_save += 1
        if k_save == saves.size:
            break
    while k_save < saves.size:
        T = temperature_at(temperature, t + min(h, saves[k_save] - t))
        # only growing classes limit the step: shrinking ones move fastest just above the size where eq. (4)
        # breaks down and would set the step for the whole run, with no gain in accuracy (0.3 % at most)
        speed = np.maximum(alloy.growth_rate(r[n > 0], c, T), 0) / width[n > 0]
        fastest = speed.max() if speed.size else 0.0
        h_try = min(h, saves[k_save] - t, courant / fastest if fastest > 0 else np.inf)
        T = temperature_at(temperature, t + h_try)
        new, _, _ = advance(n, c, T, h_try)
        c_new = content(new)
        ce = alloy.solubility(T)
        scale = max(abs(np.log(c / ce)), 1e-3)
        change = abs(np.log(c_new / c)) / scale if c_new > 0 else np.inf
        if change > tolerance and h_try > 1e-12:
            h = h_try / 2
            continue
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"no convergence within {max_steps} steps")
        t, n, c = t + h_try, new, c_new
        edge = n[-2:] @ volume[-2:]
        if edge > 1e-15 and edge > 1e-9 * (n @ volume):
            raise RuntimeError("particles reached the largest class; extend the radius grid")
        if change < tolerance / 4 and h_try == h:
            h *= 1.25
        if t >= saves[k_save] * (1 - 1e-12):
            save(t, n, c)
            k_save += 1
    return HeatTreatmentResult(r, width, np.array(times), np.array(temps), np.array(dists), np.array(contents),
                               np.array(rates), np.array(crit), steps)
