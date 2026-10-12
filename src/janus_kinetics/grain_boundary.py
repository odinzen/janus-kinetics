"""Diffusion along an isolated grain boundary from a surface source, after Fisher, J. Appl. Phys. 22 (1951) 74.

The boundary is a slab of thickness delta and diffusivity D' normal to a free
surface held at concentration 1, in a solid of diffusivity D << D'. Inside the
slab C depends on depth y only and loses material to both sides (eq. 3); outside,
diffusion runs mainly normal to the slab (eq. 4 with the y term dropped):

    dC/dt = D' d2C/dy2 + (2/delta) D (dC/dx)_0          (slab)
    dC/dt = D d2C/dx2                                  (grains)

With t1 = D t / delta^2, x1 = x / delta and y1 = y / (delta (D'/D)^(1/2)) (eq. 5)
these become parameter-free (eqs. 3a and 4a). Fisher's approximate solution is
eq. (8), C = exp(-2^(1/2) y1 / (pi^(1/4) t1^(1/4))) erfc(x1 / (2 t1^(1/2))); the
logarithm of the layer-averaged concentration then falls linearly with depth,
with slope eq. (14). Eq. (8) assumes the grains beside the boundary have seen
its present concentration for the whole time, so it overestimates penetration
at depth: at t1 = 100 and y1 = 8 the exact solution of eqs. (3a)-(4a) gives
C = 0.028 at the boundary where eq. (8) gives 0.068. :func:`solve_fisher` solves
those equations numerically and is the one to use for numbers. This is the regime of isolated boundaries, where each
boundary's diffusion field does not reach the next; it suits fine-grained films
when the grain size exceeds sqrt(D t).

Fisher's equations drop diffusion along y in the grains. Whipple (1954) solved the
problem with it kept; Le Claire (1963) writes that solution as a single integral
(:func:`whipple_concentration`) and derives from it the reading of sectioning data,
ln(c_bar) falling linearly in y^(6/5) with D' delta = 1.322 (D/t)^(1/2) times the
-5/3 power of that slope (:func:`le_claire_boundary_product`). Fisher's equations are
the large-beta limit of Whipple's, beta = (D'/D)(delta/2)/(D t)^(1/2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import quad
from scipy.special import erfc, erfcx


@dataclass(frozen=True)
class FisherBoundary:
    """A grain boundary of thickness ``delta`` (m) and diffusivity ``d_boundary`` in a
    grain of diffusivity ``d_lattice`` (both m^2/s)."""

    d_lattice: float
    d_boundary: float
    delta: float

    def __post_init__(self):
        if not 0 < self.d_lattice < self.d_boundary:
            raise ValueError("Fisher's model needs 0 < D_lattice < D_boundary")
        if self.delta <= 0:
            raise ValueError("boundary thickness must be positive")

    def scaled(self, x, y, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x1, y1, t1) of eq. (5)."""
        ratio = np.sqrt(self.d_boundary / self.d_lattice)
        return (np.asarray(x) / self.delta, np.asarray(y) / (self.delta * ratio),
                self.d_lattice * np.asarray(t) / self.delta**2)

    def concentration(self, x, y, t) -> np.ndarray:
        """Eq. (8): C at distance x from the boundary and depth y after time t, surface at 1."""
        x1, y1, t1 = self.scaled(x, y, t)
        return np.exp(-np.sqrt(2) * y1 / (np.pi**0.25 * t1**0.25)) * erfc(x1 / (2 * np.sqrt(t1)))

    def solve(self, t: float, **grid) -> FisherSolution:
        """Numerical solution of eqs. (3)-(4) at time t (s); ``grid`` is passed to :func:`solve_fisher`.

        The result is on the scaled grid; x = x1 delta and y = y1 delta (D'/D)^(1/2).
        """
        return solve_fisher(float(self.scaled(0.0, 0.0, t)[2]), **grid)

    def penetration_slope(self, t) -> float:
        """Eq. (14): d log10(a) / dy of the layer-averaged concentration, in 1/m."""
        return -np.sqrt(2) * np.sqrt(self.d_lattice / (self.d_boundary * self.delta)) * np.log10(np.e) \
            / (np.pi * self.d_lattice * t) ** 0.25


def boundary_product_from_slope(slope: float, d_lattice: float, t: float) -> float:
    """D' delta (m^3/s) from a measured penetration slope d log10(a)/dy (1/m), inverting eq. (14).

    The classic reading of a radiotracer sectioning experiment; Fisher's eq. (15)
    gives D / (D' delta) = 21.3 cm^-1 for silver at 478 C this way.
    """
    if slope >= 0:
        raise ValueError("the penetration slope of log(activity) against depth must be negative")
    root = -slope * (np.pi * d_lattice * t) ** 0.25 / (np.sqrt(2) * np.log10(np.e))
    return d_lattice / root**2


@dataclass
class FisherSolution:
    """Numerical solution on the scaled grid: C[i_y, i_x] at t1, with x1[0] = 0 the boundary."""

    t1: float
    x1: np.ndarray
    y1: np.ndarray
    c: np.ndarray

    def layer_average(self) -> np.ndarray:
        """Integral of C over x1 at each depth y1, the quantity a sectioning experiment measures (eq. 9)."""
        return np.trapezoid(self.c, self.x1, axis=1)

    def at(self, x1, y1) -> float:
        """C at scaled coordinates, interpolated linearly."""
        from scipy.interpolate import RegularGridInterpolator

        return float(RegularGridInterpolator((self.y1, self.x1), self.c)([[y1, x1]])[0])


def solve_fisher(t1: float, x_max: float | None = None, y_max: float | None = None,
                 dx: float = 0.5, dy: float = 0.25, steps: int = 400) -> FisherSolution:
    """Solve the scaled eqs. (3a)-(4a) of Fisher (1951) without his approximation.

    Backward Euler in t1 on a uniform grid; the slab takes (dC/dx1)_0 from the
    second-order one-sided difference his eq. (3b) uses, and each grain row only
    diffuses in x1, as eq. (4a) states. The surface row of the slab is held at 1.
    """
    x_max = x_max or 12 * np.sqrt(t1) + 4
    y_max = y_max or 8 * t1**0.25 + 4
    nx, ny = int(round(x_max / dx)), int(round(y_max / dy))
    x1, y1 = np.arange(nx + 1) * dx, np.arange(ny + 1) * dy
    dt = t1 / steps
    rows, cols, vals = [], [], []

    def idx(k, i):  # unknowns are rows k = 1..ny of the slab (i = 0) and grains (i = 1..nx-1)
        return (k - 1) * nx + i

    def add(r, c, v):
        rows.append(r), cols.append(c), vals.append(v)

    rhs_const = np.zeros(ny * nx)
    for k in range(1, ny + 1):
        r = idx(k, 0)
        # slab: C_t = C_yy + 2 C_x(0), with C_x(0) = (-3 C0 + 4 C1 - C2) / (2 dx)
        add(r, r, 1 / dt + 2 / dy**2 + 2 * 1.5 / dx)
        add(r, idx(k, 1), -2 * 2.0 / dx)
        add(r, idx(k, 2), 2 * 0.5 / dx)
        if k > 1:
            add(r, idx(k - 1, 0), -1 / dy**2)
        else:
            rhs_const[r] += 1 / dy**2  # surface held at 1
        if k < ny:
            add(r, idx(k + 1, 0), -1 / dy**2)
        else:
            add(r, r, -1 / dy**2)  # no flux at the far end (mirror)
        for i in range(1, nx):
            r = idx(k, i)
            add(r, r, 1 / dt + 2 / dx**2)
            add(r, idx(k, i - 1), -1 / dx**2)
            if i < nx - 1:
                add(r, idx(k, i + 1), -1 / dx**2)  # C = 0 beyond x_max
    lu = spla.splu(sp.csc_matrix((vals, (rows, cols)), shape=(ny * nx, ny * nx)))
    c = np.zeros(ny * nx)
    for _ in range(steps):
        c = lu.solve(c / dt + rhs_const)
    grid = np.zeros((ny + 1, nx + 1))
    grid[0, 0] = 1.0
    grid[0, 1:] = erfc(x1[1:] / (2 * np.sqrt(t1)))  # surface row of the grains, fed by the surface itself
    grid[1:, :nx] = c.reshape(ny, nx)
    return FisherSolution(t1, x1, y1, grid)


# --- Whipple's exact solution and Le Claire's analysis ---------------------------------------------------------

def whipple_beta(d_lattice: float, d_boundary: float, delta: float, t: float) -> float:
    """beta = (D'/D) (delta/2) / (D t)^(1/2), Le Claire, Brit. J. Appl. Phys. 14 (1963) 351, eq. (5)."""
    return d_boundary / d_lattice * 0.5 * delta / np.sqrt(d_lattice * t)


def _tail_integral(integrand, eta: float) -> float:
    """Integral of ``integrand`` from 1 to infinity, split where the factor exp(-eta^2 / 4 s) has risen,
    so the quadrature resolves both the rise and the long tail."""
    split = max(10.0, eta * eta)
    head = quad(integrand, 1.0, split, limit=400, epsabs=0.0, epsrel=1e-10)[0]
    return head + quad(integrand, split, np.inf, limit=400, epsabs=0.0, epsrel=1e-10)[0]


def whipple_concentration(eta: float, xi: float, beta: float) -> float:
    """c / c_0 of Whipple's exact solution (Phil. Mag. 45 (1954) 1225) as Le Claire writes it, his eq. (7),
    for D'/D large enough that the integral may run to infinity: at reduced depth eta = y / (D t)^(1/2) and
    distance from the slab xi = (x - delta/2) / (D t)^(1/2), from a surface held at c_0. The first term is
    direct lattice diffusion, the second the supply down the boundary."""
    def integrand(s):
        return s ** -1.5 * np.exp(-eta * eta / (4 * s)) * erfc(0.5 * ((s - 1) / beta + xi))

    return float(erfc(eta / 2) + eta / (2 * np.sqrt(np.pi)) * _tail_integral(integrand, eta))


def whipple_layer_integral(eta: float, beta: float) -> float:
    """The boundary contribution to the layer-averaged concentration of a sectioning experiment, Le Claire's
    eq. (13) without its constant factor: eta int_1^inf exp(-eta^2 / 4 s) (exp(-V^2) / pi^(1/2) - V erfc V)
    s^(-3/2) ds, V = (s - 1) / (2 beta). Only its logarithmic slope with depth is used."""
    def integrand(s):
        v = (s - 1) / (2 * beta)
        # exp(-V^2) (1/sqrt(pi) - V erfcx(V)) is exp(-V^2)/sqrt(pi) - V erfc(V) without the cancellation
        return s ** -1.5 * np.exp(-eta * eta / (4 * s) - v * v) * (1 / np.sqrt(np.pi) - v * erfcx(v))

    return float(eta * _tail_integral(integrand, eta))


def le_claire_boundary_product(slope: float, d_lattice: float, t: float, coefficient: float = 0.78) -> float:
    """D' delta (m^3/s) from the slope d ln(c_bar) / d y^(6/5) (m^-6/5) of a sectioning profile, Le Claire's
    eq. (15): D' delta = (4 D / t)^(1/2) (coefficient / |slope|)^(5/3), the coefficient being
    -d ln(c_bar) / d (eta beta^(-1/2))^(6/5), 'for practical purposes constant at 0.78 if beta is large'
    (p. 356; Whipple's solution, beta above about 10). This is the familiar 1.322 (D/t)^(1/2) form. For
    Suzuoka's thin-source condition Le Claire gives 0.72 beta^0.008 (eq. 16)."""
    if slope >= 0:
        raise ValueError("ln(c_bar) must fall with y^(6/5): the slope must be negative")
    return np.sqrt(4 * d_lattice / t) * (coefficient / -slope) ** (5 / 3)


def harrison_regime(d_lattice: float, grain_radius: float, boundary_width: float, t: float,
                    margin: float = 10.0) -> str:
    """Harrison's kinetic regime, Trans. Faraday Soc. 57 (1961) 1191: 'A' when lattice diffusion crosses a grain
    within the anneal, a_g^2 / D_g << t (his eq. 1), so the solid diffuses as a whole with Hart's average
    D_c = f D' + (1 - f) D (eq. 3; :class:`ShortCircuit`), f = 3 delta / a_g for spherical grains of radius a_g;
    'C' when lattice diffusion does not leave the boundary layer, (D_g t)^(1/2) << delta (eq. 5), so only the
    boundaries carry material; 'B' between, where Whipple's solution and :func:`le_claire_boundary_product`
    apply. Harrison leaves '<<' unquantified (and calls eq. 1 perhaps far too strict); here it means a factor
    ``margin``, a choice of this package."""
    if min(d_lattice, grain_radius, boundary_width, t, margin) <= 0:
        raise ValueError("diffusivity, grain radius, boundary width, time and margin must be positive")
    if grain_radius**2 / d_lattice * margin < t:
        return "A"
    if np.sqrt(d_lattice * t) * margin < boundary_width:
        return "C"
    return "B"
