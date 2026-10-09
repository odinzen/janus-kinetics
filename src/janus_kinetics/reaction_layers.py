"""Growth of stoichiometric reaction layers in a binary couple.

Many film reactions grow compounds with a negligible homogeneity range (silicides,
aluminides, Cu-Sn and Ni-Sn intermetallics, wustite on iron). Inside such a layer
the concentration gradient vanishes while the flux does not, so the usual
interdiffusion coefficient is undefined. Wagner's integrated diffusion
coefficient replaces it (van Loo, Prog. Solid State Chem. 20 (1990) 47, eq. 17):

    D_int = integral of D~ dN across the layer

Each phase here is a line compound of fixed mole fraction x of the second element
B and molar volume V (per mole of atoms). The flux across a layer of thickness d is
uniform, and with D~ = x_A D_B + x_B D_A (van Loo eq. 10 at equal partial molar
volumes) and Darken's D_i = M_i x_i dmu_i/dx_i, integrating across the layer gives
the intrinsic (lattice-frame) fluxes

    J_A =  x_A x_B M_A dS / (V d),   J_B = -x_A x_B M_B dS / (V d)
    D_int = x_A x_B (x_A M_B + x_B M_A) dS

where dS is the jump of mu_B - mu_A across the layer. With local equilibrium at
each interface, mu_B - mu_A there is the slope of the common tangent between the
two neighbouring compounds, (G_right - G_left) / (x_right - x_left), so dS is the
difference of the two chord slopes at the compound and is positive exactly when
the compound lies below the line joining its neighbours.

Every layer has its own lattice, moving rigidly because the flux inside it is
uniform. Conservation of each element across each moving interface,

    c_right (v - w_right) - c_left (v - w_left) = J_right - J_left

fixes, from the left end member outward, every interface velocity v and every
lattice velocity w. Inert markers ride the lattice of the phase they sit in, which
gives the Kirkendall plane positions of van Loo's Table 2.1 and Fig. 2.9. Reaction
control at an interface (van Loo pp. 56-57) and the nucleation of a phase (p. 61)
are outside this model: every phase in the sequence grows from the start.

The same layers grow as concentric shells around a cylindrical or spherical core.
Van Loo (pp. 74-76, eqs. 25-27, after Carter) treats one layer; the several-layer
form is derived here (see _solve_curved) and reduces to his eq. 26 for one layer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .errors import ConvergenceError
from .schedule import Temperature, temperature_at

Value = float | Callable[[float], float]


def _at(value, T: float):
    return value(T) if callable(value) else value


@dataclass(frozen=True)
class LineCompound:
    """A phase of fixed composition in an A-B couple.

    ``x`` is the mole fraction of B and ``molar_volume`` is per mole of atoms
    (m^3/mol). ``gibbs`` is the Gibbs energy per mole of atoms (J/mol), a number
    or a function of T. A phase that grows as a layer needs its transport, given
    either as ``mobility`` = (M_A, M_B) in m^2 mol/(J s) (the tracer diffusivity
    is D* = R T M), constant or a function of T, or as a measured
    ``integrated_diffusivity`` (m^2/s, constant or a function of T) together with
    ``tracer_ratio`` = M_B / M_A, which only sets how the flux splits between the
    elements and so where markers end up. End members need neither.
    """

    name: str
    x: float
    molar_volume: float
    gibbs: Value | None = None
    mobility: tuple[float, float] | Callable[[float], tuple[float, float]] | None = None
    integrated_diffusivity: Value | None = None
    tracer_ratio: float | None = None

    def __post_init__(self):
        if not 0.0 <= self.x <= 1.0:
            raise ValueError(f"{self.name}: x must lie in [0, 1]")
        if self.molar_volume <= 0:
            raise ValueError(f"{self.name}: molar volume must be positive")
        if self.mobility is not None and self.integrated_diffusivity is not None:
            raise ValueError(f"{self.name}: give mobilities or an integrated diffusivity, not both")
        if self.integrated_diffusivity is not None:
            if self.tracer_ratio is None or not 0.0 <= self.tracer_ratio <= np.inf:
                raise ValueError(f"{self.name}: an integrated diffusivity needs tracer_ratio = M_B / M_A in [0, inf]")

    @classmethod
    def from_pycalphad(cls, db, phase: str, elements: Sequence[str], molar_volume: float,
                       name: str | None = None, **transport) -> LineCompound:
        """A stoichiometric pycalphad phase; ``elements`` = (A, B) and G(T) is read from the database."""
        from pycalphad import calculate, variables as v

        comps = [el.upper() for el in elements]
        species = comps + ["VA"]

        def gm(T):
            res = calculate(db, species, phase, T=float(T), P=101325, N=1, output="GM")
            return float(np.ravel(res.GM.values)[0])

        res = calculate(db, species, phase, T=1000.0, P=101325, N=1, output="GM")
        points = np.asarray(res.X.sel(component=comps[1]).values).ravel()
        if np.ptp(points) > 1e-9:
            raise ValueError(f"{phase} is not a line compound in {elements}: its composition varies")
        return cls(name or phase, float(points[0]), molar_volume, gibbs=gm, **transport)

    def concentrations(self) -> np.ndarray:
        return np.array([1.0 - self.x, self.x]) / self.molar_volume

    def is_layer_ready(self) -> bool:
        return self.mobility is not None or self.integrated_diffusivity is not None


def integrated_diffusivity(left: LineCompound, layer: LineCompound, right: LineCompound, T: float) -> float:
    """Wagner's D_int (van Loo eq. 17) of ``layer`` between its neighbours at T, in m^2/s."""
    return _transport(left, layer, right, T)[0]


def _chord(a: LineCompound, b: LineCompound, T: float) -> float:
    if a.gibbs is None or b.gibbs is None:
        raise ValueError(f"layer mobilities need the Gibbs energies of {a.name} and {b.name}")
    return (_at(b.gibbs, T) - _at(a.gibbs, T)) / (b.x - a.x)


def _transport(left: LineCompound, layer: LineCompound, right: LineCompound, T: float):
    """(D_int, f_A, f_B) with J_A = D_int f_A / (V d) and J_B = -D_int f_B / (V d)."""
    xa, xb = 1.0 - layer.x, layer.x
    if layer.integrated_diffusivity is not None:
        rho = layer.tracer_ratio
        if np.isinf(rho):
            return float(_at(layer.integrated_diffusivity, T)), 0.0, 1.0 / xa
        return float(_at(layer.integrated_diffusivity, T)), 1.0 / (xa * rho + xb), rho / (xa * rho + xb)
    if layer.mobility is None:
        raise ValueError(f"{layer.name} grows as a layer and needs a mobility or integrated diffusivity")
    m_a, m_b = (float(m) for m in _at(layer.mobility, T))
    if m_a < 0 or m_b < 0 or m_a + m_b == 0:
        raise ValueError(f"{layer.name}: mobilities must be non-negative and not both zero")
    jump = _chord(layer, right, T) - _chord(left, layer, T)
    if jump <= 0:
        raise ValueError(
            f"{layer.name} lies on or above the line joining {left.name} and {right.name} at {T:g} K, "
            "so it has no driving force to form; leave it out of the sequence")
    weight = xa * m_b + xb * m_a
    return xa * xb * weight * jump, m_a / weight, m_b / weight


def _velocities(phases: Sequence[LineCompound], d: np.ndarray, T: float):
    """Interface velocities v (n+1) and lattice velocities w (n+2), left end member at rest."""
    n = len(phases) - 2
    flux = np.zeros((n + 2, 2))
    for k in range(1, n + 1):
        d_int, f_a, f_b = _transport(phases[k - 1], phases[k], phases[k + 1], T)
        flux[k] = np.array([f_a, -f_b]) * d_int / (phases[k].molar_volume * d[k - 1])
    conc = [p.concentrations() for p in phases]
    v, w = np.zeros(n + 1), np.zeros(n + 2)
    for j in range(n + 1):
        cl, cr = conc[j], conc[j + 1]
        a = np.column_stack([cr - cl, -cr])
        v[j], w[j + 1] = np.linalg.solve(a, flux[j + 1] - flux[j] - cl * w[j])
    return v, w


def _check_sequence(phases: Sequence[LineCompound]):
    if len(phases) < 2:
        raise ValueError("a couple needs at least two end members")
    xs = np.array([p.x for p in phases])
    if np.any(np.diff(xs) <= 0):
        raise ValueError("list the phases in order of strictly increasing x, end members first and last")
    names = [p.name for p in phases]
    if len(set(names)) != len(names):
        raise ValueError("phase names must be unique")


@dataclass(frozen=True)
class ParabolicGrowth:
    """Similarity solution of a couple started at t = 0 at constant T.

    Positions scale with sqrt(t) in the frame of the anchored end member, with the
    original interface at 0: interface j sits at ``interfaces[j] * sqrt(t)``, and
    layer k has d^2 = 2 k_p t (van Loo's growth constant, p. 55) with k_p in
    ``growth_constants``. ``consumed`` gives the thickness of each end member used
    up, divided by sqrt(t); ``marker`` is the Kirkendall plane position divided by
    sqrt(t) and ``marker_phase`` the phase it lies in.
    """

    names: tuple[str, ...]
    interfaces: np.ndarray
    growth_constants: dict[str, float]
    consumed: dict[str, float]
    marker: float
    marker_phase: str
    lattice_velocity: np.ndarray = field(repr=False)


def parabolic_growth(phases: Sequence[LineCompound], temperature: float, anchor: str = "left") -> ParabolicGrowth:
    """Parabolic growth constants of every layer in a fresh couple at constant T.

    All layers start together (van Loo p. 61, item c). With d_k = delta_k sqrt(t)
    every velocity scales as 1/sqrt(t), so delta_k = 2 (V_k - V_(k-1)) where V are
    the interface velocity coefficients computed with thicknesses delta. That
    nonlinear system is solved in log(delta).
    """
    _check_sequence(phases)
    if anchor not in ("left", "right"):
        raise ValueError("anchor must be 'left' or 'right'")
    n = len(phases) - 2
    T = float(temperature)
    if n == 0:
        raise ValueError("no layer between the end members")

    guess = []
    for k in range(1, n + 1):
        d_int = _transport(phases[k - 1], phases[k], phases[k + 1], T)[0]
        xl, x, xr = phases[k - 1].x, phases[k].x, phases[k + 1].x
        guess.append(np.sqrt(2 * d_int * (xr - xl) / ((xr - x) * (x - xl))))
    guess = np.log(guess)

    def residual(log_delta):
        delta = np.exp(log_delta)
        v, _ = _velocities(phases, delta, T)
        return 2 * np.diff(v) / delta - 1.0

    # Newton from the isolated-layer guess can wander off where a layer would shrink.
    # In s = ln t the scaled thicknesses obey d(delta)/ds = dv(delta) - delta/2, whose
    # fixed point is the answer and which every couple relaxes to, so march there
    # first and polish with Newton.
    relax = solve_ivp(lambda s, u: residual(u) / 2, (0.0, 80.0), guess, method="Radau", rtol=1e-10, atol=1e-12)
    sol = root(residual, relax.y[:, -1], method="hybr", options={"xtol": 1e-13})
    if not sol.success or np.max(np.abs(residual(sol.x))) > 1e-9:
        raise ConvergenceError(f"parabolic growth constants did not converge: {sol.message}")
    delta = np.exp(sol.x)
    v, w = _velocities(phases, delta, T)
    shift = w[0] if anchor == "left" else w[-1]
    v, w = v - shift, w - shift
    pos = 2 * v

    edges = np.concatenate([[-np.inf], pos, [np.inf]])
    marker_phase = None
    for p in range(n + 2):
        m = 2 * w[p]
        if edges[p] - 1e-12 * abs(m) <= m <= edges[p + 1] + 1e-12 * abs(m):
            marker_phase, marker = phases[p].name, m
            break
    if marker_phase is None:
        raise ConvergenceError("no phase holds the Kirkendall plane consistently")
    return ParabolicGrowth(
        names=tuple(p.name for p in phases),
        interfaces=pos,
        growth_constants={phases[k].name: delta[k - 1] ** 2 / 2 for k in range(1, n + 1)},
        consumed={phases[0].name: 2 * (w[0] - v[0]), phases[-1].name: 2 * (v[-1] - w[-1])},
        marker=marker,
        marker_phase=marker_phase,
        lattice_velocity=w,
    )


@dataclass
class ReactionResult:
    """Layer thicknesses (m) of every phase at ``times``.

    ``thickness[name]`` is inf for a semi-infinite end member and 0 once a phase is
    used up. ``interfaces[i]`` holds the interface positions at ``times[i]`` in the
    anchored end member's frame, original interface at 0 for a fresh couple.
    ``markers`` has one column per marker. ``consumed`` lists (time, name) for
    every end member used up during the run. In a cylinder or sphere the interfaces
    and markers are radii and the core's "thickness" is its radius.
    """

    times: np.ndarray
    thickness: dict[str, np.ndarray]
    interfaces: list[np.ndarray]
    markers: np.ndarray
    consumed: list[tuple[float, str]]
    geometry: str = "planar"


def solve_reaction_layers(
    phases: Sequence[LineCompound],
    t_end: float,
    temperature: Temperature,
    thickness: Sequence[float] | None = None,
    end_thickness: tuple[float | None, float | None] = (None, None),
    markers: Sequence[float] = (0.0,),
    save_times=None,
    anchor: str | None = None,
    rtol: float = 1e-9,
    geometry: str = "planar",
) -> ReactionResult:
    """Grow the layers of a binary couple of line compounds over an anneal.

    ``phases`` runs from the left end member to the right one in order of
    increasing x. ``thickness`` gives the starting thickness of each interior
    layer; None starts a fresh couple from the similarity solution at T(0).
    ``end_thickness`` makes an end member a film of finite thickness; when one is
    used up the next phase becomes the end member, as a metal film turns into its
    first compound and then into the next. Markers start at the given positions,
    measured from the original interface (the first interface for a pre-grown
    stack). The frame is the lattice of the semi-infinite end member, or of
    ``anchor`` ('left' or 'right') when given.

    With ``geometry`` "cylindrical" or "spherical" the first phase is a core of radius
    ``end_thickness[0]`` (a wire or particle) and the layers grow as shells around it,
    the last phase being the surrounding matrix (or a shell of thickness
    ``end_thickness[1]``). Positions are then radii, markers start at the given distance
    from the core's surface, and the frame is the core's lattice. Rigid shells are
    assumed, as in van Loo's conversion relations: no stress or voiding.
    """
    _check_sequence(phases)
    phases = list(phases)
    n = len(phases) - 2
    requested_anchor = anchor
    if anchor is None:
        anchor = "right" if end_thickness[1] is None and end_thickness[0] is not None else "left"
    if anchor not in ("left", "right"):
        raise ValueError("anchor must be 'left' or 'right'")
    for t in end_thickness:
        if t is not None and t <= 0:
            raise ValueError("finite end members need a positive thickness")
    save = np.unique(np.append(np.asarray([] if save_times is None else save_times, dtype=float), t_end))
    if np.any(save <= 0) or save[-1] > t_end:
        raise ValueError("save times must lie in (0, t_end]")
    markers = np.asarray(markers, dtype=float)
    if geometry != "planar":
        if geometry not in _EXPONENT:
            raise ValueError(f"geometry must be 'planar', 'cylindrical' or 'spherical', got {geometry!r}")
        if end_thickness[0] is None:
            raise ValueError("a cylinder or sphere needs the core radius as end_thickness[0]")
        if requested_anchor == "right":
            raise ValueError("in a cylinder or sphere the frame is the core's lattice")
        out_t, out_pos, out_mark, out_thick, consumed = _solve_curved(
            phases, t_end, temperature, thickness, end_thickness, markers, save, rtol, _EXPONENT[geometry])
        return ReactionResult(np.asarray(out_t), {k: np.asarray(v) for k, v in out_thick.items()}, out_pos,
                              np.asarray(out_mark), consumed, geometry)

    if thickness is None:
        t0 = 1e-10 * t_end
        growth = parabolic_growth(phases, temperature_at(temperature, t0), anchor)
        root_t = np.sqrt(t0)
        pos = growth.interfaces * root_t
        if np.any(markers != 0.0):
            raise ValueError("in a fresh couple, markers can only start at the original interface (0)")
        marker_pos = np.full(markers.size, growth.marker * root_t)
        d = np.diff(pos)
        shift = 2 * growth.lattice_velocity * root_t
        left = -end_thickness[0] + shift[0] if end_thickness[0] is not None else 0.0
        right = end_thickness[1] + shift[-1] if end_thickness[1] is not None else 0.0
        t_start = t0
    else:
        d = np.asarray(thickness, dtype=float)
        if d.shape != (n,) or np.any(d <= 0):
            raise ValueError(f"give {n} positive layer thicknesses, or None for a fresh couple")
        pos = np.concatenate([[0.0], np.cumsum(d)])
        marker_pos = markers.copy()
        left = -end_thickness[0] if end_thickness[0] is not None else 0.0
        right = pos[-1] + end_thickness[1] if end_thickness[1] is not None else 0.0
        t_start = 0.0

    finite = [end_thickness[0] is not None, end_thickness[1] is not None]
    names = [p.name for p in phases]
    out_t, out_pos, out_mark = [], [], []
    out_thick = {name: [] for name in names}
    consumed: list[tuple[float, str]] = []
    gone: set[str] = set()

    def record(t, left_s, xi0, d_layers, right_s, mk):
        interfaces = xi0 + np.concatenate([[0.0], np.cumsum(d_layers)])
        out_t.append(t)
        out_pos.append(interfaces)
        out_mark.append(mk.copy())
        current = {p.name: d_layers[k - 1] for k, p in enumerate(phases) if 0 < k < len(phases) - 1}
        current[phases[0].name] = interfaces[0] - left_s if finite[0] else np.inf
        current[phases[-1].name] = right_s - interfaces[-1] if finite[1] else np.inf
        for name in names:
            out_thick[name].append(current.get(name, 0.0))

    # Saves that fall before the similarity start of a fresh couple scale as sqrt(t).
    for ts in save[save < t_start]:
        scale = np.sqrt(ts / t_start)
        record(ts, left, pos[0] * scale, d * scale, right, marker_pos * scale)
    save = save[save >= t_start]

    y = np.concatenate([[left, pos[0]], np.log(d), [right], marker_pos])
    t = t_start
    while True:
        n = len(phases) - 2
        if n == 0:
            for ts in save[save > t - 1e-15 * t_end]:
                record(ts, y[0], y[1], np.array([]), y[2], y[3:])
            break

        def rhs(t, y, phases=phases, n=n):
            T = temperature_at(temperature, t)
            d_layers = np.exp(y[2:2 + n])
            v, w = _velocities(phases, d_layers, T)
            shift = w[0] if anchor == "left" else w[-1]
            v, w = v - shift, w - shift
            interfaces = y[1] + np.concatenate([[0.0], np.cumsum(d_layers)])
            dy = np.empty_like(y)
            dy[0] = w[0]
            dy[1] = v[0]
            dy[2:2 + n] = np.diff(v) / d_layers
            dy[2 + n] = w[-1]
            dy[3 + n:] = w[np.searchsorted(interfaces, y[3 + n:])]
            return dy

        events = []
        if finite[0]:
            ev_left = lambda t, y, n=n: y[1] - y[0]
            ev_left.terminal, ev_left.direction = True, -1
            events.append(ev_left)
        if finite[1]:
            ev_right = lambda t, y, n=n: y[2 + n] - (y[1] + np.exp(y[2:2 + n]).sum())
            ev_right.terminal, ev_right.direction = True, -1
            events.append(ev_right)
        scale = max(np.exp(y[2:2 + n]).sum(), 1e-9)
        atol = np.full(y.size, rtol * scale)
        atol[2:2 + n] = rtol
        sol = solve_ivp(rhs, (t, t_end), y, method="Radau", rtol=rtol, atol=atol, events=events or None,
                        dense_output=True)
        if sol.status == -1:
            raise ConvergenceError(f"layer growth integration failed: {sol.message}")
        t_stop = sol.t[-1]
        for ts in save[(save > t - 1e-15 * t_end) & (save <= t_stop)]:
            ys = sol.sol(ts)
            record(ts, ys[0], ys[1], np.exp(ys[2:2 + n]), ys[2 + n], ys[3 + n:])
        save = save[save > t_stop]
        if sol.status == 0 or save.size == 0:
            break

        y_end = sol.y[:, -1]
        d_layers = np.exp(y_end[2:2 + n])
        hit_left = bool(sol.t_events[0].size) if finite[0] else False
        if hit_left:
            consumed.append((t_stop, phases[0].name))
            gone.add(phases.pop(0).name)
            y = np.concatenate([[y_end[1], y_end[1] + d_layers[0]], y_end[3:2 + n], [y_end[2 + n]], y_end[3 + n:]])
        else:
            consumed.append((t_stop, phases[-1].name))
            gone.add(phases.pop().name)
            right_s = y_end[1] + d_layers.sum()
            y = np.concatenate([y_end[:2], y_end[2:1 + n], [right_s], y_end[3 + n:]])
        t = t_stop

    return ReactionResult(
        times=np.asarray(out_t),
        thickness={name: np.asarray(vals) for name, vals in out_thick.items()},
        interfaces=out_pos,
        markers=np.asarray(out_mark),
        consumed=consumed,
    )


_EXPONENT = {"cylindrical": 1, "spherical": 2}


def _resistance(r_in: np.ndarray, d: np.ndarray, p: int) -> np.ndarray:
    """Integral of r^-p dr across shells of inner radius r_in and thickness d: the length that takes the place
    of the thickness in the flux of a curved layer (ln(r_out / r_in) for a cylinder, 1/r_in - 1/r_out for a
    sphere), written to keep its precision for shells much thinner than their radius."""
    with np.errstate(divide="ignore"):
        return np.log1p(d / r_in) if p == 1 else d / (r_in * (r_in + d))


def _solve_curved(phases, t_end, temperature, thickness, end_thickness, markers, save, rtol, p):
    """Concentric shells around a core of the first phase; see solve_reaction_layers.

    Inside a line-compound layer each element is conserved at constant concentration, so
    r^p (c_i w + J_i) is the same at every radius, and with J_A = x_A x_B M_A / V dS/dr and
    J_B = -x_A x_B M_B / V dS/dr both the lattice velocity w and dS/dr fall as r^-p. Integrating
    dS/dr across the layer turns the planar fluxes into r^p J_i = (planar flux) with the
    thickness d replaced by the integral of r^-p dr. The interface balances, multiplied by r^p,
    are then the planar ones in V = r^p v and W = r^p w, so _velocities gives V and W directly.
    The core passes no flux, and its lattice velocity W / r^p must stay finite at the centre, so
    W = 0 there: the core's lattice is the frame. The core is followed by u = r_core^(p+1),
    whose rate (p+1) V_0 stays finite as the core shrinks while dr_core/dt does not."""
    names = [ph.name for ph in phases]
    n = len(phases) - 2
    radius = end_thickness[0]
    outer_finite = end_thickness[1] is not None
    if thickness is None:
        t0 = 1e-10 * t_end
        growth = parabolic_growth(phases, temperature_at(temperature, t0), "left")
        root_t = np.sqrt(t0)
        if np.any(markers != 0.0):
            raise ValueError("in a fresh couple, markers can only start at the original interface (0)")
        rel = growth.interfaces * root_t
        marker_r = np.full(markers.size, growth.marker * root_t)
        outer = end_thickness[1] + 2 * growth.lattice_velocity[-1] * root_t if outer_finite else rel[-1]
        t_start = t0
    else:
        d0 = np.asarray(thickness, dtype=float)
        if d0.shape != (n,) or np.any(d0 <= 0):
            raise ValueError(f"give {n} positive layer thicknesses, or None for a fresh couple")
        rel = np.concatenate([[0.0], np.cumsum(d0)])
        marker_r = markers.copy()
        outer = rel[-1] + end_thickness[1] if outer_finite else rel[-1]
        t_start = 0.0

    out_t, out_pos, out_mark = [], [], []
    out_thick = {name: [] for name in names}
    consumed: list[tuple[float, str]] = []

    def record(t, r_core, d_layers, outer_offset, marker_offsets):
        r_outer, mk = radius + outer_offset, radius + marker_offsets
        interfaces = r_core + np.concatenate([[0.0], np.cumsum(d_layers)])
        out_t.append(t)
        out_pos.append(interfaces)
        out_mark.append(mk.copy())
        current = {ph.name: d_layers[k - 1] for k, ph in enumerate(phases) if 0 < k < len(phases) - 1}
        current[phases[0].name] = r_core
        current[phases[-1].name] = r_outer - interfaces[-1] if outer_finite else np.inf
        for name in names:
            out_thick[name].append(current.get(name, 0.0))

    for ts in save[save < t_start]:
        scale = np.sqrt(ts / t_start)
        record(ts, radius + rel[0] * scale, np.diff(rel) * scale, outer, marker_r * scale)
    save = save[save >= t_start]

    y = np.concatenate([[(radius + rel[0]) ** (p + 1)], np.log(np.diff(rel)), [outer], marker_r])
    t = t_start
    while True:
        n = len(phases) - 2
        if n == 0:
            for ts in save[save > t - 1e-15 * t_end]:
                record(ts, y[0] ** (1 / (p + 1)), np.array([]), y[1], y[2:])
            break

        def kinematics(t, y, phases=phases, n=n):
            """Interface radii and velocities, layer thicknesses and the W = r^p w of every lattice."""
            r_core = max(y[0], 0.0) ** (1 / (p + 1))
            d_layers = np.exp(y[1:1 + n])
            r = r_core + np.concatenate([[0.0], np.cumsum(d_layers)])
            big_v, big_w = _velocities(phases, _resistance(r[:-1], d_layers, p), temperature_at(temperature, t))
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.where(r > 0, big_v / r**p, 0.0)
            return r, v, d_layers, big_v, big_w

        def held(r, v, big_w, j):
            """Whether a marker on interface j stays on it: the lattice inside moves outward at least as
            fast as the interface and the one outside no faster (equality on one side when only one
            element diffuses, which pins the Kirkendall plane to the interface)."""
            if r[j] <= 0:
                return False
            inner, outside, here = big_w[j] / r[j] ** p, big_w[j + 1] / r[j] ** p, v[j]
            slack = 1e-9 * max(abs(inner), abs(outside), abs(here), 1e-300)
            return inner >= here - slack and outside <= here + slack

        # A marker on such an interface moves with it. Followed through the lattices instead it
        # would leave: in a cylinder or sphere lattice speeds fall as r^-p, so a marker a rounding
        # error off the interface drifts away from it, which the planar model does not see.
        r, v, _, _, big_w = kinematics(t, y)
        size = max(np.exp(y[1:1 + n]).sum(), 1e-9)
        on_interface = np.full(y.size - 2 - n, -1)
        for i, m in enumerate(radius + y[2 + n:]):
            near = np.flatnonzero(np.abs(r - m) <= 1e-6 * size)
            if near.size and held(r, v, big_w, int(near[0])):
                on_interface[i] = int(near[0])

        def rhs(t, y, n=n, on_interface=on_interface):
            r, v, d_layers, big_v, big_w = kinematics(t, y)
            dy = np.empty_like(y)
            dy[0] = (p + 1) * big_v[0]
            dy[1:1 + n] = np.diff(v) / d_layers
            # the outer surface rides the outer lattice; in an endless matrix it is only a lattice point
            dy[1 + n] = big_w[-1] / (radius + y[1 + n]) ** p
            mk = radius + y[2 + n:]
            # the floor keeps r^p from underflowing for a marker carried to the centre by a vanishing core
            speed = big_w[np.searchsorted(r, mk)] / np.maximum(mk, 1e-100) ** p
            for i, j in enumerate(on_interface):
                if j >= 0 and held(r, v, big_w, j):
                    speed[i] = v[j]
            dy[2 + n:] = speed
            return dy

        def markers_at(ts, ys, on_interface=on_interface):
            offsets = ys[2 + n:].copy()
            r = kinematics(ts, ys)[0]
            attached = on_interface >= 0
            offsets[attached] = r[on_interface[attached]] - radius
            return offsets

        # as the core vanishes its surface speed diverges (the integral of r^-p dr across the
        # first layer does), so it counts as used up at 1e-4 of the radius it had when this
        # stretch began; the run then ends 2.4e-8 (sphere) and 1.4e-7 (cylinder) of the conversion
        # time early against van Loo's eq. 26
        u_end = (1e-4 * max(y[0], 0.0) ** (1 / (p + 1))) ** (p + 1)
        ev_core = lambda t, y, u_end=u_end: y[0] - u_end
        ev_core.terminal, ev_core.direction = True, -1
        events = [ev_core]
        if outer_finite:
            ev_outer = lambda t, y, n=n: radius + y[1 + n] - (max(y[0], 0.0) ** (1 / (p + 1))
                                                             + np.exp(y[1:1 + n]).sum())
            ev_outer.terminal, ev_outer.direction = True, -1
            events.append(ev_outer)
        atol = np.full(y.size, rtol * size)
        # u = r^(p+1) to the accuracy wanted of r: du = (p+1) r^p dr
        atol[0] = rtol * (p + 1) * max(y[0], 0.0) ** (p / (p + 1)) * size
        atol[1:1 + n] = rtol
        sol = solve_ivp(rhs, (t, t_end), y, method="Radau", rtol=rtol, atol=atol, events=events, dense_output=True)
        if sol.status == -1:
            raise ConvergenceError(f"layer growth integration failed: {sol.message}")
        t_stop = sol.t[-1]
        for ts in save[(save > t - 1e-15 * t_end) & (save <= t_stop)]:
            ys = sol.sol(ts)
            record(ts, max(ys[0], 0.0) ** (1 / (p + 1)), np.exp(ys[1:1 + n]), ys[1 + n], markers_at(ts, ys))
        save = save[save > t_stop]
        if sol.status == 0 or save.size == 0:
            break

        y_end = sol.y[:, -1].copy()
        y_end[2 + n:] = markers_at(t_stop, y_end)
        d_layers = np.exp(y_end[1:1 + n])
        if sol.t_events[0].size:
            consumed.append((t_stop, phases.pop(0).name))
            r_new = max(y_end[0], 0.0) ** (1 / (p + 1)) + d_layers[0]
            y = np.concatenate([[r_new ** (p + 1)], y_end[2:1 + n], [y_end[1 + n]], y_end[2 + n:]])
        else:
            consumed.append((t_stop, phases.pop().name))
            r_last = max(y_end[0], 0.0) ** (1 / (p + 1)) + d_layers.sum()
            y = np.concatenate([y_end[:1 + n - 1], [r_last - radius], y_end[2 + n:]])
        t = t_stop
    return out_t, out_pos, out_mark, out_thick, consumed
