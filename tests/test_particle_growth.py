"""Growth of a particle with local equilibrium in a multicomponent matrix, against Chen, Jeppsson and Agren, Acta
Mater. 56 (2008) 1890: eq. (2) and its limits (p. 1891), and eqs. (9), (10) and (13) checked against Zener's exact
sphere solution and this package's sharp-interface solver. The paper's own example uses the TCFE4 and MOB2
commercial databases and is not reproduced."""

import functools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from pycalphad import Database
from scipy.optimize import brentq, root

from janus_kinetics import SolutionPhase
from janus_kinetics.thermo import R
from janus_kinetics.moving_boundary import Layer, LocalEquilibrium, solve_layers
from janus_kinetics.particle_growth import (
    _fixed_composition_batch,
    diffusion_distance_factor,
    fixed_composition_growth_rate,
    growth_parameter,
    particle_growth_rate,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_geometry import zener_alpha  # noqa: E402


def test_eq_2_is_zeners_sphere():
    """Eq. (2) is Zener's (1949) eq. (12) for a sphere with alpha = 2 lambda, R = alpha (D t)^(1/2), derived
    separately in test_geometry; and n -> 1 as Omega -> 0, n -> Omega (1 - Omega) / 3 as Omega -> 1 (p. 1891),
    the second followed to Omega = 1 - 1e-11, where eq. (2) cancels in floating point."""
    for omega in (1e-6, 1e-3, 0.05, 0.3, 0.7, 0.95):
        assert 2 * growth_parameter(omega) == pytest.approx(zener_alpha(3, omega), rel=1e-9)
    assert diffusion_distance_factor(1e-7) == pytest.approx(1.0, abs=1e-3)
    for omega in (0.999, 1 - 1e-6, 1 - 1e-8, 1 - 1e-11):
        assert diffusion_distance_factor(omega) == pytest.approx(omega * (1 - omega) / 3, rel=2e-3)
    # near Omega = 1 against eq. (2) evaluated through Laplace's continued fraction for erfc,
    # sqrt(pi) erfcx(x) = 1 / (x + K), K = (1/2) / (x + 1 / (x + (3/2) / (x + ...))), which gives
    # f = 2 lambda^2 K / (lambda + K) with no cancellation
    def cf(x, terms=400):
        k = 0.0
        for j in range(terms, 0, -1):
            k = (j / 2) / (x + k)
        return k

    for omega in (0.995, 0.9985, 0.999, 0.99999, 1 - 1e-7):
        lam = brentq(lambda x: 2 * x * x * cf(x) / (x + cf(x)) - omega, 3, 1e6, xtol=1e-12, rtol=1e-15)
        assert growth_parameter(omega) == pytest.approx(lam, rel=1e-9)
    # the vectorized factor (tabulated inverse) against the scalar root over the whole range
    om = np.concatenate([np.logspace(-11.9, -0.01, 300), 1 - np.logspace(-11.9, -0.31, 300), [0.4999, 0.5, 0.5001]])
    np.testing.assert_allclose(diffusion_distance_factor(om), [o / (2 * growth_parameter(o) ** 2) for o in om], rtol=1e-8)
    n = diffusion_distance_factor(np.linspace(0.01, 0.99, 99))
    assert np.all(np.diff(n) < 0)
    assert diffusion_distance_factor(-0.2) == 1.0  # no root of eq. (2): the steady-state value, a package choice
    with pytest.raises(ValueError, match="0 < Omega < 1"):
        growth_parameter(1.0)


T = 1000.0
KA, KB, KC = 0.1, 50.0, 30.0  # partition ratios x^P / x^I of the ideal phases


def _tdb(elements, d):
    lines = ["ELEMENT VA VACUUM 0 0 0 !"] + [f"ELEMENT {e} BLANK 1 0 0 !" for e in elements]
    lines += ["TYPE_DEFINITION % SEQ * !", "PHASE FCC % 1 1 !", f"CONSTITUENT FCC :{','.join(elements)}: !",
              "PHASE PPT % 1 1 !", f"CONSTITUENT PPT :{','.join(elements)}: !"]
    for e, k in zip(elements, (KA, KB, KC)):
        lines += [f"PARAMETER G(FCC,{e};0) 298.15 0; 6000 N !", f"PARAMETER G(PPT,{e};0) 298.15 -R*T*LN({k}); 6000 N !",
                  f"PARAMETER MQ(FCC&{e},*;0) 298.15 R*T*LN({d[e]}); 6000 N !",
                  f"PARAMETER MQ(PPT&{e},*;0) 298.15 R*T*LN(1E-25); 6000 N !"]
    return "\n".join(lines)


@functools.lru_cache(maxsize=None)
def _phases(elements: tuple, d: tuple):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(_tdb(elements, dict(zip(elements, d))))
        order = list(elements[1:]) + [elements[0]]
        return SolutionPhase(db, "FCC", order), SolutionPhase(db, "PPT", order)


def test_binary_growth_is_exact():
    """A-B with partition ratios 0.1 and 50 and equal tracer diffusivities (so the interdiffusivity is D): without
    capillarity v R is Zener's D alpha^2 / 2 for the supersaturation of the tie-line, x^I_B = 0.9/49.9. The direct
    form, eq. (13), carries x^I M (mu^M - mu^I) = x^I D ln(x^M / x^I) for the flux, so in this ideal dilute matrix it
    falls short of the exact rate by x^I ln(x^M/x^I) / (x^M - x^I), the price of writing the flux with interface
    quantities alone."""
    d = 1e-14
    matrix, particle = _phases(("A", "B"), (d, d))
    xi, xp = 0.9 / 49.9, 50 * 0.9 / 49.9
    for far in (0.03, 0.1, 0.3):
        st = particle_growth_rate(matrix, particle, [far], 1e-6, T, ([0.02], [0.8]), diffusivity=[[d]])
        assert st.matrix_interface[0] == pytest.approx(xi, rel=1e-9)
        assert st.particle_interface[0] == pytest.approx(xp, rel=1e-9)
        omega = (far - xi) / (xp - xi)
        assert st.velocity * st.radius == pytest.approx(d * zener_alpha(3, omega) ** 2 / 2, rel=1e-8)
        direct = particle_growth_rate(matrix, particle, [far], 1e-6, T, ([0.02], [0.8]), form="mobility")
        assert direct.velocity / st.velocity == pytest.approx(xi * np.log(far / xi) / (far - xi), rel=1e-6)


TERNARY = (1e-13, 1e-14, 4e-14)  # tracer diffusivities of A, B, C in the matrix


@functools.lru_cache(maxsize=None)
def _sharp_interface(far, cells=750):
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    dm = np.diag(TERNARY[1:])
    st = particle_growth_rate(matrix, particle, far, 1e-6, T, ([0.01, 0.013], [0.5, 0.4]), diffusivity=dm)
    k = st.velocity * st.radius
    r0, r1 = 0.1e-6, 2e-6
    t_end = (r1**2 - r0**2) / (2 * k)
    layers = [Layer(np.tile(st.particle_interface[:2], (20, 1)), np.diag([1e-25, 1e-25])),
              Layer(np.tile(np.asarray(far), (cells, 1)), dm)]
    cond = [LocalEquilibrium(particle, matrix, T, st.particle_interface[:2], st.matrix_interface[:2])]
    res = solve_layers([0.0, r0, 40 * r1], layers, cond, t_end, t_end / 800, save_times=t_end * np.linspace(0.6, 1, 3),
                       geometry="spherical")
    return st, res


@pytest.mark.slow
@pytest.mark.parametrize("far", [(0.02, 0.02), (0.012, 0.03)])
def test_ternary_growth_matches_the_sharp_interface_solver(far):
    """A-B-C, ideal, with B and C partitioned 50 and 30 times into the particle and C diffusing 4 times faster:
    each element gets its own factor (n = 0.79 and 0.89 for the first alloy). With a diagonal D and no capillarity
    every field is the binary one, so eq. (10) should be exact for growth from zero size; a sphere grown from
    0.1 to 2 um by solve_layers, with local equilibrium at its interface, reaches d(R^2)/dt = 2 v R within 0.1 %
    (0.05 % left whatever the grid, from its step-profile start) and the operating tie-line within 2e-6. With
    n = 1 for both, the two flux balances would ask for growth rates 20 % and 11 % below it."""
    st, res = _sharp_interface(far)
    k = st.velocity * st.radius
    r = res.interfaces[:, 0]
    assert (r[-1] ** 2 - r[-2] ** 2) / (res.times[-1] - res.times[-2]) == pytest.approx(2 * k, rel=1e-3)
    np.testing.assert_allclose(res.interface_right[-1, 0], st.matrix_interface[:2], atol=2e-6)
    np.testing.assert_allclose(res.interface_left[-1, 0], st.particle_interface[:2], atol=1e-4)
    assert np.all(st.factor < 0.92) and st.factor[0] < st.factor[1]
    steady = np.diag(TERNARY[1:]) @ (np.asarray(far) - st.matrix_interface[:2]) / (st.particle_interface[:2] - st.matrix_interface[:2])
    np.testing.assert_allclose(steady / k, st.factor, rtol=1e-6)


def test_cross_diffusion_enters_eq_10_column_by_column():
    """With off-diagonal D_ij the paper's sum divides each element's driving difference by its own factor n_j
    before the matrix acts, sum_j D_ij (c^M_j - c^I_j) / (n_j R); the returned state satisfies that, and not the
    variant dividing row by row, which would differ by 3 % here."""
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    far = np.array([0.012, 0.03])
    d = np.array([[1e-14, 0.4e-14], [-0.6e-14, 4e-14]])
    st = particle_growth_rate(matrix, particle, far, 1e-6, T, ([0.01, 0.013], [0.5, 0.4]), diffusivity=d)
    xi, xp = st.matrix_interface[:2], st.particle_interface[:2]
    lhs = st.velocity * st.radius * (xp - xi)
    np.testing.assert_allclose(lhs, d @ ((far - xi) / st.factor), rtol=1e-8)
    assert np.max(np.abs(lhs / ((d @ (far - xi)) / st.factor) - 1)) > 0.01


def test_capillarity_sets_the_critical_radius():
    """With sigma V_m on the particle the growth rate changes sign at R* = 2 sigma V_m / dG, dG the driving force
    of the parallel tangent to the far-field matrix (found here independently); the rate vanishes there within
    1e-6 of its size at 2 R*, and the operating tie-line is then the parallel tangent itself."""
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    far = np.array([0.02, 0.02])
    mu = matrix.chemical_potentials(far, T)

    def tangent(u):
        return particle.chemical_potentials(u[:2], T) - mu - u[2]

    sol = root(tangent, [0.5, 0.4, 0.0])
    xp, d_mu = sol.x[:2], sol.x[2]
    drive = -d_mu  # per mole of particle: sum_k x^P_k mu^M_k - G^P = -d_mu
    gamma, vm = 0.3, 7e-6
    r_star = 2 * gamma * vm / drive
    kw = dict(diffusivity=np.diag(TERNARY[1:]), gamma=gamma, molar_volume=vm)
    at = particle_growth_rate(matrix, particle, far, r_star, T, (far * 0.99, xp), **kw)
    big = particle_growth_rate(matrix, particle, far, 2 * r_star, T, ([0.01, 0.013], [0.5, 0.4]), **kw)
    small = particle_growth_rate(matrix, particle, far, 0.5 * r_star, T, (far * 0.99, xp), **kw)
    assert abs(at.velocity) < 1e-6 * big.velocity and big.velocity > 0 > small.velocity
    np.testing.assert_allclose(at.particle_interface[:2], xp, atol=1e-6)
    np.testing.assert_allclose(at.matrix_interface[:2], far, atol=1e-8)


def test_input_checks():
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    start = ([0.01, 0.013], [0.5, 0.4])
    with pytest.raises(ValueError, match="needs the matrix interdiffusion"):
        particle_growth_rate(matrix, particle, [0.02, 0.02], 1e-6, T, start)
    with pytest.raises(ValueError, match="2 x 2"):
        particle_growth_rate(matrix, particle, [0.02, 0.02], 1e-6, T, start, diffusivity=np.eye(3))
    with pytest.raises(ValueError, match="simplex"):
        particle_growth_rate(matrix, particle, [0.02, 0.02], 1e-6, T, ([0.01, 0.013], [0.7, 0.4]), diffusivity=np.eye(2))
    with pytest.raises(ValueError, match="one fewer"):
        particle_growth_rate(matrix, particle, [0.02], 1e-6, T, start, diffusivity=np.eye(2))
    with pytest.raises(ValueError, match="form"):
        particle_growth_rate(matrix, particle, [0.02, 0.02], 1e-6, T, start, form="other")


def test_fixed_composition_reproduces_the_free_particle():
    """A particle held at the composition the free solution gives its interface needs only the one equilibrium
    condition sum_i x^P_i mu^I_i = G^P + 2 sigma V_m / R; it then grows at the same rate with the same matrix
    interface, to 1e-8, in both forms and with capillarity."""
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    for far in ([0.02, 0.02], [0.012, 0.03]):
        for form in ("diffusivity", "mobility"):
            kw = dict(diffusivity=np.diag(TERNARY[1:]), form=form, gamma=0.2, molar_volume=7e-6)
            free = particle_growth_rate(matrix, particle, far, 1e-7, T, ([0.01, 0.013], [0.5, 0.4]), **kw)
            xp = free.particle_interface
            g = xp @ particle.chemical_potentials(xp[:2], T)
            held = fixed_composition_growth_rate(matrix, xp, g, far, 1e-7, T, **kw)
            assert held.velocity == pytest.approx(free.velocity, rel=1e-8)
            np.testing.assert_allclose(held.matrix_interface, free.matrix_interface, atol=1e-12)


def test_particle_too_small_for_any_interface_is_reported():
    """At 0.01 nm the capillary pressure, 2 sigma V_m / R = 0.28 MJ/mol, exceeds what any matrix composition can
    balance: the batch marks the particle unsolved (the precipitation solver then shrinks it at the extremal rate)
    and the single-particle call raises."""
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    xp = np.array([0.5, 0.4, 0.1])
    g = xp @ particle.chemical_potentials(xp[:2], T)
    v, _, _, _, _, done = _fixed_composition_batch(matrix, np.tile(xp, (2, 1)), [g, g], [1e-11, 1e-7],
                                                   np.array([0.02, 0.02, 0.96]), T, 0.2, 7e-6, "diffusivity",
                                                   np.diag(TERNARY[1:]), strict=False)
    assert list(done) == [False, True] and np.isnan(v[0]) and np.isfinite(v[1])
    with pytest.raises(RuntimeError, match="did not converge"):
        fixed_composition_growth_rate(matrix, xp, g, [0.02, 0.02], 1e-11, T, diffusivity=np.diag(TERNARY[1:]),
                                      gamma=0.2, molar_volume=7e-6)


@functools.lru_cache(maxsize=None)
def _coarsening_db():
    """Ideal A-B-C fcc with a B0.6C0.4 line compound (solubility product 0.01) and a solvent A diffusing 100 times
    slower than B, at 433 K."""
    d = {"A": 5e-22, "B": 5e-20, "C": 1e-20}
    lines = ["ELEMENT VA VACUUM 0 0 0 !"] + [f"ELEMENT {e} BLANK 1 0 0 !" for e in "ABC"]
    lines += ["TYPE_DEFINITION % SEQ * !", "PHASE FCC % 1 1 !", "CONSTITUENT FCC :A,B,C: !"]
    lines += [f"PARAMETER G(FCC,{e};0) 298.15 0; 6000 N !" for e in "ABC"]
    lines += [f"PARAMETER MQ(FCC&{e},*;0) 298.15 R*T*LN({d[e]}); 6000 N !" for e in "ABC"]
    lines += ["PHASE PPT % 2 0.6 0.4 !", "CONSTITUENT PPT :B:C: !", "PARAMETER G(PPT,B:C;0) 298.15 R*T*LN(0.01); 6000 N !"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(chr(10).join(lines))
        return db, SolutionPhase(db, "FCC", ["B", "C", "A"], mobility_db=db)


@functools.lru_cache(maxsize=None)
def _coarsening_with_local_equilibrium():
    """The coarsening case of test_coarsening (particles seeded at 5 nm) in _coarsening_db, grown by local
    equilibrium (mobility form)."""
    from janus_kinetics import NucleationSettings, PrecipitateThermo, solve_multicomponent_precipitation
    from janus_kinetics.multicomponent_precipitation import N_A
    from scipy.optimize import brentq

    temp = 433.15
    db, matrix = _coarsening_db()
    xp = np.array([0.6, 0.4, 0.0])
    x0 = 0.98 * np.array([0.012, 0.008, 0.98]) + 0.02 * xp
    phi_eq = brentq(lambda f: 0.6 * np.log((x0[0] - f * 0.6) / (1 - f)) + 0.4 * np.log((x0[1] - f * 0.4) / (1 - f))
                    - np.log(0.01), 0.0, 0.0395)
    x_eq = (x0 - phi_eq * xp) / (1 - phi_eq)
    radii = np.linspace(4.5e-9, 5.5e-9, 41)
    num = np.exp(-((radii - 5e-9) / 0.3e-9) ** 2)
    num *= phi_eq / (num @ (4 * np.pi / 3 * radii**3))
    times = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 3e8])
    settings = NucleationSettings(gamma=0.13, molar_volume=1.6e-29 * N_A, lattice_parameter=4.04e-10)
    res = solve_multicomponent_precipitation(
        matrix, PrecipitateThermo(db, "PPT", ["B", "C", "A"], composition=list(xp)), list(x0), temp, times[-1],
        settings, times, t_start=1.0, min_classes=100, max_classes=200,
        initial={"number": num, "radius": radii, "composition": np.tile(xp, (radii.size, 1))},
        growth="local_equilibrium", growth_form="mobility")
    return res, times, x_eq, xp, matrix, settings, temp


def test_precipitation_with_local_equilibrium_growth_coarsens_at_its_own_rate():
    """Linearized about the tie-line, eq. (13) with the single equilibrium condition gives the growth law
    dR/dt = (F - 2 sigma V_m / R) / (R Q), Q = sum_j a_j dx_j / (x_j M_j) over the solutes, with
    a_j = x^P_j - x^P_A x_j / x_A from Gibbs-Duhem; Philippe and Voorhees' asymptotics then give K = 8 sigma V_m / (9 Q).
    The run follows it within 3 % from 1e7 s. It does not follow the extremal law: SFFK's dissipation also charges
    the solvent's flux, sum_i dx_i^2 / (x_i M_i) over all elements, and with a solvent 100 times slower than B that
    puts its K 1.8 times lower; the two laws differ in whether the solvent's diffusion resists growth."""
    from janus_kinetics.coarsening import NUMBER_DENSITY_FACTOR

    res, times, x_eq, xp, matrix, settings, temp = _coarsening_with_local_equilibrium()
    mob = matrix.mobility(x_eq, temp)
    dx = xp - x_eq
    a = xp[:2] - xp[2] * x_eq[:2] / x_eq[2]
    k_le = 8 * settings.gamma * settings.molar_volume / (9 * np.sum(a * dx[:2] / (x_eq[:2] * mob[:2])))
    k_sffk = 8 * settings.gamma * settings.molar_volume / (9 * np.sum(dx**2 / (x_eq * mob)))
    rates = (np.diff(res.mean_radius**3) / np.diff(times))[2:]
    assert np.mean(rates) / k_le == pytest.approx(1.0, abs=0.03)
    decay = (np.diff(1 / res.number_density) / np.diff(times) * NUMBER_DENSITY_FACTOR * res.volume_fraction[1:])[2:]
    assert np.mean(decay) / k_le == pytest.approx(1.0, abs=0.03)
    assert k_le / k_sffk == pytest.approx(1.8, abs=0.1)


def test_local_equilibrium_growth_options_are_checked():
    from janus_kinetics import NucleationSettings, PrecipitateThermo, solve_multicomponent_precipitation
    from janus_kinetics.multicomponent_precipitation import N_A

    db, matrix = _coarsening_db()
    prec = PrecipitateThermo(db, "PPT", ["B", "C", "A"], composition=[0.6, 0.4, 0.0])
    settings = NucleationSettings(gamma=0.13, molar_volume=1.6e-29 * N_A, lattice_parameter=4.04e-10)
    alloy = [0.02, 0.016, 0.964]
    with pytest.raises(ValueError, match="growth is"):
        solve_multicomponent_precipitation(matrix, prec, alloy, 433.15, 10.0, settings, [10.0], growth="sharp")
    with pytest.raises(ValueError, match="growth is"):
        solve_multicomponent_precipitation(matrix, prec, alloy, 433.15, 10.0, settings, [10.0],
                                           growth="local_equilibrium", growth_form="flux")


def test_misfit_energy_acts_as_pressure():
    """A misfit energy per volume enters local equilibrium as the capillary pressure does: misfit 2 sigma / r at no
    sigma grows a particle exactly as sigma does at radius r."""
    matrix, particle = _phases(("A", "B", "C"), TERNARY)
    xp = np.array([0.5, 0.4, 0.1])
    g = xp @ particle.chemical_potentials(xp[:2], T)
    kw = dict(diffusivity=np.diag(TERNARY[1:]), molar_volume=7e-6)
    capillary = fixed_composition_growth_rate(matrix, xp, g, [0.02, 0.02], 2e-7, T, gamma=0.3, **kw)
    elastic = fixed_composition_growth_rate(matrix, xp, g, [0.02, 0.02], 2e-7, T, misfit=2 * 0.3 / 2e-7, **kw)
    assert elastic.velocity == pytest.approx(capillary.velocity, rel=1e-9)
    assert capillary.velocity < fixed_composition_growth_rate(matrix, xp, g, [0.02, 0.02], 2e-7, T, **kw).velocity


@pytest.mark.parametrize("form", ["mobility", "diffusivity"])
def test_solver_classes_grow_by_the_requested_law(form):
    """The precipitation solver's class rates are fixed_composition_growth_rate's, class by class, with the
    population's interfacial energy, misfit and form (and the matrix interdiffusion at the far field for eq. 10)."""
    from janus_kinetics.multicomponent_precipitation import N_A, NucleationSettings, Population, PrecipitateThermo, _PopulationState

    db, matrix = _coarsening_db()
    temp, xp = 433.15, np.array([0.6, 0.4, 0.0])
    prec = PrecipitateThermo(db, "PPT", ["B", "C", "A"], composition=list(xp))
    settings = NucleationSettings(gamma=0.13, molar_volume=1.6e-29 * N_A, lattice_parameter=4.04e-10, misfit=2e6)
    radii = np.array([1.2e-9, 3e-9, 8e-9])
    initial = {"number": np.full(3, 1e20), "radius": radii, "composition": np.tile(xp, (3, 1))}
    state = _PopulationState(Population(prec, settings, initial, growth="local_equilibrium", growth_form=form),
                             3, False, temp, matrix)
    x0 = np.array([0.0125, 0.0085, 0.979])
    mu0 = matrix.chemical_potentials(x0, temp)
    tracer = R * temp * matrix.mobility(x0, temp)
    state.rates(x0, mu0, tracer, temp, 10.0, 1e-3)
    g = prec.gm(xp, temp)
    d = matrix.interdiffusion_matrix(x0, temp)
    for r, v in zip(radii, state.rate):
        ref = fixed_composition_growth_rate(matrix, xp, g, x0[:2], r, temp, diffusivity=d, gamma=0.13,
                                            molar_volume=settings.molar_volume, misfit=2e6, form=form)
        assert v == pytest.approx(ref.velocity, rel=1e-6)
