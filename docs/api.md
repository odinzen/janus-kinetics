# API reference

Generated from the docstrings by `tools/gen_api.py`; do not edit by hand.

## `janus_kinetics`

### `R`

Constant, value `8.3145`.

## `janus_kinetics.coarsening`

### `def coarsening_rate_constant(dc: 'Sequence[float]', mobility, molar_volume: 'float', gamma: 'float') -> 'float'`

K in <R>^3 = K t (m^3/s), eq. (42). ``dc`` is C^beta - C^alpha over the independent components, ``mobility``
the matrix M with D = M G (m^2 mol / (J s)), ``molar_volume`` V_m of the precipitate (m^3/mol) and ``gamma`` the
interfacial energy (J/m^2). For a diagonal M this is eq. (47).

### `def critical_radius(dc: 'Sequence[float]', hessian, supersaturation: 'Sequence[float]', molar_volume: 'float', gamma: 'float') -> 'float'`

R_c = 2 sigma V_m / (dC^T G dC^inf), eq. (29): the radius that neither grows nor shrinks.

### `def matrix_supersaturation(t, dc: 'Sequence[float]', mobility, hessian, molar_volume: 'float', gamma: 'float') -> 'np.ndarray'`

C^inf - C^alpha at times ``t`` (s), eq. (43); one row per time.

### `def number_density(t, rate_constant: 'float', volume_fraction: 'float')`

Particles per volume at time ``t``, eq. (44), with the LSW factor 0.2113 the paper rounds to 0.21.

### `def precipitate_composition_shift(t, dc: 'Sequence[float]', mobility, hessian, precipitate_hessian, molar_volume: 'float', gamma: 'float', partial_volume_difference: 'Sequence[float] | None' = None) -> 'np.ndarray'`

<C^beta> - C^beta_eq of the average particle at times ``t``, eq. (45), one row per time. The term in
``precipitate_hessian`` turns the shift off the tie-line unless the two Hessians agree;
``partial_volume_difference`` is V_i^beta - V_1^beta for i = 2..N (zero when omitted).

## `janus_kinetics.couple_analysis`

### `def interdiffusion_from_profile(z, x, t: 'float', molar_volume=None) -> 'np.ndarray'`

D~ (m^2/s) at every point of the profile ``x`` (mole fraction of one element), eq. 3.

    D~(Y*) = V_m/(2t) (dz/dY)* [ (1 - Y*) int_{-inf}^{z*} Y/V_m dz + Y* int_{z*}^{+inf} (1 - Y)/V_m dz ]

with Y = (N - N^-)/(N^+ - N^-). ``molar_volume`` (m^3/mol) is a constant, or
an array along z; it cancels when constant (eq. 2). Values where the profile
is flat (the undisturbed ends) are meaningless, so read D~ inside the zone.

### `def intrinsic_from_profile(z, x, t: 'float', z_marker: 'float', molar_volume=None) -> 'float'`

Intrinsic diffusivity D_i (m^2/s) of the element whose mole fraction is ``x``, at the Kirkendall plane, eq. 12.

    D_i(z_K) = 1/(2t) (dz/dC_i)_K [ N_i^+ int_{-inf}^{z_K} Y/V_m dz - N_i^- int_{z_K}^{+inf} (1 - Y)/V_m dz ]

with C_i = N_i / V_m. Call it with N_A and with 1 - N_A for the two elements of
a binary. ``z_marker`` is the measured marker position in the same frame as z.

## `janus_kinetics.datasets`

### `def campbell2002_fcc_ni() -> 'Database'`

Atomic mobilities in fcc Ni-Al-Co-Cr-Hf-Mo-Re-Ta-Ti-W, for Ni-base superalloys.

Table 2 of Campbell, Boettinger and Kattner, Acta Mater. 50 (2002) 775, with
their estimation rule for end members the table does not list (p. 778). The
authors assessed it against Kattner's Ni-superalloy thermodynamics, so other
descriptions shift the chemical diffusivities somewhat (their Tables 5 and 7
show the size of that effect).

### `def helander1999_bcc_alni() -> 'tuple[Database, dict]'`

Mobilities of Al and Ni in B2-ordered bcc Al-Ni: the disordered database and the ordering terms.

Appendix A of Helander and Agren, Acta Mater. 47 (1999) 1141, for
:class:`janus_kinetics.B2Phase`. The disordered parameters are for the
hypothetical bcc A2 phase (the same for pure Al and pure Ni by their
assumption, p. 1147). The authors fitted them with the Al-Ni thermodynamics of
Ansara et al. (1997); Al-Fe is not shipped because its magnetic term (their
eq. 8) is not usable as printed.

### `def jonsson1995_bcc_crfeni() -> 'Database'`

Atomic mobilities of Cr, Fe and Ni in bcc Cr-Fe-Ni, paramagnetic part.

Transcribed from the Appendix (p. 1421) of B. Jonsson, ISIJ Int. 35 (1995)
1415. It holds mobility (MQ/MF) parameters only, to be combined with a
thermodynamic description through ``SolutionPhase(..., mobility_db=...)``. The ferromagnetic
factor of the paper's eq. (3) is not included. It is largest below the Curie
temperature but still lowers Fe-rich mobilities above it: against the paper's
Fig. 5 the Fe tracer diffusivity in pure bcc Fe comes out 0.12 decades too high
at 1250 C and 0.38 at 960 C, while Cr-rich alloys agree within 0.07.

### `def kjellqvist2008_cro() -> 'Database'`

Thermodynamics of Cr-O between Cr and Cr2O3: bcc Cr with interstitial O, corundum
(Cr+2,Cr+3)2(Cr+3,Va)1(O-2)3 and O2 gas.

The Cr-O part of Table 5 of Kjellqvist, Selleby and Sundman, Calphad 32 (2008)
577, for :func:`janus_kinetics.oxides.chromia`. Hallstrom et al. (2013) fitted
their Cr2O3 mobilities with Taylor and Dinsdale's Cr-O description, which this
one builds on; their oxygen tracer curves (Fig. 7) come out within 0.07 decades
with it. The antiferromagnetic and structure factors are not printed and are
the usual ones (see the file header); they change nothing above 600 K that
matters here.

## `janus_kinetics.dendrite_tip`

### `class DendriteTip(velocity: 'float', gradient: 'float', radius: 'float', peclet: 'np.ndarray', liquid: 'np.ndarray', solid: 'np.ndarray', partition: 'np.ndarray', slopes: 'np.ndarray', liquidus_temperature: 'float', curvature_undercooling: 'float', temperature: 'float') -> None`

A dendrite tip at ``velocity`` (m/s) in a liquid gradient ``gradient`` (K/m): its radius (m), the solutes'
Peclet numbers, the liquid and solid at the tip (independent mole fractions), the partition ratios and liquidus
slopes used, the liquidus at the tip liquid T_L(C*), the curvature undercooling 2 Gamma / R and the tip
temperature T* = T_L(C*) - 2 Gamma / R (K).


### `class Liquidus(temperature: 'float', solid: 'np.ndarray', partition: 'np.ndarray', slopes: 'np.ndarray') -> None`

The liquidus at one liquid composition: its temperature (K), the solid on it (independent mole fractions),
the partition ratios k_i = x_S,i / x_L,i and the slopes m_i = dT_L / dx_L,i (K per mole fraction, the other
independent fractions held).


### `def dendrite_tip(velocity: 'float', gradient: 'float', alloy: 'Sequence[float]', partition: 'Sequence[float]', slopes: 'Sequence[float]', diffusivities, gibbs_thomson: 'float', alloy_liquidus: 'float') -> 'DendriteTip'`

The tip of eqs. (2)-(5) for a linear phase diagram: constant partition ratios and liquidus slopes, so that
T_L(C*) = T_L(C0) + sum_i m_i (C*_i - C0_i) with ``alloy_liquidus`` = T_L(C0). ``diffusivities`` are the
solutes' liquid diffusivities (m^2/s, a scalar for all), ``gibbs_thomson`` Gamma in K m.

### `def dendrite_tip_calphad(solid: 'SolutionPhase', liquid: 'SolutionPhase', velocity: 'float', gradient: 'float', alloy: 'Sequence[float]', diffusivities, gibbs_thomson: 'float', temperature: 'float', tol: 'float' = 1e-10, max_iter: 'int' = 200) -> 'DendriteTip'`

The tip of eqs. (2)-(5) with k_i and m_i from the solid-liquid equilibrium at T_L(C*), iterated with the
tip equations until C* settles, as the paper does (p. 526). ``temperature`` is a guess of the alloy's liquidus;
compositions are independent mole fractions in the phases' element order.

### `def ivantsov(peclet)`

Iv(P) = P exp(P) E1(P), the supersaturation a paraboloid needle growing at Peclet number P needs (eq. 2).

### `def liquidus(solid: 'SolutionPhase', liquid: 'SolutionPhase', x_liquid: 'Sequence[float]', temperature: 'float', x_solid: 'Sequence[float] | None' = None, tol: 'float' = 1e-12, max_iter: 'int' = 100) -> 'Liquidus'`

The temperature at which ``liquid`` of composition ``x_liquid`` (independent mole fractions, the phases'
element order) is in equilibrium with ``solid``: every element's chemical potential equal in both, solved by
Newton from ``temperature`` and ``x_solid`` (the liquid composition by default). The slopes follow from the
same Jacobian by implicit differentiation.

## `janus_kinetics.diffusivity`

### `def intrinsic_matrix(x: 'np.ndarray', mobility: 'np.ndarray', dmu: 'np.ndarray') -> 'np.ndarray'`

Intrinsic (lattice-frame) diffusivities D^I_kj = x_k M_k dmu_k/dx_j, shape (..., n, n-1), m^2/s.

The lattice-fixed flux of the vacancy mechanism is J'_k = -c_k M_k grad mu_k
(Andersson and Agren, J. Appl. Phys. 72 (1992) 1350, eqs. (58) and (62)). At
constant molar volume this is J'_k = -sum_j D^I_kj dc_j/dz. In a binary the two
rows are Darken's intrinsic diffusivities D_A and -D_B (van Loo, Prog. Solid
State Chem. 20 (1990) 47, eqs. (9)-(10)).

### `def lattice_velocity(intrinsic: 'np.ndarray', dx_dz: 'np.ndarray') -> 'np.ndarray'`

Velocity of the lattice (inert markers) relative to the number-fixed frame, m/s.

The two frames are related by J_k = J'_k + c_k v (van Loo, Prog. Solid State
Chem. 20 (1990) 47, eq. (4)); summing over k with sum_k J_k = 0 gives
v = -V_m sum_k J'_k = sum_k sum_j D^I_kj dx_j/dz. For a binary this is
v = (D_A - D_B) dx_A/dz, van Loo's eq. (9) at equal partial volumes.
``dx_dz`` holds the independent gradients, shape (..., n-1).

### `def number_fixed_matrix(x: 'np.ndarray', mobility: 'np.ndarray', dmu: 'np.ndarray') -> 'np.ndarray'`

Interdiffusion matrix in the number-fixed frame, shape (..., n-1, n-1), m^2/s.

D_kj = sum_i (delta_ik - x_k) x_i M_i dmu_i/dx_j, the vacancy-mechanism result
of Andersson and Agren, J. Appl. Phys. 72 (1992) 1350, eqs. (59)-(60), quoted
as eq. (2) of Jonsson, ISIJ Int. 35 (1995) 1415. Fluxes follow J_k = -sum_j D_kj dc_j/dz
with c_j = x_j / V_m and a constant molar volume.

### `def to_concentrations(x: 'np.ndarray', partial_volumes: 'np.ndarray') -> 'np.ndarray'`

Independent concentrations c_k = x_k / V_m (mol/m^3) from all n mole fractions, V_m = sum_i x_i V_i.

### `def to_mole_fractions(c: 'np.ndarray', partial_volumes: 'np.ndarray') -> 'np.ndarray'`

All n mole fractions from the independent concentrations, closing sum_i V_i c_i = 1.

### `def volume_fixed_lattice_velocity(x: 'np.ndarray', mobility: 'np.ndarray', dmu: 'np.ndarray', partial_volumes: 'np.ndarray', dc_dz: 'np.ndarray') -> 'np.ndarray'`

Lattice (marker) velocity relative to the volume-fixed frame, m/s.

With sum_k V_k J_k = 0 in that frame and J_k = J'_k + c_k v (van Loo, Prog.
Solid State Chem. 20 (1990) 47, eqs. 4 and 6), v = -sum_k V_k J'_k, where the
lattice flux is J'_k = -c_k M_k dmu_k/dz. In a binary this is van Loo's eq. 9,
v = V_A (D_A - D_B) dc_A/dz. ``dc_dz`` holds the independent concentration
gradients (mol/m^4), shape (..., n-1).

### `def volume_fixed_matrix(x: 'np.ndarray', mobility: 'np.ndarray', dmu: 'np.ndarray', partial_volumes: 'np.ndarray') -> 'np.ndarray'`

Interdiffusion matrix in the volume-fixed frame, shape (..., n-1, n-1), m^2/s.

Andersson and Agren, J. Appl. Phys. 72 (1992) 1350, eqs. (28)-(29), (40),
(50), (54) and (58), kept with general partial volumes rather than their later
equal-volume approximation. With constant partial molar volumes V_i, c_i = x_i / V_m and
sum_i V_i c_i = 1, so c_n is dependent:

    D^V_kj = sum_i (delta_ik - c_k V_i) c_i M_i dmu_i/dc_j

where dmu_i/dc_j follows from dx_k/dc_j = V_m [delta_kj - x_k (1 - V_j/V_n)].
The fluxes then satisfy sum_k V_k J_k = 0 and J_k = -sum_j D^V_kj dc_j/dz.
Equal partial volumes recover :func:`number_fixed_matrix`.

## `janus_kinetics.electromigration`

### `class LineEnd(kind: 'str' = 'blocked', width: 'float | None' = None, stress: 'float' = 0.0) -> None`

Condition at an end of the line: ``blocked`` (no atomic flux), a ``void`` from which atoms flow
as Korhonen et al.'s eq. (14) gives (``width`` w, the line width), or a ``reservoir`` held at
``stress`` (a bond pad or a wide line, say).


### `class StressResult(times: 'np.ndarray', x: 'np.ndarray', stress: 'np.ndarray', steps: 'int') -> None`

StressResult(times: 'np.ndarray', x: 'np.ndarray', stress: 'np.ndarray', steps: 'int')


### `def blech_product(stress_difference: 'float', effective_charge: 'float', resistivity: 'float', atomic_volume: 'float') -> 'float'`

Threshold current density times length (A/m), Blech's eq. (6): (jL)_c = Omega Delta sigma /
(|Z*| e rho), for the largest stress difference Delta sigma (Pa) the line sustains between its ends.

### `def drift_velocity(current_density: 'float', length: 'float', diffusivity: 'float', temperature: 'float', effective_charge: 'float', resistivity: 'float', atomic_volume: 'float', stress_difference: 'float') -> 'float'`

Net drift velocity (m/s) of a line of the given length, Blech's eqs. (1) and (3)-(4): the wind
drives atoms at v = (D/kT) e |Z*| rho j and the stress difference between the ends drives them back
at (D/kT) Omega Delta sigma / l; below the threshold of eq. (5) nothing moves.

### `def finite_line_stress(x, t: 'float', length: 'float', kappa: 'float', gradient: 'float', terms: 'int' = 400)`

Stress in a line blocked at both ends, Korhonen et al. eq. (13), summed over ``terms`` modes
m_n = (2n + 1) pi.

### `def semi_infinite_stress(x, t, kappa: 'float', gradient: 'float')`

Stress in a semi-infinite line blocked at x = 0 from t = 0, Korhonen et al. eq. (9); at the
block it is 2 G sqrt(kappa t / pi), eq. (10). ``x`` and ``t`` broadcast; the line starts stress-free.

### `def solve_line_stress(length: 'float', n_cells: 'int', gradient: 'float', kappa: 'float | Callable[[np.ndarray], np.ndarray]', t_end: 'float', dt: 'float', left: 'LineEnd' = LineEnd(kind='blocked', width=None, stress=0.0), right: 'LineEnd' = LineEnd(kind='blocked', width=None, stress=0.0), save_times: 'Sequence[float] | None' = None, initial: 'np.ndarray | float' = 0.0, yield_stress: 'float | None' = None, tolerance: 'float | None' = None, picard_tol: 'float' = 1e-10, max_picard: 'int' = 50) -> 'StressResult'`

Integrate Korhonen et al.'s eq. (8) along a line of ``length`` (m) by backward Euler on ``n_cells``
equal cells. ``kappa`` is constant or a function of the stress, such as their eq. (18) with the
activation volumes neglected, kappa(sigma) = kappa_0 exp(Omega sigma / kT); then each step iterates
until the stress changes by less than ``picard_tol`` times its largest value. ``tolerance`` (Pa) adapts the steps as in :func:`solve_planar`.

``yield_stress`` caps |sigma| by ideal plasticity (their section III, eq. 11): a yielded zone holds
the yield stress and exchanges no flux with the rest of the line, and grows as the stress beside it
reaches yield. Loading is taken as monotonic: a yielded cell does not unload (after a current
reversal, say).

### `def steady_stress(x, length: 'float', gradient: 'float')`

Blech's steady state of a line blocked at both ends, Korhonen et al. eq. (12).

### `def stress_diffusivity(diffusivity: 'float', modulus: 'float', atomic_volume: 'float', temperature: 'float') -> 'float'`

kappa = D_a B Omega / kT in m^2/s (Korhonen et al. eq. 8).

### `def void_line_stress(x, t: 'float', length: 'float', kappa: 'float', gradient: 'float', width: 'float', terms: 'int' = 400)`

Stress in a line with a void at x = 0 and a block at x = L, Korhonen et al. eqs. (15)-(16), with
alpha = L / w and beta_n the roots of beta tan(beta) = alpha. At the void the flux follows their
eq. (14), so d sigma/dx + G = sigma / w there. The leading term printed as -G xi is taken as -G L xi:
it must carry a length, and only -G x vanishes with the series at t = 0 and remains as the steady
state they describe (compressive except at the void, where the stress is zero).

### `def wind_stress_gradient(resistivity: 'float', current_density: 'float', effective_charge: 'float', atomic_volume: 'float') -> 'float'`

G = E q* / Omega = rho j |Z*| e / Omega in Pa/m (Korhonen et al. p. 3792), from rho in ohm m,
j in A/m^2 and Omega in m^3.

## `janus_kinetics.errors`

### `class ConvergenceError`

The nonlinear (Picard) iteration of a time step did not converge.


### `class MissingParameterError`

A mobility parameter needed for a well-defined model is absent from the database.


### `class UnstableCompositionError(compositions: 'np.ndarray', eigenvalues: 'np.ndarray', elements: 'tuple[str, ...]')`

The Gibbs energy curvature is not positive definite at some composition.

Such a composition lies inside a spinodal, which sits inside a miscibility
gap or two-phase field. A single-phase diffusion model has no physical
meaning there, so the calculation stops instead of patching the curvature.
Model the problem with a multiphase (moving-boundary or dispersed-phase)
model instead.


### `class UnsupportedPhaseError`

The phase constitution is outside what this version can model.


## `janus_kinetics.grain_boundary`

### `class FisherBoundary(d_lattice: 'float', d_boundary: 'float', delta: 'float') -> None`

A grain boundary of thickness ``delta`` (m) and diffusivity ``d_boundary`` in a
grain of diffusivity ``d_lattice`` (both m^2/s).

- `concentration(self, x, y, t) -> 'np.ndarray'`: Eq. (8): C at distance x from the boundary and depth y after time t, surface at 1.
- `penetration_slope(self, t) -> 'float'`: Eq. (14): d log10(a) / dy of the layer-averaged concentration, in 1/m.
- `scaled(self, x, y, t) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]'`: (x1, y1, t1) of eq. (5).
- `solve(self, t: 'float', **grid) -> 'FisherSolution'`: Numerical solution of eqs. (3)-(4) at time t (s); ``grid`` is passed to :func:`solve_fisher`.

### `def boundary_product_from_slope(slope: 'float', d_lattice: 'float', t: 'float') -> 'float'`

D' delta (m^3/s) from a measured penetration slope d log10(a)/dy (1/m), inverting eq. (14).

The classic reading of a radiotracer sectioning experiment; Fisher's eq. (15)
gives D / (D' delta) = 21.3 cm^-1 for silver at 478 C this way.

### `def harrison_regime(d_lattice: 'float', grain_radius: 'float', boundary_width: 'float', t: 'float', margin: 'float' = 10.0) -> 'str'`

Harrison's kinetic regime, Trans. Faraday Soc. 57 (1961) 1191: 'A' when lattice diffusion crosses a grain
within the anneal, a_g^2 / D_g << t (his eq. 1), so the solid diffuses as a whole with Hart's average
D_c = f D' + (1 - f) D (eq. 3; :class:`ShortCircuit`), f = 3 delta / a_g for spherical grains of radius a_g;
'C' when lattice diffusion does not leave the boundary layer, (D_g t)^(1/2) << delta (eq. 5), so only the
boundaries carry material; 'B' between, where Whipple's solution and :func:`le_claire_boundary_product`
apply. Harrison leaves '<<' unquantified (and calls eq. 1 perhaps far too strict); here it means a factor
``margin``, a choice of this package.

### `def le_claire_boundary_product(slope: 'float', d_lattice: 'float', t: 'float', coefficient: 'float' = 0.78) -> 'float'`

D' delta (m^3/s) from the slope d ln(c_bar) / d y^(6/5) (m^-6/5) of a sectioning profile, Le Claire's
eq. (15): D' delta = (4 D / t)^(1/2) (coefficient / |slope|)^(5/3), the coefficient being
-d ln(c_bar) / d (eta beta^(-1/2))^(6/5), 'for practical purposes constant at 0.78 if beta is large'
(p. 356; Whipple's solution, beta above about 10). This is the familiar 1.322 (D/t)^(1/2) form. For
Suzuoka's thin-source condition Le Claire gives 0.72 beta^0.008 (eq. 16).

### `def solve_fisher(t1: 'float', x_max: 'float | None' = None, y_max: 'float | None' = None, dx: 'float' = 0.5, dy: 'float' = 0.25, steps: 'int' = 400) -> 'FisherSolution'`

Solve the scaled eqs. (3a)-(4a) of Fisher (1951) without his approximation.

Backward Euler in t1 on a uniform grid; the slab takes (dC/dx1)_0 from the
second-order one-sided difference his eq. (3b) uses, and each grain row only
diffuses in x1, as eq. (4a) states. The surface row of the slab is held at 1.

### `def suzuoka_coefficient(beta: 'float') -> 'float'`

-d ln(c_bar) / d (eta beta^(-1/2))^(6/5) for Suzuoka's condition, 0.72 beta^0.008 (Le Claire eq. 16), to use in
:func:`le_claire_boundary_product` when the diffusant was deposited as a thin layer.

### `def suzuoka_layer_integral(eta: 'float', beta: 'float') -> 'float'`

The boundary contribution to the layer average for Suzuoka's thin-source condition (Trans. Jap. Inst. Met.
2 (1961) 25: a deposit of fixed amount, its surface flux free), without its constant factor. Le Claire (p. 356)
obtains it from Whipple's eq. (13) by the operation d/d eta, so it is -d/d eta of
:func:`whipple_layer_integral`: int_1^inf (eta^2 / 2s - 1) exp(-eta^2 / 4s) (exp(-V^2) / pi^(1/2) - V erfc V)
s^(-3/2) ds. Near the surface it is negative, the boundary drawing material from the lattice (Suzuoka p. 27):
below eta beta^(-1/2) of about 1.7 at beta = 1 and 0.7 at beta = 10. At beta = 1 it then still rises to a peak
near 3, inside the usual fitting range of 2-10.

### `def whipple_beta(d_lattice: 'float', d_boundary: 'float', delta: 'float', t: 'float') -> 'float'`

beta = (D'/D) (delta/2) / (D t)^(1/2), Le Claire, Brit. J. Appl. Phys. 14 (1963) 351, eq. (5).

### `def whipple_concentration(eta: 'float', xi: 'float', beta: 'float') -> 'float'`

c / c_0 of Whipple's exact solution (Phil. Mag. 45 (1954) 1225) as Le Claire writes it, his eq. (7),
for D'/D large enough that the integral may run to infinity: at reduced depth eta = y / (D t)^(1/2) and
distance from the slab xi = (x - delta/2) / (D t)^(1/2), from a surface held at c_0. The first term is
direct lattice diffusion, the second the supply down the boundary.

### `def whipple_layer_integral(eta: 'float', beta: 'float') -> 'float'`

The boundary contribution to the layer-averaged concentration of a sectioning experiment, Le Claire's
eq. (13) without its constant factor: eta int_1^inf exp(-eta^2 / 4 s) (exp(-V^2) / pi^(1/2) - V erfc V)
s^(-3/2) ds, V = (s - 1) / (2 beta). Only its logarithmic slope with depth is used.

## `janus_kinetics.heat_treatment`

### `class HeatTreatableAlloy(c0: 'float', cp: 'float', cs: 'float', qs: 'float', d0: 'float', qd: 'float', j0: 'float', a0: 'float', gamma: 'float', molar_volume: 'float', growth_margin: 'float' = 0.05) -> None`

The inputs of Myhr and Grong's Table 2: solute content of alloy ``c0`` and precipitate ``cp``, solvus
prefactor ``cs`` and enthalpy ``qs`` (J/mol, eq. 17), diffusivity prefactor ``d0`` (m^2/s) and activation
energy ``qd`` (J/mol), nucleation prefactor ``j0`` (1/m^3 s) and barrier parameter ``a0`` (J/mol, eq. 2),
interfacial energy ``gamma`` (J/m^2) and precipitate molar volume ``molar_volume`` (m^3/mol).

- `critical_radius(self, c: 'float', T: 'float') -> 'float'`: r* of eq. (6); infinite when the matrix is not supersaturated.
- `equilibrium_fraction(self, T)`: Lever-rule volume fraction (C_0 - C_e) / (C_p - C_e), zero above the solvus.
- `growth_rate(self, radius, c: 'float', T: 'float')`: dr/dt of eqs. (4)-(5). A particle so small that C_i reaches C_p, below
- `interface_content(self, radius, T)`: C_i of eq. (5).
- `nucleation_rate(self, c: 'float', T: 'float') -> 'float'`: j of eq. (3), zero when the matrix is not supersaturated.
- `solubility(self, T)`: C_e, eq. (17).

### `class HeatTreatmentResult(radii: 'np.ndarray', widths: 'np.ndarray', times: 'np.ndarray', temperature: 'np.ndarray', distribution: 'np.ndarray', matrix_content: 'np.ndarray', nucleation: 'np.ndarray', critical_radius: 'np.ndarray', steps: 'int') -> None`

``distribution[k]`` is N (1/m^4) at ``times[k]`` on the class radii ``radii`` with widths ``widths``.

- `state(self, k: 'int' = -1) -> 'np.ndarray'`: The distribution at save ``k``, to start another treatment from.

### `def aa6082(gamma: 'float' = 0.2) -> 'HeatTreatableAlloy'`

Myhr and Grong's Table 2: Mg in an AA 6082 alloy precipitating as Mg2Si, in wt% Mg.

### `def log_normal_distribution(radii, mean_radius: 'float', volume_fraction: 'float', spread: 'float') -> 'np.ndarray'`

Myhr and Grong's eq. (18), normal in ln r^2 with standard deviation ``spread`` (their b), placed so that the
number-mean radius is ``mean_radius`` (median exp(-b^2 / 8) below it) and scaled to ``volume_fraction``.

### `def lsw_distribution(radii, mean_radius: 'float', volume_fraction: 'float') -> 'np.ndarray'`

The Lifshitz-Slyozov-Wagner distribution N (1/m^4) with the given mean radius, on the class radii, scaled
to ``volume_fraction``.

### `def solve_heat_treatment(alloy: 'HeatTreatableAlloy', temperature: 'Temperature', t_end: 'float', radii=None, save_times=None, initial=None, dt: 'float' = 0.001, tolerance: 'float' = 0.02, courant: 'float' = 1.0, max_steps: 'int' = 2000000) -> 'HeatTreatmentResult'`

Run the Myhr-Grong model from t = 0 to ``t_end`` (s) along ``temperature`` (K, constant, schedule or
callable). ``radii`` are the class radii (default :func:`uniform_radii` up to 20 nm); ``initial`` a starting
distribution N on them (1/m^4), for instance :meth:`HeatTreatmentResult.state` of an earlier treatment.
Steps are also kept below ``courant`` class widths of travel for growing classes: the implicit upwind
scheme is stable at any step, but longer steps smear the distribution over many classes.
Raises RuntimeError if particles reach the largest class.

### `def stretched_radii(step: 'float', uniform_until: 'float', ratio: 'float', r_max: 'float') -> 'np.ndarray'`

Radii ``step`` apart up to ``uniform_until``, then each ``ratio`` times the last up to ``r_max``: fine
classes where particles nucleate and dissolve, few where they have coarsened.

### `def uniform_radii(r_max: 'float', classes: 'int' = 1000) -> 'np.ndarray'`

Class radii i dr, i = 1..classes, as in the paper (first wall at dr / 2).

## `janus_kinetics.homogenization`

### `class LocalEquilibriumTable(db: 'Database', phases: 'dict[str, SolutionPhase]', T: 'float', averaging: 'Averaging' = 'hashin_shtrikman_lower', spacing: 'float | None' = None, pressure: 'float' = 101325.0)`

Chemical potentials and Gamma* at local equilibrium, tabulated over composition.

``phases`` maps pycalphad phase names to :class:`SolutionPhase` objects that
supply each phase's mobilities; all must list the same elements in the same
order. ``spacing`` is the grid step in mole fraction (default 0.0025 for a
binary, 0.02 for a ternary). Compositions outside the tabulated range raise
instead of being extrapolated, so pure-element ends need a finer spacing.

- `chemical_potentials(self, x) -> 'np.ndarray'`: mu_k at local equilibrium, J/mol, shape (M, n); x holds independent fractions (M, n-1).
- `dmu_dx(self, x) -> 'np.ndarray'`: Slopes of the interpolated potentials, shape (M, n, n-1).
- `kinetic_factor(self, x) -> 'np.ndarray'`: Gamma*_k in m^2 mol / (J s), shape (M, n).
- `phase_fractions(self, x) -> 'dict[str, np.ndarray]'`: Volume (mole) fraction of each phase at local equilibrium, keyed by phase name.

### `class OnDemandEquilibriumTable(db: 'Database', phases: 'dict[str, SolutionPhase]', T: 'float', averaging: 'Averaging' = 'hashin_shtrikman_lower', spacing: 'float' = 0.01, pressure: 'float' = 101325.0)`

Local equilibrium computed only where a simulation goes, for any number of elements.

Larsson and Hoglund (2009, section 2.5): composition space is divided into a
grid of step ``spacing``, equilibrium is calculated at a grid node only when a
composition near it is first requested, and node results are stored under the
node's integer index (their hash table) for reuse. Between nodes, values are
interpolated linearly on the Kuhn simplex of the surrounding grid cell, so the
potentials are continuous and :meth:`dmu_dx` is the exact slope of that
simplex; in a binary and on grid lines this equals the full table of
:class:`LocalEquilibriumTable`. Only one temperature is held, as in the paper,
where stored data are dropped when T changes. ``calculations`` counts the
equilibria made. Nodes need every mole fraction, the dependent one included,
to be at least one step from zero.

- `chemical_potentials(self, x) -> 'np.ndarray'`: mu_k at local equilibrium, J/mol, shape (M, n); x holds independent fractions (M, n-1).
- `dmu_dx(self, x) -> 'np.ndarray'`: Exact slopes of the interpolated potentials on each point's simplex, shape (M, n, n-1).
- `kinetic_factor(self, x) -> 'np.ndarray'`: Gamma*_k in m^2 mol / (J s), shape (M, n), from ln Gamma* interpolated linearly.
- `phase_fractions(self, x) -> 'dict[str, np.ndarray]'`: Mole fraction of each phase at local equilibrium, keyed by phase name.

### `class TemperatureTable(db: 'Database', phases: 'dict[str, SolutionPhase]', temperatures, **table_options)`

Local-equilibrium tables at several temperatures, for anneals along a schedule.

Holds one :class:`LocalEquilibriumTable` per temperature in ``temperatures``
(binaries only, to keep the cost to one pycalphad grid per temperature). At an
intermediate T, chemical potentials are interpolated linearly in T and ln Gamma*
linearly in 1/T, which is exact for Arrhenius mobilities. Space the
temperatures so that phase boundaries move little between neighbours.


### `def average_kinetic_factor(gamma: 'np.ndarray', fraction: 'np.ndarray', how: 'Averaging') -> 'np.ndarray'`

Gamma* from per-phase Gamma (P, n) and volume fractions (P,), shape (n,).

Larsson and Hoglund (2009): Wiener upper and lower bounds, eqs. (6)-(7), and the
Hashin-Shtrikman bounds of eqs. (8)-(11), where the reference phase alpha is the
one with the lowest (lower bound) or highest (upper bound) Gamma_k present.

### `def solve_homogenized(z_faces: 'np.ndarray', x0: 'np.ndarray', table: 'LocalEquilibriumTable | OnDemandEquilibriumTable | TemperatureTable', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, tol: 'float' = 1e-10, max_iter: 'int' = 60, temperature: 'Temperature | None' = None, ideal_flux: 'float' = 0.0, geometry: 'Geometry' = 'planar', tolerance: 'float | None' = None) -> 'DiffusionResult'`

Integrate multiphase diffusion with the homogenization model in a plate (default), a long
cylinder or a sphere (``geometry``, radii and centre as for :func:`solve_planar`).

Backward Euler on cell compositions. Each iteration linearizes the potentials
in the composition change (Newton) and holds Gamma* and the frame correction
at the current iterate (Picard). Face fluxes use the potential difference
between neighbouring cells, so transport across a phase boundary is driven by
the true potential jump rather than by a diffusivity at a mean composition.
With a :class:`TemperatureTable`, ``temperature`` (a constant or a schedule)
sets the table used for each step.

``ideal_flux`` is epsilon of Larsson and Hoglund's eq. 13, a flux
-epsilon (Gamma*_k / u_k) RT du_k/dz added to every element. In a multiphase
region a composition ripple need not change the potentials, so nothing damps
it; this term does. They suggest 0.01-0.05, as small as removes oscillations,
since it also adds numerical diffusion. It equals driving the flux with
mu_k + epsilon RT ln u_k, which is how it enters here.

``tolerance`` adapts the step as in :func:`solve_planar`, in mole fraction; a step that does
not converge, or leaves the table, is halved and redone.

## `janus_kinetics.interface_nucleation`

### `class InterfaceNucleus(angle_a: 'float', angle_b: 'float', cap_radius_a: 'float', cap_radius_b: 'float', rim_radius: 'float', height: 'float', volume: 'float', area_energy: 'float', barrier: 'float') -> None`

The critical lens at an interface: the angles (rad) its caps make with the old interface on the A and B
sides, their radii, the radius of the rim, its height normal to the interface, its volume (m^3), the interfacial
energy it adds (J) and the barrier (J).


### `def interface_nucleus(sigma_old: 'float', sigma_a: 'float', sigma_b: 'float', driving_force: 'float') -> 'InterfaceNucleus'`

The critical nucleus of a new phase at the A/B interface (energy ``sigma_old``), bounded by A/new and new/B
interfaces of energies ``sigma_a`` and ``sigma_b`` (J/m^2), for a driving force per volume ``driving_force``
(J/m^3). Raises ValueError where no lens forms: if sigma_a + sigma_b <= sigma_old the new phase wets the
interface and there is no barrier, and if one of the new interfaces costs more than the other two together
(eqs. 17-18) the nucleus leaves the interface.

### `def molar_volume(cell_volume: 'float', formula_units: 'int') -> 'float'`

Volume per mole of formula units, N_A V_cell / Z (m^3/mol), as d'Heurle obtains it from unit cells.

### `def specific_thickness(interface_energy_change: 'float', driving_force: 'float') -> 'float'`

d_s = Delta sigma / Delta h (m), eq. (12), for Delta sigma in J/m^2 and the formation energy released per
volume of the new phase Delta h in J/m^3.

## `janus_kinetics.interfacial_energy`

### `def broken_bond_energy(solution_enthalpy: 'float', molar_volume: 'float', structure: 'Structure' = 'fcc', normal: 'Sequence[float] | None' = None, exponent: 'float' = 6.0, factor: 'float | None' = None) -> 'float'`

Interfacial energy in J/m^2, eqs. (9)-(10), from the enthalpy of solution of the precipitate in the matrix
(J per mole of atoms, positive when dissolving absorbs heat) and the matrix molar volume (m^3/mol). The
structural factor is :func:`broken_bond_factor` unless ``factor`` is given.

### `def broken_bond_factor(structure: 'Structure', normal: 'Sequence[float] | None' = None, exponent: 'float' = 6.0, shells: 'int' = 100) -> 'float'`

The structural factor z_S,eff / z_L,eff of eq. (9) for an interface with Miller-index ``normal``, or its
average over all orientations when ``normal`` is None. Sonderegger and Kozeschnik tabulate it for m = 6-8
(Table II) and give 0.3291 (fcc) and 0.3280 (bcc) as the r^-6 mean (Table III).

### `def effective_coordination(structure: 'Structure', exponent: 'float' = 6.0, shells: 'int' = 100) -> 'float'`

z_L,eff, the shell-weighted number of bonds per atom, eq. (7) with the continuum tail of eq. (31).

### `def solution_enthalpy(db: 'Database', components: 'Sequence[str]', matrix_phase: 'str', matrix_x: 'Sequence[float]', precipitate_phase: 'str', precipitate_x: 'Sequence[float]', T: 'float', dT: 'float' = 0.5) -> 'float'`

Enthalpy of dissolving one mole of atoms of precipitate in an infinite matrix, J/mol: the thought
experiment of Sonderegger and Kozeschnik's Appendix I (eqs. 19-20), sum_j x_j^P H_j^M - H^P, with H_j^M the
partial molar enthalpies of the matrix. It is taken from the Gibbs energy of solution
Delta G = sum_j x_j^P mu_j^M - G^P by Gibbs-Helmholtz, Delta H = -T^2 d(Delta G / T)/dT at fixed compositions.

Each phase is held as one homogeneous state with its internal degrees of freedom (ordering) relaxed, by
:class:`PhaseMinimizer` started from pycalphad's equilibrium at T. A phase at its solvus composition would
otherwise split into two composition sets on one side of T, and the derivative would follow the tie-line.

## `janus_kinetics.interstitial`

### `class InterstitialPhase(db: 'Database', phase_name: 'str', substitutional, interstitial, dependent=None, mobility_db: 'Database | None' = None, pressure: 'float' = 101325.0)`

Thermodynamics, mobilities and diffusivities of a phase with interstitial elements, in u-fractions.

Composition arguments are u-fractions in the order of :attr:`independent`
(substitutional elements except the dependent one, then the interstitials), or
all of them in the order of :attr:`components`.

- `chemical_potentials(self, u, T) -> 'np.ndarray'`: mu_k in J/mol for every component, shape (..., n).
- `dmu_du(self, u, T, check: 'bool' = True) -> 'np.ndarray'`: dmu_k/du_j for every component k and independent j, shape (..., n, n-1).
- `full_composition(self, u) -> 'np.ndarray'`: All u-fractions (..., n) from the independent ones (..., n-1), or checked if already full.
- `interdiffusion_function(self, temperature: 'Temperature')`: D as a function of (u, t) for the solvers, at a constant T or along a schedule.
- `interdiffusion_matrix(self, u, T) -> 'np.ndarray'`: Eq. (64)-(65) of Andersson and Agren in the independent u-fractions, shape (..., n-1, n-1), m^2/s.
- `mobility(self, u, T) -> 'np.ndarray'`: Omega_k = exp((MF + MQ)/RT)/RT for every component, shape (..., n), m^2 mol/(J s).
- `u_from_x(self, x) -> 'np.ndarray'`: Independent u-fractions from mole fractions given in the order of :attr:`components`.

## `janus_kinetics.layer_competition`

### `class CompetitionResult(times: 'np.ndarray', beta: 'np.ndarray', gamma: 'np.ndarray') -> None`

CompetitionResult(times: 'np.ndarray', beta: 'np.ndarray', gamma: 'np.ndarray')


### `class CompoundLayer(ratio: 'float', delta_c: 'float', diffusivity: 'float', barriers: 'tuple[float, float]') -> None`

A compound layer: ``ratio`` A atoms per B atom, equilibrium A concentration drop ``delta_c`` across it
(atoms per m^3), interdiffusivity ``diffusivity`` (m^2/s) and the reaction barriers of its two interfaces
(m/s), combined by eq. (7).

- `barrier` (property): kappa_eff, eq. (7).
- `changeover` (property): x* = D / kappa_eff, eq. (11).
- `flux(self, thickness)`: A flux across the layer (atoms m^-2 s^-1), eqs. (6), (17)-(18).

### `def composition_factors(alpha: 'float', beta: 'float', gamma: 'float', delta: 'float', atomic_volume: 'float')`

(G_beta, G_gamma, G_beta,gamma, G_gamma,beta), eqs. (10) and (19)-(21); alpha may be infinite and delta
zero, the pure components of section IV.

### `def critical_thickness(beta_layer: 'CompoundLayer', gamma_layer: 'CompoundLayer', alpha: 'float', delta: 'float') -> 'float'`

Thickness the gamma layer must reach before an interface-controlled beta layer can grow beside it, eq. (34):
x_crit = r_1 Delta C_gamma D_gamma / (Delta C_beta kappa_beta). Eq. (34) takes the gamma layer as diffusion
controlled; for a beta layer of zero thickness the exact onset is x_crit - x*_gamma.

### `def growth_window(alpha: 'float', beta: 'float', gamma: 'float', delta: 'float') -> 'tuple[float, float]'`

(r_1, r_2): both layers grow when r_1 < j_beta / j_gamma < r_2, eqs. (27)-(28).

### `def solve_two_layers(beta_layer: 'CompoundLayer', gamma_layer: 'CompoundLayer', alpha: 'float', delta: 'float', atomic_volume: 'float', t_end: 'float', initial: 'tuple[float, float]' = (0.0, 0.0), save_times: 'Sequence[float] | None' = None, rtol: 'float' = 1e-10) -> 'CompetitionResult'`

Integrate eqs. (15)-(18) from layer thicknesses ``initial`` (m). A layer of zero thickness whose rate would
be negative stays at zero (the phase is kinetically suppressed); a shrinking layer stops at zero.

## `janus_kinetics.layer_nucleation`

### `class LayeredRun(segments: 'list', events: 'list' = <factory>) -> None`

The run in segments, one :class:`StackResult` (times absolute) per stretch with a fixed set of layers, and
the nucleation events between them.

- `inventory(self) -> 'list[np.ndarray]'`: Each segment's integral of every independent mole fraction over the stack (r^p dr in a cylinder or sphere).
- `thickness(self, name: 'str') -> 'tuple[np.ndarray, np.ndarray]'`: (times, thickness) of the layer called ``name`` over the segments where it exists.

### `class NucleatingLayer(phase: 'SolutionPhase', interface: 'int', energies: 'tuple[float, float, float]', molar_volume: 'float', diffusivity: 'Diffusivity', name: 'str' = '', cells: 'int' = 20, thickness: 'float' = 2e-09, grading: 'float' = 4.0, barrier: 'float' = 60.0) -> None`

A phase that may nucleate at interface ``interface`` (between layers i and i+1, held in local equilibrium):
the interface energies of the old interface and of the new phase against its left and right neighbours (J/m^2),
its molar volume (m^3/mol), its diffusivity (as for :class:`Layer`), and the layer it starts as.


### `class NucleationEvent(time: 'float', temperature: 'float', interface: 'int', name: 'str', barrier_kt: 'float', driving_force: 'float', composition: 'np.ndarray') -> None`

NucleationEvent(time: 'float', temperature: 'float', interface: 'int', name: 'str', barrier_kt: 'float', driving_force: 'float', composition: 'np.ndarray')


### `def driving_force(phase: 'SolutionPhase', potentials: 'Sequence[float]', T: 'float') -> 'tuple[float, np.ndarray]'`

(Delta G_m in J/mol, mole fractions) of the composition of ``phase`` that gains most from forming out of
material at chemical potentials ``potentials`` (J/mol, one per element): the parallel tangent,
max_x (sum_i x_i mu_i - G_m(x)). Negative when the phase cannot form.

### `def solve_layers_with_nucleation(positions, layers: 'list[Layer]', conditions: 'list', t_end: 'float', dt: 'float', temperature: 'Temperature', candidates: 'Sequence[NucleatingLayer]', check_interval: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, tol: 'float' = 1e-10, geometry: 'Geometry' = 'planar') -> 'LayeredRun'`

:func:`solve_layers` in a plate, cylinder or sphere (``geometry``) in stretches of ``check_interval``, inserting each candidate layer once its
nucleation barrier at its interface falls to its ``barrier`` (in kT) at the temperature ``temperature`` (the
one the interface conditions follow). Candidates are indexed by the interfaces of the stack they find; after one
nucleates, the interfaces to its right shift up by one. Returns the run in segments with the events.

## `janus_kinetics.mobility`

### `class MobilityModel(db: 'Database', lattice: 'SubstitutionalLattice')`

Andersson-Agren atomic mobilities M_B(x, T) for one substitutional phase.

Phi_B = sum_i x_i Phi_B^i + sum_{i<j} x_i x_j sum_r Phi_B^{ij,r} (x_i - x_j)^r
+ ternary terms, the expansion of eq. (67) in Andersson and Agren, J. Appl.
Phys. 72 (1992) 1350. The ternary term with v_ijk^s = x_s + (1 - x_i - x_j - x_k)/3,
the split Phi_B = RT Theta_B - Q_B and M_B = exp(Phi_B / RT) / RT are eqs.
(3)-(6) of Jonsson, ISIJ Int. 35 (1995) 1415. Andersson and Agren's eq. (66)
writes the mobility without the 1/RT; the MQ/MF database convention follows
Jonsson.

Every diffusing element must have an end-member parameter (MQ or MF) in every
pure element of the system, otherwise Phi_B would silently fall to zero near
that corner and the mobility would be meaningless.

- `mobility(self, x: 'np.ndarray', T, magnetic_factor: 'np.ndarray | float' = 1.0) -> 'np.ndarray'`: Atomic mobility M_B in m^2 mol / (J s), shape (..., n).
- `parameters(self, x: 'np.ndarray', T) -> 'tuple[np.ndarray, np.ndarray]'`: Return (MF, MQ) sums in J/mol for each diffusing element, each shaped (..., n).
- `phi(self, x: 'np.ndarray', T) -> 'np.ndarray'`: Phi_B = MF_B + MQ_B = RT ln(M0_B) - Q_B in J/mol, shape (..., n).
- `tracer_diffusivity(self, x: 'np.ndarray', T, magnetic_factor: 'np.ndarray | float' = 1.0) -> 'np.ndarray'`: D*_B = RT M_B in m^2/s, neglecting the correlation factor (Jonsson 1995, eq. 1).

## `janus_kinetics.moving_boundary`

### `class FixedInterface(x_left, x_right)`

Interface compositions held constant; only the velocity is solved for.

Exact for a binary under local equilibrium at constant temperature, where the
tie-line is unique, and the setting of Zener's analysis. With more elements a
fixed pair of compositions cannot satisfy every flux balance at once.


### `class Layer(x0: 'np.ndarray', diffusivity: 'Diffusivity', name: 'str' = '', faces: 'np.ndarray | None' = None) -> None`

One phase layer of a planar stack.

``x0`` is the initial profile on the layer's cells, shape (N, n-1) or (N,) for a binary;
``diffusivity`` is a constant matrix or a callable of (x) or (x, t), as for :func:`solve_planar`.
``faces`` places the N + 1 faces as fractions 0..1 of the layer (see :func:`graded_faces`); the default
is a uniform grid. The fractions are kept as the layer stretches or shrinks.


### `class LocalEquilibrium(left: 'SolutionPhase', right: 'SolutionPhase', temperature: 'Temperature', x_left, x_right)`

Equal chemical potentials of every element across the interface.

``x_left`` and ``x_right`` are starting compositions for the tie-line (n-1
independent fractions each, the dependent element last as in the phases), for
example from :func:`pycalphad.equilibrium`. Newton follows the tie-line
nearest the start, so when a phase has a miscibility gap the starting
compositions decide which branch is found. Both phases must list the same
elements in the same order. ``temperature`` is a constant or a
:class:`TemperatureSchedule`; the tie-line follows it during an anneal.

- `prepare(self, t: 'float') -> 'None'`: Set the temperature for the step ending at time t.
- `tie_line(self, tol: 'float' = 1e-12, max_iter: 'int' = 50) -> 'tuple[np.ndarray, np.ndarray]'`: Solve the binary tie-line, where equal potentials alone fix both compositions.

### `class MovingBoundaryResult(times: 'np.ndarray', interface: 'np.ndarray', z_left: 'np.ndarray', x_left: 'np.ndarray', z_right: 'np.ndarray', x_right: 'np.ndarray', interface_left: 'np.ndarray', interface_right: 'np.ndarray', geometry: 'Geometry' = 'planar') -> None`

MovingBoundaryResult(times: 'np.ndarray', interface: 'np.ndarray', z_left: 'np.ndarray', x_left: 'np.ndarray', z_right: 'np.ndarray', x_right: 'np.ndarray', interface_left: 'np.ndarray', interface_right: 'np.ndarray', geometry: 'Geometry' = 'planar')

- `inventory(self) -> 'np.ndarray'`: Integral of each independent mole fraction over both phases (r^p dr), shape (n_t, m).

### `class PhaseDissolvedError`

The interface reached an outer boundary, so one phase has vanished.


### `class StackResult(times: 'np.ndarray', interfaces: 'np.ndarray', faces: 'list', x: 'list', interface_left: 'np.ndarray', interface_right: 'np.ndarray', names: 'tuple' = (), geometry: 'Geometry' = 'planar') -> None`

Saved state of a layered run; index [t] is a saved time, [j] a layer, [i] an interface.

- `inventory(self) -> 'np.ndarray'`: Integral of each independent mole fraction over the whole stack (r^p dr), shape (n_t, m).
- `profile(self, index: 'int' = -1) -> 'tuple[np.ndarray, np.ndarray]'`: Cell centres (m) and compositions through the whole stack at one saved time.
- `thickness(self) -> 'np.ndarray'`: Layer thicknesses [t, j] in m.

### `def graded_faces(cells: 'int', ratio: 'float' = 1.0, toward: "Literal['left', 'right', 'both']" = 'both') -> 'np.ndarray'`

Face positions across a layer as fractions 0..1 for ``cells`` cells whose widths grow geometrically away
from the ``toward`` end (or both ends), the widest ``ratio`` times the narrowest. A moving interface takes its
flux from the composition gradient over the first half cell, first-order accurate in that width, so fine
cells beside it buy accuracy where it is spent; ``ratio = 1`` is the uniform grid.

### `def solve_layers(positions, layers: 'list[Layer]', conditions: 'list', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, tol: 'float' = 1e-10, picard_tol: 'float' = 1e-10, max_newton: 'int' = 30, max_picard: 'int' = 50, geometry: 'Geometry' = 'planar', tolerance: 'float | None' = None, stop: 'Callable[[StackResult], bool] | None' = None) -> 'StackResult'`

Integrate diffusion through a stack of phase layers with moving interfaces, in a plate (default)
or in concentric shells of a long cylinder or a sphere (``geometry``), positions then being radii.

``positions`` are the layer boundaries from 0 to the stack thickness (len(layers)
+ 1 values); the inner ones are the initial interfaces. ``conditions[i]`` holds
interface i, between layers i and i+1, at fixed compositions or in local
equilibrium. Every interface moves to conserve every element (the flux balance
of the module docstring), and all interface velocities and operating
tie-lines are solved together each step. A reaction layer must be present from
the start, if only a few nanometres thick. If a layer is consumed,
:class:`PhaseDissolvedError` is raised; it and any :class:`ConvergenceError`
carry the results up to the last saved time as ``.result``.

``tolerance`` adapts the step as in :func:`solve_planar`, counting compositions in mole
fraction and each interface position in thicknesses of the thinner layer beside it, so a
thin reaction layer is followed as closely as a thick one. A step that does not converge,
or in which a layer seems to vanish while it still holds 2 % of its first thickness, is
halved and redone.

``stop``, if given, is called at each save time with that one saved state (a StackResult of
length 1); the run ends there, returning what it has, once it returns True.

### `def solve_moving_boundary(length: 'float', interface: 'float', x0_left, x0_right, d_left: 'Diffusivity', d_right: 'Diffusivity', condition: 'FixedInterface | LocalEquilibrium', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, tol: 'float' = 1e-10, picard_tol: 'float' = 1e-10, max_newton: 'int' = 30, max_picard: 'int' = 50, geometry: 'Geometry' = 'planar', tolerance: 'float | None' = None, grading: 'float' = 1.0) -> 'MovingBoundaryResult'`

Integrate two-phase diffusion with a moving interface, as two layers of :func:`solve_layers`; in a
cylinder or sphere (``geometry``) the left phase is the core and the right one the surrounding shell.

``x0_left`` (N_L, n-1) and ``x0_right`` (N_R, n-1) are the initial profiles on the cells over
[0, interface] and [interface, length]; a binary may pass 1D arrays. The cells are uniform, or with
``grading`` > 1 they shrink geometrically toward the interface on both sides, the widest ``grading`` times
the narrowest (:func:`graded_faces`); the result's face arrays give their positions. ``d_left``/``d_right`` are constant matrices or callables like those
of :func:`solve_planar`, for example ``lambda x: alpha.interdiffusion_matrix(x, T)``.
``left``/``right`` are the conditions at z = 0 and z = length. ``tolerance`` adapts the step as in
:func:`solve_layers`.

## `janus_kinetics.multicomponent_precipitation`

### `class MulticomponentResult(times: 'np.ndarray', mean_radius: 'np.ndarray', number_density: 'np.ndarray', volume_fraction: 'np.ndarray', matrix_composition: 'np.ndarray', precipitate_composition: 'np.ndarray', classes: 'list', steps: 'int') -> None`

Saved states of one precipitate population: times (s), mean radius (m), number density (m^-3),
its volume fraction, the matrix mole fractions (t, n), its mean mole fractions (t, n,
volume-weighted), and the class radii, densities and compositions at each saved time.


### `class NucleationSettings(gamma: 'float', molar_volume: 'float', lattice_parameter: 'float', sites: 'float | None' = None, misfit: 'float' = 0.0, site: "Literal['bulk', 'boundary', 'edge', 'corner']" = 'bulk', boundary_ratio: 'float' = 0.0) -> None`

Interfacial energy ``gamma`` (J/m^2), molar volume ``molar_volume`` (m^3 per mole of atoms,
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


### `class PhaseMinimizer(db: 'Database', phase: 'str', components: 'Sequence[str]')`

Internal equilibrium of one phase at fixed composition: site fractions that minimise its molar
Gibbs energy, found by Newton's method on the constrained problem for many compositions at once.

pycalphad's equilibrium does this too, but too slowly to call for every precipitate class at every
step; here the Gibbs energy, gradient and Hessian are compiled once and each solve starts from the
previous site fractions. The phase must hold no vacancies, so that composition is linear in the
site fractions. Chemical potentials follow from the Lagrange multipliers of the composition
constraints: with x_n dependent, dG/dx_i = -nu_i, mu_n = G + sum_i x_i nu_i and mu_i = mu_n - nu_i.

- `disordered(self, x) -> 'np.ndarray'`: Site fractions with every sublattice at composition ``x``, a starting point.
- `hessian(self, y, T: 'float') -> 'np.ndarray'`: d^2 G / dx_i dx_j over the independent mole fractions (k, n-1, n-1), with the site fractions ``y``
- `solve(self, x, y0, T: 'float', tol: 'float' = 1e-11, max_iter: 'int' = 60)`: (site fractions (k, ny), G (k,) in J/mol, mu (k, n)) at mole fractions ``x`` (k, n), from ``y0``.

### `class Population(precipitate: 'PrecipitateThermo', settings: 'NucleationSettings', initial: 'dict | None' = None, precipitate_tracer: 'Sequence[float] | None' = None, growth: "Literal['extremal', 'local_equilibrium']" = 'extremal', growth_form: "Literal['mobility', 'diffusivity']" = 'mobility') -> None`

One precipitate phase nucleating at one kind of site, for :func:`solve_competing_precipitation`.
``initial``, ``precipitate_tracer``, ``growth`` and ``growth_form`` are as in
:func:`solve_multicomponent_precipitation`. A phase may form several populations, in the bulk and on grain
boundaries say, as part II (p. 158) creates one kinetic object per phase and nucleation site.


### `class PrecipitateThermo(db: 'Database', phase: 'str', components: 'Sequence[str]', composition: 'Sequence[float] | None' = None)`

Gibbs energy of the precipitate phase from pycalphad, with its internal degrees of freedom
(ordering, sublattice occupancy) at equilibrium for each composition.

``components`` are the elements in the matrix's order, dependent element last.
A ``composition`` fixes the precipitate's mole fractions (a line compound or a
chosen stoichiometry); without it nuclei take the composition of largest driving force.

- `gm(self, x: 'Sequence[float]', T: 'float') -> 'float'`: Molar Gibbs energy (J per mole of atoms) at mole fractions ``x``.
- `nucleus(self, mu_matrix: 'np.ndarray', T: 'float') -> 'tuple[np.ndarray, float]'`: (composition, molar driving force in J/mol) of the precipitate with the largest driving force
- `site_fractions(self, x: 'Sequence[float]', T: 'float') -> 'np.ndarray'`: Equilibrium site fractions of the phase at mole fractions ``x``, in pycalphad's order, from the

### `def constrained_evolution_rates(radius, composition, mu_precipitate, x_matrix, mu_matrix, tracer, precipitate_tracer, constraints, settings: 'NucleationSettings', T: 'float', interface_mobility: 'float' = inf)`

(drho/dt (k,), dx/dt (k, n)) with the precipitate composition held to linear constraints, after
Kozeschnik, Svoboda and Fischer, Calphad 28 (2004) 379, eqs. (24)-(32). Each row a_j of ``constraints``
(p, n) keeps sum_i a_ji c_ki fixed (eq. 19): a row of ones is the plain substitutional balance, and a row
picking the elements of one sublattice holds a stoichiometry such as (Cr + Fe) : (Mo + W) = 2 : 1. Per class
the n composition rates, the radius rate and the p Lagrange multipliers solve

    A_ij = (R T rho^3 / 45) (1/(c_ki D_ki) + 5/(c_0i D_0i)) delta_ij,
    A_i,n+1 = (R T rho^2 / 3) (c_ki - c_0i) / (c_0i D_0i),   A_i,n+1+j = a_ji,
    A_n+1,n+1 = 1/M + R T rho sum_i (c_ki - c_0i)^2 / (c_0i D_0i),
    B_i = -(rho / 3) (mu_ki - mu_0i),   B_n+1 = -2 gamma / rho - lambda - sum_i c_ki (mu_ki - mu_0i),

with c = x / Omega and M the interface mobility (infinite in part I). With the single row of ones these are
part I's equations, :func:`evolution_rates`, written without a reference element.

### `def grain_structure(grain_size: 'float') -> 'tuple[float, float, float]'`

(grain boundary area in m^2, grain edge length in m, number of grain corners) per m^3 for equal
tetrakaidecahedral grains whose square faces lie ``grain_size`` D apart, Cahn, Acta Metall. 4 (1956)
449, eq. (1): S = 3.35 / D, L = 8.5 / D^2, C = 12.0 / D^3. The exact values of that solid are used
(edge D / (2 sqrt 2), 2 / D^3 grains per volume, faces shared by two grains, edges by three and
corners by four): (3 + 6 sqrt 3) / (4 D), 6 sqrt 2 / D^2 and 12 / D^3.

### `def junction_barrier_factor(site: 'str', k: 'float') -> 'float'`

Nucleation barrier at a grain junction over the homogeneous one. Clemm and Fisher's
W* = (4/27) (b gamma_AB - a gamma_AA)^3 / (c^2 Delta f_v^2) with gamma_AA = 2 k gamma_AB, divided by
16 pi gamma_AB^3 / (3 Delta f_v^2), is (b - 2 k a)^3 / (36 pi c^2). The critical radius of curvature
is that of the homogeneous nucleus, 2 gamma / F. Beyond the critical k of the paper (1, sqrt(3)/2,
sqrt(2/3)) there is no barrier and the phase grows from the junction without nucleating.

### `def misfit_energy(strain: 'float', youngs_modulus: 'float', poisson: 'float', yield_stress: 'float | None' = None) -> 'float'`

Energy per volume of precipitate (J/m^3) from a linear misfit strain eps0 (volume misfit 3 eps0),
part I Appendix A: elastic lambda0 = eps0^2 E / (1 - nu), eq. (A.1), for equal elastic constants in
both phases. With a yield stress the matrix around the particle may yield (Fischer and Oberaigner,
as the appendix gives it): S = sigma_f (1 - nu) / (E eps0), eq. (A.2), and lambda = lambda0 f(S) with
f(S) = 2 S (1 - ln S) - S^2 for S < 1, eq. (A.3), which exceeds 1 between S = 0.199 and 1 (plastic
work added to the stored elastic energy).

### `def nucleation_sites(site: 'str', molar_volume: 'float', grain_size: 'float | None' = None, dislocation_density: 'float | None' = None) -> 'float'`

Nucleation sites per m^3 for :class:`NucleationSettings`: every atom for "bulk", and otherwise the
atoms on the defect, as part II (p. 162) counts the atoms on a dislocation line. With the atomic
distance w = (Omega / N_A)^(1/3), a boundary of area S holds S / w^2 atoms in one layer, an edge of
length L holds L / w, and a corner is taken as one site; the areas, lengths and corners per volume
come from ``grain_size`` through :func:`grain_structure`. "dislocation" counts
``dislocation_density`` (m^-2) / w. Taking one atomic layer, column or site is this package's choice;
neither paper fixes it for grain junctions.

### `def solve_competing_precipitation(matrix: 'SolutionPhase', populations: 'Sequence[Population]', alloy: 'Sequence[float]', T: 'Temperature', t_end: 'float', save_times: 'Sequence[float]', t_start: 'float' = 0.001, max_temperature_change: 'float' = 1.0, max_radius_change: 'float' = 0.02, max_matrix_change: 'float' = 0.0002, max_composition_change: 'float' = 0.001, max_classes: 'int' = 400, min_classes: 'int' = 100, smallest_radius: 'float' = 1e-10, nucleus_update: 'float' = 0.001, min_class_number: 'float' = 1e-06) -> 'list[MulticomponentResult]'`

Several precipitate populations drawing on one matrix, one result per population in the order
given. Part I's mass balance, eq. (4), sums over precipitates of every kind, and part II (section
3.5) runs such competitions to follow precipitation sequences.

Each population nucleates, grows and coarsens as in :func:`solve_multicomponent_precipitation`,
whose arguments these are. The step is common to all of them: its matrix-change limits count the
depletion by every population together, and shrinking classes of all populations share one
dissolution budget. All populations need the same molar volume, part I's single Omega (p. 167),
with which volume fractions carry the mass balance.

### `def solve_multicomponent_precipitation(matrix: 'SolutionPhase', precipitate: 'PrecipitateThermo', alloy: 'Sequence[float]', T: 'Temperature', t_end: 'float', settings: 'NucleationSettings', save_times: 'Sequence[float]', t_start: 'float' = 0.001, max_temperature_change: 'float' = 1.0, max_radius_change: 'float' = 0.02, max_matrix_change: 'float' = 0.0002, max_composition_change: 'float' = 0.001, max_classes: 'int' = 400, min_classes: 'int' = 100, smallest_radius: 'float' = 1e-10, nucleus_update: 'float' = 0.001, min_class_number: 'float' = 1e-06, initial: 'dict | None' = None, precipitate_tracer: 'Sequence[float] | None' = None, growth: "Literal['extremal', 'local_equilibrium']" = 'extremal', growth_form: "Literal['mobility', 'diffusivity']" = 'mobility') -> 'MulticomponentResult'`

Precipitation from a supersaturated matrix of mole fractions ``alloy`` at temperature ``T``, a
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

``growth="local_equilibrium"`` grows each class instead by the law of Chen, Jeppsson and Agren (Acta Mater.
56 (2008) 1890; :func:`fixed_composition_growth_rate`): local equilibrium between the matrix at the interface
and the class's composition under its capillary pressure, and one flux balance per element over its own
effective distance, in the matrix mobilities (``growth_form="mobility"``, their eq. 13) or its interdiffusion
matrix at the far-field composition (``"diffusivity"``, eq. 10). Classes keep their nucleation composition;
nucleation is unchanged.

## `janus_kinetics.ordered`

### `class B2Phase(db: 'Database', phase_name: 'str', elements, mobility_db: 'Database', disordered_phase: 'str', ordering: 'Mapping[tuple[str, str, str], float]', dependent: 'str | None' = None)`

Mobilities and diffusivities of a B2-ordered phase.

``db`` holds the Gibbs energy of ``phase_name``: two sublattices of equal site
number that together hold the elements (and possibly vacancies), plus any
vacancy-only sublattices. ``mobility_db`` holds the disordered mobilities of
``disordered_phase``; ``ordering`` the dQ^order_kij terms. Composition
arguments follow the package convention: mole fractions of ``elements`` with
the dependent element last, vacancies excluded.

- `chemical_potentials(self, x, T: 'float') -> 'np.ndarray'`: Equilibrium chemical potentials of :attr:`elements` (J/mol) with order relaxed.
- `interdiffusion_matrix(self, x, T: 'float', step: 'float' = 0.001) -> 'np.ndarray'`: Number-fixed interdiffusion matrix (m^2/s), dependent element last.
- `mobility(self, x, T: 'float') -> 'np.ndarray'`: Atomic mobilities M_k in m^2 mol / (J s), in the order of :attr:`elements`.
- `site_fractions(self, x, T: 'float') -> 'tuple[np.ndarray, np.ndarray]'`: Equilibrium site fractions (y_a, y_b) over :attr:`constituents` at one composition.
- `tracer_diffusivity(self, x, T: 'float') -> 'np.ndarray'`: D*_k = RT M_k in m^2/s, without the correlation factor.

### `def ordering_energy(terms: 'Mapping[tuple[str, str, str], float]', element: 'str', constituents, y_a, y_b) -> 'np.ndarray'`

Eq. 12: dQ^ord (J/mol) of ``element`` for site fractions y_a, y_b over ``constituents``.

``terms`` maps (k, i, j) to dQ^order_kij, with i and j in either order. The two
sublattices must have equal site numbers, as in B2, so x_i = (y_i^a + y_i^b) / 2.

## `janus_kinetics.oxidation`

### `def deal_grove_constants(diffusivity: 'float', surface_rate: 'float', gas_transport: 'float', solubility: 'float', sites: 'float') -> 'tuple[float, float]'`

(A in m, B in m^2/s) from D_eff (m^2/s), the interface reaction constant k and gas-phase
transport coefficient h (m/s), the oxidant solubility C* and the oxidant molecules per volume of
oxide N_1 (any one unit for both), eqs. (12a)-(12b).

### `def fit_linear_parabolic(times: 'Sequence[float]', thickness: 'Sequence[float]', shift: 'float' = 0.0) -> 'tuple[float, float]'`

(A, B) from measured thickness against time as Deal and Grove extract them (their Figs. 4 and 7): eq.
(12) rearranged as x_0 = B (t + tau) / x_0 - A is a straight line in (t + tau)/x_0, of slope B and
intercept -A, fitted by least squares.

### `def grow_oxide(t_end: 'float', a: 'Value', b: 'Value', temperature: 'Temperature' = 0.0, initial: 'float' = 0.0, save_times: 'Sequence[float] | None' = None, rtol: 'float' = 1e-10) -> 'tuple[np.ndarray, np.ndarray]'`

(times, thickness) of an oxide growing by eq. (9), dx/dt = B / (A + 2 x), with A and B constants or
functions of T along ``temperature`` (a constant or a schedule), from ``initial`` at t = 0. Eq. (9) is
the instantaneous law, so it carries over to an anneal; at constant T it reproduces eq. (13).

### `def oxidation_time(thickness, a: 'float', b: 'float', initial: 'float' = 0.0)`

Time to grow from ``initial`` to ``thickness``, eq. (12) solved for t.

### `def oxide_thickness(t, a: 'float', b: 'float', initial: 'float' = 0.0)`

Oxide thickness at time t, eq. (13), from x_0^2 + A x_0 = B (t + tau). Written as
2 B s / (A + sqrt(A^2 + 4 B s)), s = t + tau, which keeps its precision when A^2 >> B s.

## `janus_kinetics.oxide_scale`

### `class ScaleGrowth(k: 'float', cation: 'float', anion: 'float', mu_inner: 'float', mu_outer: 'float', n_metal_range: 'tuple[float, float]') -> None`

Parabolic constant ``k`` (m^2/s, X^2 = 2 k t) split into its ``cation`` (scale grown at the
gas side) and ``anion`` (grown at the metal side) parts, with the oxygen potentials (J/mol)
at the metal/oxide and oxide/gas interfaces and the range of metal per anion site across the scale.

- `inward_fraction` (property): Share of the scale formed at the metal/oxide interface by oxygen transport.
- `thickness(self, t)`: Scale thickness (m) after time ``t`` (s).

### `def metal_oxide_potential(db: 'Database', T: 'float', metal: 'str', metal_phase: 'str', oxide_phase: 'str', x_anion: 'float', anion: 'str' = 'O') -> 'float'`

Anion chemical potential (J/mol) at which ``metal_phase`` and ``oxide_phase`` coexist,
from an equilibrium of the two at overall anion fraction ``x_anion`` lying between them.

### `def parabolic_scale_growth(oxide: 'OxideCationDiffusion', T: 'float', p_outer: 'float', metal_phase: 'str | None' = None, p_inner: 'float | None' = None, x_anion: 'float | None' = None, nodes: 'int' = 16) -> 'ScaleGrowth'`

Parabolic rate constant of a scale of ``oxide`` between O2 at ``p_outer`` (bar) and either
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

## `janus_kinetics.oxides`

### `class AnionPath(sublattice: 'int', mobility: 'Callable[[float], float]') -> None`

Anion jumps into anion vacancies, their fraction taken proportional to the metal on
sublattice ``sublattice``: M_O = mobility(T) y_M (Hallstrom et al. 2013, eq. 6).


### `class GrainBoundaryMobility(bulk: 'Arrhenius', grain_size: 'float', alpha: 'float' = 0.5, width: 'float' = 5e-10) -> None`

Effective mobility of a polycrystal, Hallstrom et al. (2013) eq. (8):
M_eff = f M_gb + (1 - f) M_b with f = width / grain_size.

The boundary mobility keeps the bulk prefactor and takes ``alpha`` times its
activation energy (their section 4.5.2, with alpha = 1/2 as the rule of thumb
and a 5 Angstrom boundary width). The average holds when the diffusion fields
of neighbouring boundaries overlap, a planar front in a scale with grains much
thinner than the scale; their results are very sensitive to alpha (Fig. 10).


### `class JumpPath(sublattice: 'int', mobility: 'Callable[[float], float]') -> None`

Jumps of the metal into vacancies of sublattice ``sublattice`` (index in the phase), with
mobility ``mobility(T)`` in m^2 mol / (J s).


### `class OxideCationDiffusion(db: 'Database', phase: 'str', paths: 'Sequence[JumpPath]', metal: 'str' = 'FE', anion: 'str' = 'O', anion_path: 'AnionPath | None' = None)`

Metal diffusion in an oxide phase by vacancy jumps on the given sublattices, and
optionally anion diffusion through ``anion_path``.

``metal`` and ``anion`` are element names (e.g. "FE" and "O"); every constituent
of the metal (its ions) counts as the metal on a sublattice. The oxygen
potential is given as a partial pressure of O2 relative to the database's gas
standard state (1 bar for the usual R*T*LN(1E-05*P) term), using its GAS phase.

- `anion_tracer_diffusivity(self, T: 'float', p_o2: 'float') -> 'float'`: D*_O = RT M_O in m^2/s, Hallstrom et al. (2013) eq. (7); zero without an anion path.
- `chemical_diffusivity(self, T: 'float', p_o2: 'float', step: 'float' = 0.02) -> 'float'`: D_M = (D*_M / RT) n_M dmu_M/dn_M (eq. 9) in m^2/s, the derivative taken along the
- `oxygen_potential(self, T: 'float', p_o2: 'float') -> 'float'`: mu_O (J/mol) of O2 gas at T and partial pressure p_o2, from the database's GAS phase.
- `state(self, T: 'float', p_o2: 'float') -> 'dict'`: Site fractions keyed (sublattice, species), metal per anion site n_M, chemical potentials and mole fractions.
- `state_at_potential(self, T: 'float', mu_o: 'float') -> 'dict'`: As :meth:`state`, at the oxygen chemical potential ``mu_o`` (J/mol) instead of a pressure.
- `tracer_diffusivity(self, T: 'float', p_o2: 'float') -> 'float'`: D*_M in m^2/s, eq. (7) (eq. (13) for a single path).

## `janus_kinetics.particle_growth`

### `class GrowthState(radius: 'float', velocity: 'float', matrix_interface: 'np.ndarray', particle_interface: 'np.ndarray', omega: 'np.ndarray', factor: 'np.ndarray') -> None`

Growth rate ``velocity`` (m/s) of a particle of ``radius``, with all mole fractions of the matrix
(``matrix_interface``) and particle (``particle_interface``) at its interface, and each independent element's
supersaturation ``omega`` and distance factor ``factor``.


### `def diffusion_distance_factor(omega) -> 'np.ndarray'`

n = Omega / (2 lambda^2) of eq. (5); 1 where Omega <= 0, below 1e-12, or at or above 1 - 1e-12 taken there.
Vectorized: ln lambda by cubic Hermite interpolation of the inverse of ln f below Omega = 1/2 and of ln(1 - f)
above (which keeps both ends free of cancellation), tabulated with their slopes.

### `def fixed_composition_growth_rate(matrix: 'SolutionPhase', composition: 'Sequence[float]', gibbs: 'float', x_matrix: 'Sequence[float]', radius: 'float', T: 'float', diffusivity=None, gamma: 'float' = 0.0, molar_volume: 'float' = 1e-05, misfit: 'float' = 0.0, form: "Literal['diffusivity', 'mobility']" = 'diffusivity') -> 'GrowthState'`

Growth rate of a sphere of fixed mole fractions ``composition`` (all n) and molar Gibbs energy ``gibbs``
(J/mol), a stoichiometric phase or one whose composition is held, as the paper's model allows (p. 1891).
Local equilibrium is then the single condition sum_i x^P_i mu^I_i = G^P + V_m (2 sigma / R + misfit), with
``misfit`` an elastic energy per volume; the flux balances are those of :func:`particle_growth_rate`, whose
other arguments these are. The matrix interface starts from the far-field composition.

### `def growth_parameter(omega: 'float') -> 'float'`

lambda of eq. (2) for a supersaturation 0 < Omega < 1.

### `def particle_growth_rate(matrix: 'SolutionPhase', particle: 'SolutionPhase', x_matrix: 'Sequence[float]', radius: 'float', T: 'float', start: 'tuple[Sequence[float], Sequence[float]]', diffusivity=None, gamma: 'float' = 0.0, molar_volume: 'float' = 1e-05, form: "Literal['diffusivity', 'mobility']" = 'diffusivity') -> 'GrowthState'`

Growth rate of a sphere of ``radius`` in a matrix of far-field composition ``x_matrix`` (independent mole
fractions, the phases' dependent element last), solving eq. (9) with eq. (10) (``form="diffusivity"``, with the
matrix interdiffusion ``diffusivity``, an (n-1) x (n-1) array in m^2/s) or eq. (13) (``form="mobility"``, with the
matrix's own mobilities at the interface composition). ``start`` gives the interface compositions (matrix,
particle) to begin from, for instance a tie-line from pycalphad; where more than one operating tie-line exists
(partitioning and non-partitioning growth) it decides which is found. ``gamma`` (J/m^2) and ``molar_volume``
(m^3/mol) add the Gibbs-Thomson pressure on the particle.

## `janus_kinetics.particle_reaction`

### `def conversion(t, growth_constant: 'float | Callable[[float], float]', radius: 'float', z: 'float', geometry: 'Geometry', temperature: 'Temperature | None' = None) -> 'np.ndarray'`

Fraction of the plate, cylinder or sphere converted after time t (s).

``growth_constant`` is the planar k_p (m^2/s), a number or a function of T
evaluated along ``temperature`` (a constant or a schedule). ``radius`` is r0,
the half-thickness of a plate. Eq. 26 is inverted for alpha, which is 1 once
the core is used up.

### `def conversion_integral(alpha, z: 'float', geometry: 'Geometry')`

Left-hand side of van Loo's eq. 26, equal to 2 k_p t / r0^2 at conversion alpha.

### `def growth_constant_from_conversion(alpha, t, radius: 'float', z: 'float', geometry: 'Geometry')`

Planar k_p (m^2/s) from a measured conversion alpha at time t, by eq. 26.

This is van Loo's use of the curves: powder or wire data give the parabolic
constant of a planar couple.

### `def layer_thickness(alpha, z: 'float', geometry: 'Geometry')`

Van Loo's eq. 25: layer thickness d / r0 at conversion alpha.

## `janus_kinetics.phase`

### `class ShortCircuit(fraction: 'float', mobility_db: 'Database') -> None`

Fast diffusion paths (dislocations or grain boundaries) averaged into the lattice.

Hart, Acta Metall. 5 (1957) 597, eq. (5), for self-diffusion along dislocations:
an atom that spends a fraction f of its time in the paths sees
D_eff = f D_path + (1 - f) D_lattice. Two extensions are this package's, not
Hart's. The same time-fraction argument is applied to grain boundaries, and,
through D* = RT M (Andersson and Agren, J. Appl. Phys. 72 (1992) 1350,
eq. (61)), to each element's mobility and so to every diffusion matrix built
from it. Hart's condition is that a diffusion distance spans many paths; a
coarse microstructure with isolated boundaries is outside it.

``fraction`` is Hart's f, the fraction of atoms in the paths. ``mobility_db``
holds MQ/MF parameters for diffusion along the paths, entered under the same
phase name and constitution as the lattice parameters.


### `class SolutionPhase(db: 'Database', phase_name: 'str', elements: 'list[str] | tuple[str, ...]', dependent: 'str | None' = None, mobility_db: 'Database | None' = None, short_circuit: 'ShortCircuit | None' = None, mobilities: 'bool' = True)`

Thermodynamics and mobilities of one substitutional solution phase.

Composition arguments accept either the n-1 independent mole fractions or all
n mole fractions, in the order of :attr:`elements` (dependent element last).

- `chemical_potentials(self, x, T) -> 'np.ndarray'`: Chemical potentials of every element in J/mol, shape (..., n).
- `dmu_dx(self, x, T) -> 'np.ndarray'`: dmu_i/dx_j (dependent eliminated), J/mol, shape (..., n, n-1).
- `interdiffusion_function(self, temperature: 'Temperature', frame: 'Frame' = 'number', partial_volumes=None)`: D~ as a function of the solver variables and t, at a constant T or along a schedule.
- `interdiffusion_matrix(self, x, T, frame: 'Frame' = 'number', partial_volumes: 'np.ndarray | None' = None, magnetic_factor=1.0) -> 'np.ndarray'`: Reduced interdiffusion matrix D~ in m^2/s, shape (..., n-1, n-1).
- `intrinsic_matrix(self, x, T, magnetic_factor=1.0) -> 'np.ndarray'`: Lattice-frame diffusivities D^I_kj in m^2/s, shape (..., n, n-1).
- `lattice_velocity(self, x, gradient, T, magnetic_factor=1.0, partial_volumes=None) -> 'np.ndarray'`: Kirkendall (marker) velocity, m/s.
- `lattice_velocity_function(self, temperature: 'Temperature', partial_volumes=None)`: Marker velocity as a function of (variables, gradients, t), for ``solve_planar(..., lattice_velocity=...)``.
- `mobility(self, x, T, magnetic_factor=1.0) -> 'np.ndarray'`: Atomic mobilities in m^2 mol / (J s), including any short-circuit paths.
- `reduced_hessian(self, x, T) -> 'np.ndarray'`: d2G/dx_k dx_j with the dependent element eliminated, J/mol, shape (..., n-1, n-1).
- `thermodynamic_factor(self, x, T) -> 'np.ndarray'`: Dimensionless (x_i/RT) dmu_i/dx_j, shape (..., n, n-1).

## `janus_kinetics.precipitation`

### `class DiluteSolution(x_eq: 'float', x_p: 'float', v_at_p: 'float', v_at_m: 'float', gamma: 'float', T: 'float') -> None`

Matrix/precipitate thermodynamics of a dilute binary, Perez et al. (2008) eq. (9).

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

- `alpha` (property): v_at^M / v_at^P, the ratio in Perez et al. eq. (8).
- `capillary_length` (property): R0 = 2 gamma v_at^P / kT in m.
- `critical_radius(self, x) -> 'np.ndarray'`: R* = -2 gamma / dg; infinite where the matrix is not supersaturated.
- `driving_force(self, x) -> 'np.ndarray'`: Chemical driving force per unit precipitate volume, J/m^3; negative when supersaturated.
- `interface_composition(self, radius) -> 'np.ndarray'`: X^i(R), the matrix solute fraction in equilibrium with a particle of radius R (eq. 9).
- `smallest_radius` (property): Radius below which eq. (9) has no matrix composition below X^P.

### `class Kinetics(diffusivity: 'float', lattice_parameter: 'float', sites: 'float') -> None`

Transport and nucleation inputs: solute diffusivity in the matrix (m^2/s),
lattice parameter ``a`` (m) of the condensation rate, and the number of
nucleation sites per unit volume N0 (m^-3) of eq. (3).

The diffusivity can come from a mobility database as D* = RT M of the solute
(Andersson and Agren, J. Appl. Phys. 72 (1992) 1350, eq. (61)), for example
``R * T * phase.mobility(x, T)[solute_index]``.


### `class MeanRadiusResult(times: 'np.ndarray', mean_radius: 'np.ndarray', number_density: 'np.ndarray', matrix_fraction: 'np.ndarray', critical_radius: 'np.ndarray') -> None`

MeanRadiusResult(times: 'np.ndarray', mean_radius: 'np.ndarray', number_density: 'np.ndarray', matrix_fraction: 'np.ndarray', critical_radius: 'np.ndarray')


### `class PrecipitationResult(times: 'np.ndarray', radius_edges: 'np.ndarray', number: 'np.ndarray', matrix_fraction: 'np.ndarray', critical_radius: 'np.ndarray', steps: 'int' = 0, rejected: 'int' = 0) -> None`

PrecipitationResult(times: 'np.ndarray', radius_edges: 'np.ndarray', number: 'np.ndarray', matrix_fraction: 'np.ndarray', critical_radius: 'np.ndarray', steps: 'int' = 0, rejected: 'int' = 0)

- `mean_radius` (property): R_mean = sum R_i N_i / sum N_i (eq. 20); NaN before any precipitate exists.
- `number_density` (property): N = sum_i N_i (eq. 19).
- `volume_fraction` (property): f_T = (4/3) pi sum_i N_i R_i^3 (eq. 21).

### `def coarsening_fraction(radius, critical) -> 'np.ndarray'`

f_coars = 1 - 1000 (R / R* - 1)^2 for 0.99 R* < R < 1.01 R*, Perez et al. eq. (17), and 0 outside, where the
paper leaves it undefined. The function drops from 0.9 to 0 at both edges of the band; it is used as printed.

### `def growth_rate(thermo: 'DiluteSolution', kin: 'Kinetics', x, radius) -> 'np.ndarray'`

dR/dt = (D/R) (X - X^i(R)) / (alpha X^P - X^i(R)), Perez et al. eq. (10).

### `def nucleation_rate(thermo: 'DiluteSolution', kin: 'Kinetics', x: 'float', t: 'float') -> 'tuple[float, float]'`

(dN/dt in m^-3 s^-1, nucleation radius R*_kT in m) at matrix fraction x and time t.

Eq. (6) with the Zeldovich factor Z = v_at^P / (2 pi R*^2) sqrt(gamma/kT)
(corrected eq. 4), beta* = 4 pi R*^2 D X / a^4 (eq. 5), tau = 4 / (2 pi beta* Z^2),
and R*_kT = R* + (1/2) sqrt(kT / (pi gamma)) (corrected eq. 7).

### `def solubility_from_pycalphad(db, components, matrix: 'str', precipitate: 'str', solute: 'str', T: 'float', x_overall: 'float', pressure: 'float' = 101325.0) -> 'tuple[float, float]'`

(X^i(inf), X^P): solute fractions of matrix and precipitate in two-phase equilibrium.

From :func:`pycalphad.equilibrium` with only the two phases allowed, at an
overall solute fraction ``x_overall`` inside the two-phase field; this is the
thermodynamic input that eqs. (9)-(10) of Perez et al. (2008) take as X^i and X^P.

### `def solve_kwn(thermo: 'DiluteSolution', kin: 'Kinetics', x0: 'float', t_end: 'float', radius_edges: 'np.ndarray', number0: 'np.ndarray | None' = None, save_times=None, dt0: 'float' = 0.001, max_change: 'float' = 0.01, growth: 'float' = 1.1) -> 'PrecipitationResult'`

Evolve the precipitate size distribution from an alloy of solute fraction ``x0``.

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

### `def solve_mean_radius(thermo: 'DiluteSolution', kin: 'Kinetics', x0: 'float', t_end: 'float', number0: 'float' = 0.0, radius0: 'float | None' = None, save_times=None, rtol: 'float' = 1e-06) -> 'MeanRadiusResult'`

Number density N and mean radius R of the precipitates alone, the "mean radius approach" of Perez et al.
(section 3.1). New nuclei of radius R*_kT join at the nucleation rate of eq. (6), pulling R toward their size
(eq. 12); growth is eq. (10) at R. Coarsening is the LSW rate of eq. (11), d R / dt = (4/27) X^i / (alpha X^P -
X^i) R0 D / R^2 (R0 over X^P for a compound, as the paper notes), and eq. (14) for N; the two are blended by
the coarsening fraction of eq. (17), R by eq. (15) and N by eq. (18). The matrix fraction X follows from the mass
balance, eq. (13). Eq. (18) compares the coarsening and nucleation rates of N, of opposite signs; it is read
here as comparing their sizes, -dN/dt|coars against dN/dt|nucl. Integrated by LSODA, the paper's adaptive
Runge-Kutta being unsuited to the stiff start, when 1/N dN/dt is large.

Starting from ``number0`` particles per m^3 of radius ``radius0`` (none by default).

## `janus_kinetics.reaction_layers`

### `class LineCompound(name: 'str', x: 'float', molar_volume: 'float', gibbs: 'Value | None' = None, mobility: 'tuple[float, float] | Callable[[float], tuple[float, float]] | None' = None, integrated_diffusivity: 'Value | None' = None, tracer_ratio: 'float | None' = None) -> None`

A phase of fixed composition in an A-B couple.

``x`` is the mole fraction of B and ``molar_volume`` is per mole of atoms
(m^3/mol). ``gibbs`` is the Gibbs energy per mole of atoms (J/mol), a number
or a function of T. A phase that grows as a layer needs its transport, given
either as ``mobility`` = (M_A, M_B) in m^2 mol/(J s) (the tracer diffusivity
is D* = R T M), constant or a function of T, or as a measured
``integrated_diffusivity`` (m^2/s, constant or a function of T) together with
``tracer_ratio`` = M_B / M_A, which only sets how the flux splits between the
elements and so where markers end up. End members need neither.

- `from_pycalphad(db, phase: 'str', elements: 'Sequence[str]', molar_volume: 'float', name: 'str | None' = None, **transport) -> 'LineCompound'`: A stoichiometric pycalphad phase; ``elements`` = (A, B) and G(T) is read from the database.

### `class ParabolicGrowth(names: 'tuple[str, ...]', interfaces: 'np.ndarray', growth_constants: 'dict[str, float]', consumed: 'dict[str, float]', marker: 'float', marker_phase: 'str', lattice_velocity: 'np.ndarray') -> None`

Similarity solution of a couple started at t = 0 at constant T.

Positions scale with sqrt(t) in the frame of the anchored end member, with the
original interface at 0: interface j sits at ``interfaces[j] * sqrt(t)``, and
layer k has d^2 = 2 k_p t (van Loo's growth constant, p. 55) with k_p in
``growth_constants``. ``consumed`` gives the thickness of each end member used
up, divided by sqrt(t); ``marker`` is the Kirkendall plane position divided by
sqrt(t) and ``marker_phase`` the phase it lies in.


### `class ReactionResult(times: 'np.ndarray', thickness: 'dict[str, np.ndarray]', interfaces: 'list[np.ndarray]', markers: 'np.ndarray', consumed: 'list[tuple[float, str]]', geometry: 'str' = 'planar') -> None`

Layer thicknesses (m) of every phase at ``times``.

``thickness[name]`` is inf for a semi-infinite end member and 0 once a phase is
used up. ``interfaces[i]`` holds the interface positions at ``times[i]`` in the
anchored end member's frame, original interface at 0 for a fresh couple.
``markers`` has one column per marker. ``consumed`` lists (time, name) for
every end member used up during the run. In a cylinder or sphere the interfaces
and markers are radii and the core's "thickness" is its radius.


### `def integrated_diffusivity(left: 'LineCompound', layer: 'LineCompound', right: 'LineCompound', T: 'float') -> 'float'`

Wagner's D_int (van Loo eq. 17) of ``layer`` between its neighbours at T, in m^2/s.

### `def parabolic_growth(phases: 'Sequence[LineCompound]', temperature: 'float', anchor: 'str' = 'left') -> 'ParabolicGrowth'`

Parabolic growth constants of every layer in a fresh couple at constant T.

All layers start together (van Loo p. 61, item c). With d_k = delta_k sqrt(t)
every velocity scales as 1/sqrt(t), so delta_k = 2 (V_k - V_(k-1)) where V are
the interface velocity coefficients computed with thicknesses delta. That
nonlinear system is solved in log(delta).

### `def solve_reaction_layers(phases: 'Sequence[LineCompound]', t_end: 'float', temperature: 'Temperature', thickness: 'Sequence[float] | None' = None, end_thickness: 'tuple[float | None, float | None]' = (None, None), markers: 'Sequence[float]' = (0.0,), save_times=None, anchor: 'str | None' = None, rtol: 'float' = 1e-09, geometry: 'str' = 'planar') -> 'ReactionResult'`

Grow the layers of a binary couple of line compounds over an anneal.

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

## `janus_kinetics.schedule`

### `class TemperatureSchedule(times, temperatures)`

Piecewise-linear temperature T(t) in K through (time, temperature) points.

Held at the first temperature before the first point and at the last one after
it, so a single point is an isothermal hold. Ramps, spikes and multi-step anneals
are lists of points, e.g. ``TemperatureSchedule([0, 10, 70, 80], [300, 1300, 1300, 300])``.


## `janus_kinetics.solidification`

### `class BackDiffusionResult(fs: 'np.ndarray', liquid: 'np.ndarray', interface_solid: 'np.ndarray', z_faces: 'np.ndarray', solid: 'np.ndarray', geometry: 'Geometry') -> None`

Fraction solid, liquid composition and interface solid composition (both over C0) at each step,
with the solid profile (positions in m, compositions over C0) at the end.

- `mean_solid(self) -> 'float'`: Average composition of the solid over C0 at the end; with no liquid left this is 1 by mass balance.

### `class CompleteSolidification(back_diffusion: 'SolidificationResult', scheil: 'ScheilResult | None', handoff_temperature: 'float | None', primary: 'str', length: 'float', geometry: 'Geometry') -> None`

A back-diffusion run handed on to a Scheil-Gulliver path once a second solid appears: the run up to the
hand-off, the Scheil part from there (its fractions are of the liquid left at the hand-off, or None if no second
solid formed), the hand-off temperature, and the run's geometry and volume element.

- `fraction_solid` (property): Fraction of the element solid along the whole path.
- `mean_composition(self) -> 'np.ndarray'`: Independent mole fractions of everything in the element (solids and any liquid left), which the run
- `phase_fractions(self) -> 'dict'`: Amount of each solid phase at the end, as a fraction of the element.

### `class ScheilResult(temperatures: 'np.ndarray', fraction_solid: 'np.ndarray', liquid_composition: 'np.ndarray', solids: 'list', components: 'tuple') -> None`

Temperatures (K), fraction solid, liquid mole fractions (n_T, n) and the solid formed at each step:
for every temperature step the phases that formed with their amounts (fraction of the alloy) and
mole fractions.

- `increments(self) -> 'tuple[np.ndarray, np.ndarray]'`: (fraction-solid edges, independent mole fractions of the solid laid down in each step), the input
- `solid_phase_fractions(self) -> 'dict'`: Amount of each solid phase formed over the whole path, as a fraction of the alloy.

### `class SolidificationResult(times: 'np.ndarray', temperature: 'np.ndarray', fraction_solid: 'np.ndarray', liquid: 'np.ndarray', interface_solid: 'np.ndarray', solid_faces: 'list', solid: 'list', finished: 'bool', stopped: 'bool' = False) -> None`

Times (s), temperatures (K), fraction solid (by volume of the element), the liquid's mole fractions (t, n-1),
the interface solid's (t, n-1), the solid's faces (m) and compositions at each saved time, and whether the
liquid ran out (``finished``).


### `def equilibrium_start(db: 'Database', components: 'Sequence[str]', solid_phase: 'str', liquid_phase: 'str', alloy: 'Sequence[float]', T: 'float') -> 'tuple[float, np.ndarray, np.ndarray]'`

(fraction solid, solid and liquid independent mole fractions) of the solid-liquid equilibrium of ``alloy``
at ``T`` from pycalphad, a start for :func:`solve_solidification` a little below the liquidus.

### `def kobayashi(k: 'float', gamma: 'float', fs, x=None, terms: 'int' = 10000) -> 'np.ndarray'`

Exact solution of the Brody-Flemings model over C0 (Kobayashi eqs. 37-40): the interface solid
composition at fraction solid ``fs``, or with ``x`` (x* = x / X, 0 at the dendrite spine) the profile
in the solid. The series converges for 0 < k < 2 and gamma > 0; ``terms`` must grow as f_s
approaches 1 with k small, where the terms fall off slowly.

### `def scheil_gulliver(db: 'Database', components: 'Sequence[str]', phases: 'Sequence[str]', composition: 'Sequence[float]', start: 'float', step: 'float' = 1.0, liquid: 'str' = 'LIQUID', stop_fraction: 'float' = 0.0001, t_min: 'float' = 300.0, disordered: 'dict | None' = None) -> 'ScheilResult'`

Scheil-Gulliver solidification by the algorithm Bocklund et al., Materialia 11 (2020) 100689, give
(section 2): from ``start`` (K, above the liquidus) the remaining liquid is equilibrated at each
temperature, the solid that forms is set aside and never changes again, the liquid's fraction is
reduced by the solid's, and its composition becomes that of the equilibrium liquid, until less than
``stop_fraction`` liquid is left. ``composition`` is the n-1 independent mole fractions in the order of
``components`` (the last element dependent). ``disordered`` maps an ordered phase to the name its disordered
composition sets take (``{"L12_FCC": "FCC_A1"}`` for gamma and gamma' described by one model); without it
both appear under the phase's own name.

### `def segregation_index(x) -> 'np.ndarray'`

Residual segregation (max - min) of each profile in ``x`` (times, cells, ...) relative to the first,
the index homogenization treatments are judged by.

### `def segregation_profile(fs_edges, solid_composition, length: 'float', n_cells: 'int', geometry: 'Geometry' = 'planar') -> 'tuple[np.ndarray, np.ndarray]'`

Face positions (m) and cell-averaged solid composition (n_cells, m) across a volume element of size
``length`` from the solid laid down in increments: increment i, of composition ``solid_composition[i]``
(m independent fractions), filled the fraction solid from ``fs_edges[i]`` to ``fs_edges[i + 1]``,
counted by volume. The solid grows from r = 0, so fraction f sits at r = length f^(1/(p+1)). Solid not
yet formed when the record ends (fs_edges[-1] < 1) takes the last increment's composition. This turns
a Scheil or back-diffusion record into the starting profile of a homogenization anneal.

### `def solve_back_diffusion(k: 'float', diffusivity: 'float', length: 'float', solidification_time: 'float', n_cells: 'int' = 200, steps: 'int' = 2000, geometry: 'Geometry' = 'planar', fraction_solid: 'Callable[[float], float] | None' = None, fs_start: 'float' = 0.0001, fs_end: 'float' = 1.0) -> 'BackDiffusionResult'`

The Brody-Flemings model solved numerically: a solid grows from the dendrite spine (r = 0) across
a volume element of size ``length`` (half the arm spacing) while the liquid stays uniform; the
interface solid holds k C_L, the solid diffuses with constant D, and nothing leaves the element.

The solid grows to ``fraction_solid(t / solidification_time)`` (default sqrt, the parabolic growth
Kobayashi assumes, his eq. 14); in a cylinder or sphere the fraction is that of the volume. Each step
the solid is solved on its stretched grid by the package's moving-grid scheme, and the liquid
composition from the element's solute balance. The solid's response is linear in the held interface
value, so two solves give the exact C_L of the step. The run starts at ``fs_start`` with the solid at
k C0 (the Scheil and exact solutions agree there to order f_s). Compositions are over C0.

### `def solve_solidification(solid: 'SolutionPhase', liquid: 'SolutionPhase', temperature: 'TemperatureSchedule', t_end: 'float', length: 'float', start: 'tuple[float, Sequence[float], Sequence[float]]', solid_diffusivity=None, cells: 'int' = 200, grading: 'float' = 20.0, dt: 'float' = 0.05, save_times=None, geometry: 'Geometry' = 'planar', liquid_diffusivity: 'float' = 1e-06, liquid_cells: 'int' = 4, tol: 'float' = 1e-08, tolerance: 'float | None' = None, stop: 'Callable[[float, float, np.ndarray], bool] | None' = None) -> 'SolidificationResult'`

Solidification with back-diffusion in the solid and local equilibrium at its interface with the liquid, for
any number of elements, the volume-element calculation of DICTRA's solidification runs (Borgenstam et al.,
J. Phase Equilib. 21 (2000) 269; Lippard et al., Metall. Mater. Trans. B 29 (1998) 205). The primary solid
grows from r = 0 (a plate, or the axis of a cylinder or centre of a sphere) across a volume element of
``length``, half the dendrite arm spacing, as ``temperature`` falls; the liquid beyond it is kept uniform (a
diffusivity ``liquid_diffusivity`` far above the solid's), the interface follows the tie-line of local
equilibrium at the current temperature, and nothing leaves the element.

``start`` is (fraction solid, solid and liquid independent mole fractions) at t = 0, for instance
:func:`equilibrium_start` a little below the liquidus. ``solid_diffusivity`` is a constant matrix or a callable
of (x) or (x, t); by default the solid's own interdiffusion matrix at the current temperature. The solid's
``cells`` are graded ``grading``:1 toward the interface. Only the primary solid forms here; for a run that goes on
to a second solid (a eutectic) see :func:`solve_solidification_to_end`. When the liquid runs out the run stops and returns what it has
(``finished``). With no diffusion in the solid it follows the Scheil-Gulliver path, with fast diffusion the
lever rule. ``stop(t, T, liquid)``, if given, is asked at each save time and ends the run when it returns True
(``stopped``); :func:`solve_solidification_to_end` uses it to hand a run on once a second solid appears.

### `def solve_solidification_to_end(db: 'Database', phases: 'Sequence[str]', solid: 'SolutionPhase', liquid: 'SolutionPhase', temperature: 'TemperatureSchedule', t_end: 'float', length: 'float', start: 'tuple[float, Sequence[float], Sequence[float]]', step: 'float' = 1.0, threshold: 'float' = 1e-06, disordered: 'dict | None' = None, **kwargs) -> 'CompleteSolidification'`

:func:`solve_solidification` until the liquid would form a solid other than the primary one, then the
Scheil-Gulliver path of the liquid left, as Kattner, Boettinger and Coriell, Z. Metallkd. 87 (1996) 522, treat
the eutectic (p. 525): 'During eutectic solidification, solid diffusion was neglected.' At every ``step`` K of
cooling the liquid is equilibrated alone with ``phases`` (pycalphad names, the liquid's and the primary solid's
among them); once another phase takes more than ``threshold`` of it the run is handed on from that state, the
Scheil steps going on at ``step`` K from there, so the hand-off is as late as one step. The primary solid keeps
the profile back-diffusion gave it; later primary solid joins the Scheil solids. ``temperature`` must fall;
other keyword arguments go to :func:`solve_solidification`.

Solids are told apart by phase name. Where one ordered model describes the primary solid and the second, as
Dupin's Al-Cr-Ni does gamma and gamma' (pycalphad names both L12_FCC), ``disordered`` names the phase its
disordered composition sets stand for, ``{"L12_FCC": "FCC_A1"}``, and only ordered sets count as the second
solid (see :func:`scheil_gulliver`).

## `janus_kinetics.solver`

### `class Boundary(kind: "Literal['no_flux', 'fixed']", composition: 'np.ndarray | Callable | None' = None, held: 'tuple[int, ...] | None' = None) -> None`

Planar boundary condition: zero flux, or some or all components held at the face.

Build it with :meth:`no_flux`, :meth:`fixed` or :meth:`potential`. Compositions
are the n-1 independent fractions in the order the phase lists them. With
``held`` naming a subset of components, only those are held and the others
pass no flux through the face.

- `face(self, t: 'float', x_cell: 'np.ndarray', m: 'int') -> 'tuple[np.ndarray, np.ndarray]'`: Face composition and the mask of held components at time t.
- `fixed(composition, components=None) -> 'Boundary'`: Composition held at the face (independent fractions), constant or a function of time.
- `no_flux() -> 'Boundary'`: Zero flux of every element through the face.
- `potential(phase, potentials: 'dict', temperature) -> 'Boundary'`: Hold the chemical potentials of some components at the face, as a gas atmosphere does.

### `class DiffusionResult(times: 'np.ndarray', z_faces: 'np.ndarray', x: 'np.ndarray', picard_iterations: 'list[int]' = <factory>, markers: 'np.ndarray | None' = None, geometry: 'Geometry' = 'planar') -> None`

Saved profiles: ``x[t_index, cell, component]`` for the independent components.

- `inventory(self) -> 'np.ndarray'`: Integral of each mole fraction over the domain (r^p dr) at every saved time, shape (n_t, n-1).
- `widths` (property): Cell measures: widths for a plate, volumes per unit angle for a cylinder or sphere.

### `def solve_planar(z_faces: 'np.ndarray', x0: 'np.ndarray', diffusivity: 'Diffusivity', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, picard_tol: 'float' = 1e-10, max_picard: 'int' = 50, markers=None, lattice_velocity: 'LatticeVelocity | None' = None, geometry: 'Geometry' = 'planar', tolerance: 'float | None' = None) -> 'DiffusionResult'`

Integrate interdiffusion from ``x0`` to ``t_end`` in a plate (default), a long cylinder or a sphere.

Parameters
----------
z_faces : cell face positions (m), strictly increasing, length N+1.
x0 : initial independent mole fractions, shape (N, n-1) or (N,) for a binary.
diffusivity : constant (n-1, n-1) matrix in m^2/s, or a callable mapping
    compositions (M, n-1) to matrices (M, n-1, n-1), such as
    ``lambda x: phase.interdiffusion_matrix(x, T)``. A callable of (x, t) is
    evaluated at the end of each step, which runs a non-isothermal anneal;
    ``phase.interdiffusion_function(schedule)`` builds one.
dt : maximum time step (s); steps are shortened to land on ``save_times``.
picard_tol : convergence threshold on the max change in the variables between
    Picard iterates, relative to their largest initial magnitude when that
    exceeds 1 (concentrations in mol/m^3, say); absolute for mole fractions.
markers : initial positions (m) of inert Kirkendall markers to follow.
lattice_velocity : callable (x, dx_dz) or (x, dx_dz, t) -> v giving the lattice
    velocity at the interior faces, such as ``lambda x, g: phase.lattice_velocity(x, g, T)``.
    Required with ``markers``.
geometry : "planar", "cylindrical" or "spherical"; for the curved ones ``z_faces`` are
    radii, and a first face at r = 0 is the axis or centre, where no flux passes.
    Markers are followed in the planar geometry only.
tolerance : with it the step adapts, ``dt`` becoming the largest step. Backward Euler
    errs by about (step^2 / 2) d^2x/dt^2 per step; the second derivative is taken from
    the change in dx/dt between consecutive steps, and a step whose estimate exceeds
    ``tolerance`` (in the units of ``picard_tol``) is halved and redone, as is one whose
    Picard iteration fails. A step whose estimate is under an eighth of ``tolerance``
    is followed by one twice as long. Steps are dt / 2^k, so a constant D needs few
    factorizations; the first is dt / 1024.

Returns profiles at t=0 and at every save time (always including ``t_end``).

### `def uniform_faces(length: 'float', n_cells: 'int', origin: 'float' = 0.0) -> 'np.ndarray'`

Face positions (m) of ``n_cells`` equal cells spanning [origin, origin + length].

## `janus_kinetics.thermo`

### `class GibbsEnergy(db: 'Database', lattice: 'SubstitutionalLattice', pressure: 'float' = 101325.0)`

Compiled molar Gibbs energy (J per mole of atoms) and its exact Hessian.

- `chemical_potentials(self, x: 'np.ndarray', T) -> 'np.ndarray'`: mu_i = G + dG/dx_i - sum_k x_k dG/dx_k in J/mol, shape (..., n).
- `gm(self, x: 'np.ndarray', T) -> 'np.ndarray'`: Molar Gibbs energy at full mole fractions ``x`` (..., n).
- `hessian(self, x: 'np.ndarray', T) -> 'np.ndarray'`: Unconstrained second derivatives d2GM/dx_i dx_j, shape (..., n, n).

### `def dmu_dx(x: 'np.ndarray', g_reduced: 'np.ndarray') -> 'np.ndarray'`

Chemical potential derivatives dmu_i/dx_j with x_n dependent, shape (..., n, n-1).

From mu_i = G + sum_k (delta_ik - x_k) dG/dx_k (k < n), differentiating gives
dmu_i/dx_j = sum_k (delta_ik - x_k) G''_kj. This equals the bracket
(dmu_i/dx_j - dmu_i/dx_n) of Andersson and Agren, J. Appl. Phys. 72 (1992)
1350, as written in eq. (2) of Jonsson, ISIJ Int. 35 (1995) 1415.

### `def reduced_hessian(hessian: 'np.ndarray') -> 'np.ndarray'`

Curvature of G with x_n eliminated, G''_kj = H_kj - H_kn - H_nj + H_nn, shape (..., n-1, n-1).

### `def thermodynamic_factor(x: 'np.ndarray', dmu: 'np.ndarray', T) -> 'np.ndarray'`

Dimensionless thermodynamic factor (x_i / RT) dmu_i/dx_j, shape (..., n, n-1).

For an ideal solution the upper (n-1) block is the identity and the last
row is -1, the multicomponent analogue of d ln a / d ln x = 1.

## `janus_kinetics.kawin_bridge` (optional, needs kawin, `pip install janus-kinetics[kawin]`)

### `class KawinPrecipitation(times: 'np.ndarray', mean_radius: 'np.ndarray', number_density: 'np.ndarray', volume_fraction: 'np.ndarray', matrix_composition: 'np.ndarray', driving_force: 'np.ndarray', strength: 'np.ndarray | None' = None) -> None`

kawin's states of one precipitate in the names of :class:`MulticomponentResult`: times (s), mean
radius (m), number density (m^-3), volume fraction, the matrix mole fractions (t, n-1) in janus's order,
the driving force per volume of precipitate (J/m^3, as kawin keeps it), and the yield strength (Pa)
if a strength model was coupled.


### `def independent_copy(db: 'Database') -> 'Database'`

A copy of ``db`` sharing nothing with it. pycalphad's deepcopy of a Database copies the parameters but
shares the phase and function dictionaries, and kawin adds a DIS_ phase for each order-disorder model to
the database it is given, which then breaks janus's equilibria on the original.

### `def kawin_elements(elements: 'Sequence[str]', dependent: 'str | None' = None) -> 'list[str]'`

kawin's element list for janus's ``elements``: the solvent first, the independent elements in janus's
order, then VA.

### `def kawin_history(model, strength_model=None) -> 'KawinPrecipitation'`

The solved ``model``'s states (its first precipitate) as a :class:`KawinPrecipitation`; with
``strength_model`` (coupled before solving) also kawin's total yield strength at each state.

### `def kawin_model(thermodynamics, precipitate_phase: 'str', alloy: 'Sequence[float]', T, settings, solutes: 'Sequence[str]', atoms_per_cell: 'float | None' = None)`

An unsolved kawin PrecipitateModel for the precipitate, alloy and temperature that
:func:`solve_multicomponent_precipitation` takes, with janus's :class:`NucleationSettings`: the same
interfacial energy, the same molar volume for both phases, homogeneous nucleation on
``settings.site_density`` sites per m^3. kawin takes the unit-cell volume and ``atoms_per_cell``; by
default the count is the one that makes its cell edge equal ``settings.lattice_parameter`` (4 for fcc
when the two are consistent), and the cell volume is then that count times the atomic volume, so
kawin's molar volume is ``settings.molar_volume`` either way. ``T`` is a temperature in K or a kawin TemperatureParameters.

Couple a kawin strength model with ``model.addCouplingModel`` before solving. Only bulk nucleation
without misfit energy is carried over; anything else raises, since kawin describes it differently.

### `def kawin_thermodynamics(db: 'Database', matrix_phase: 'str', precipitate_phases: 'Sequence[str]', elements: 'Sequence[str]', mobility_db: 'Database | None' = None, dependent: 'str | None' = None, **kwargs)`

kawin's MulticomponentThermodynamics on janus's database (merged with ``mobility_db`` by
:func:`merge_mobilities` when the mobilities sit in a file of their own) and janus's element order.
kawin gets a copy of the database, never ``db`` itself (see :func:`independent_copy`). Keyword arguments
go to kawin.

### `def local_compositions(z: 'np.ndarray', x: 'np.ndarray', positions: 'Sequence[float]') -> 'np.ndarray'`

Mole fractions (len(positions), n-1) of a janus profile ``x`` (cells, n-1) at cell centres ``z``,
interpolated linearly to ``positions`` (m), for instance the dendrite axis and the arm edge of a
homogenized casting to age in kawin.

### `def merge_mobilities(db: 'Database', mobility_db: 'Database', phase: 'str', elements: 'Sequence[str]') -> 'Database'`

A copy of ``db`` holding also the MQ and MF parameters of ``phase`` from ``mobility_db``, so a program
that reads one database (kawin does) sees janus's thermodynamics and mobilities together.

A mobility database often writes the phase with a vacancy-only sublattice the thermodynamic one lacks,
fcc as (Al,Cr,Ni)(Va) against (Al,Cr,Ni), say; the two are the same lattice, so each parameter is
re-keyed onto ``db``'s sublattices. Parameters naming elements outside ``elements`` are left out, as are
functions the copied parameters do not use. Parameters of that kind already in ``db`` are kept and a
clash raises, since two values for one mobility would be summed.
