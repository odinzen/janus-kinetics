# API reference

Generated from the docstrings by `tools/gen_api.py`; do not edit by hand.

## `janus_kinetics`

### `R`

Constant, value `8.3145`.

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

### `def solve_fisher(t1: 'float', x_max: 'float | None' = None, y_max: 'float | None' = None, dx: 'float' = 0.5, dy: 'float' = 0.25, steps: 'int' = 400) -> 'FisherSolution'`

Solve the scaled eqs. (3a)-(4a) of Fisher (1951) without his approximation.

Backward Euler in t1 on a uniform grid; the slab takes (dC/dx1)_0 from the
second-order one-sided difference his eq. (3b) uses, and each grain row only
diffuses in x1, as eq. (4a) states. The surface row of the slab is held at 1.

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

### `def solve_homogenized(z_faces: 'np.ndarray', x0: 'np.ndarray', table: 'LocalEquilibriumTable | OnDemandEquilibriumTable | TemperatureTable', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, tol: 'float' = 1e-10, max_iter: 'int' = 60, temperature: 'Temperature | None' = None, ideal_flux: 'float' = 0.0) -> 'DiffusionResult'`

Integrate planar multiphase diffusion with the homogenization model.

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


### `class Layer(x0: 'np.ndarray', diffusivity: 'Diffusivity', name: 'str' = '') -> None`

One phase layer of a planar stack.

``x0`` is the initial profile on a uniform grid across the layer, shape (N, n-1)
or (N,) for a binary; ``diffusivity`` is a constant matrix or a callable of
(x) or (x, t), as for :func:`solve_planar`.


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

### `def solve_layers(positions, layers: 'list[Layer]', conditions: 'list', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, tol: 'float' = 1e-10, picard_tol: 'float' = 1e-10, max_newton: 'int' = 30, max_picard: 'int' = 50, geometry: 'Geometry' = 'planar') -> 'StackResult'`

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

### `def solve_moving_boundary(length: 'float', interface: 'float', x0_left, x0_right, d_left: 'Diffusivity', d_right: 'Diffusivity', condition: 'FixedInterface | LocalEquilibrium', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, tol: 'float' = 1e-10, picard_tol: 'float' = 1e-10, max_newton: 'int' = 30, max_picard: 'int' = 50, geometry: 'Geometry' = 'planar') -> 'MovingBoundaryResult'`

Integrate two-phase diffusion with a moving interface, as two layers of :func:`solve_layers`; in a
cylinder or sphere (``geometry``) the left phase is the core and the right one the surrounding shell.

``x0_left`` (N_L, n-1) and ``x0_right`` (N_R, n-1) are the initial profiles on
uniform grids over [0, interface] and [interface, length]; a binary may pass
1D arrays. ``d_left``/``d_right`` are constant matrices or callables like those
of :func:`solve_planar`, for example ``lambda x: alpha.interdiffusion_matrix(x, T)``.
``left``/``right`` are the conditions at z = 0 and z = length.

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
of the homogeneous one. ``sites`` should then count the boundary, edge or corner sites per m^3.


### `class PhaseMinimizer(db: 'Database', phase: 'str', components: 'Sequence[str]')`

Internal equilibrium of one phase at fixed composition: site fractions that minimise its molar
Gibbs energy, found by Newton's method on the constrained problem for many compositions at once.

pycalphad's equilibrium does this too, but too slowly to call for every precipitate class at every
step; here the Gibbs energy, gradient and Hessian are compiled once and each solve starts from the
previous site fractions. The phase must hold no vacancies, so that composition is linear in the
site fractions. Chemical potentials follow from the Lagrange multipliers of the composition
constraints: with x_n dependent, dG/dx_i = -nu_i, mu_n = G + sum_i x_i nu_i and mu_i = mu_n - nu_i.

- `disordered(self, x) -> 'np.ndarray'`: Site fractions with every sublattice at composition ``x``, a starting point.
- `solve(self, x, y0, T: 'float', tol: 'float' = 1e-11, max_iter: 'int' = 60)`: (site fractions (k, ny), G (k,) in J/mol, mu (k, n)) at mole fractions ``x`` (k, n), from ``y0``.

### `class Population(precipitate: 'PrecipitateThermo', settings: 'NucleationSettings', initial: 'dict | None' = None, precipitate_tracer: 'Sequence[float] | None' = None) -> None`

One precipitate phase nucleating at one kind of site, for :func:`solve_competing_precipitation`.
``initial`` and ``precipitate_tracer`` are as in :func:`solve_multicomponent_precipitation`. A phase
may form several populations, in the bulk and on grain boundaries say, as part II (p. 158) creates
one kinetic object per phase and nucleation site.


### `class PrecipitateThermo(db: 'Database', phase: 'str', components: 'Sequence[str]', composition: 'Sequence[float] | None' = None)`

Gibbs energy of the precipitate phase from pycalphad, with its internal degrees of freedom
(ordering, sublattice occupancy) at equilibrium for each composition.

``components`` are the elements in the matrix's order, dependent element last.
A ``composition`` fixes the precipitate's mole fractions (a line compound or a
chosen stoichiometry); without it nuclei take the composition of largest driving force.

- `gm(self, x: 'Sequence[float]', T: 'float') -> 'float'`: Molar Gibbs energy (J per mole of atoms) at mole fractions ``x``.
- `nucleus(self, mu_matrix: 'np.ndarray', T: 'float') -> 'tuple[np.ndarray, float]'`: (composition, molar driving force in J/mol) of the precipitate with the largest driving force
- `site_fractions(self, x: 'Sequence[float]', T: 'float') -> 'np.ndarray'`: Equilibrium site fractions of the phase at mole fractions ``x``, in pycalphad's order, from the

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

### `def solve_competing_precipitation(matrix: 'SolutionPhase', populations: 'Sequence[Population]', alloy: 'Sequence[float]', T: 'Temperature', t_end: 'float', save_times: 'Sequence[float]', t_start: 'float' = 0.001, max_temperature_change: 'float' = 1.0, max_radius_change: 'float' = 0.02, max_matrix_change: 'float' = 0.0002, max_composition_change: 'float' = 0.001, max_classes: 'int' = 400, min_classes: 'int' = 100, smallest_radius: 'float' = 1e-10, nucleus_update: 'float' = 0.001, min_class_number: 'float' = 1e-06) -> 'list[MulticomponentResult]'`

Several precipitate populations drawing on one matrix, one result per population in the order
given. Part I's mass balance, eq. (4), sums over precipitates of every kind, and part II (section
3.5) runs such competitions to follow precipitation sequences.

Each population nucleates, grows and coarsens as in :func:`solve_multicomponent_precipitation`,
whose arguments these are. The step is common to all of them: its matrix-change limits count the
depletion by every population together, and shrinking classes of all populations share one
dissolution budget. All populations need the same molar volume, part I's single Omega (p. 167),
with which volume fractions carry the mass balance.

### `def solve_multicomponent_precipitation(matrix: 'SolutionPhase', precipitate: 'PrecipitateThermo', alloy: 'Sequence[float]', T: 'Temperature', t_end: 'float', settings: 'NucleationSettings', save_times: 'Sequence[float]', t_start: 'float' = 0.001, max_temperature_change: 'float' = 1.0, max_radius_change: 'float' = 0.02, max_matrix_change: 'float' = 0.0002, max_composition_change: 'float' = 0.001, max_classes: 'int' = 400, min_classes: 'int' = 100, smallest_radius: 'float' = 1e-10, nucleus_update: 'float' = 0.001, min_class_number: 'float' = 1e-06, initial: 'dict | None' = None, precipitate_tracer: 'Sequence[float] | None' = None) -> 'MulticomponentResult'`

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


### `class SolutionPhase(db: 'Database', phase_name: 'str', elements: 'list[str] | tuple[str, ...]', dependent: 'str | None' = None, mobility_db: 'Database | None' = None, short_circuit: 'ShortCircuit | None' = None)`

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


### `class PrecipitationResult(times: 'np.ndarray', radius_edges: 'np.ndarray', number: 'np.ndarray', matrix_fraction: 'np.ndarray', critical_radius: 'np.ndarray', steps: 'int' = 0, rejected: 'int' = 0) -> None`

PrecipitationResult(times: 'np.ndarray', radius_edges: 'np.ndarray', number: 'np.ndarray', matrix_fraction: 'np.ndarray', critical_radius: 'np.ndarray', steps: 'int' = 0, rejected: 'int' = 0)

- `mean_radius` (property): R_mean = sum R_i N_i / sum N_i (eq. 20); NaN before any precipitate exists.
- `number_density` (property): N = sum_i N_i (eq. 19).
- `volume_fraction` (property): f_T = (4/3) pi sum_i N_i R_i^3 (eq. 21).

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


### `class ReactionResult(times: 'np.ndarray', thickness: 'dict[str, np.ndarray]', interfaces: 'list[np.ndarray]', markers: 'np.ndarray', consumed: 'list[tuple[float, str]]') -> None`

Layer thicknesses (m) of every phase at ``times``.

``thickness[name]`` is inf for a semi-infinite end member and 0 once a phase is
used up. ``interfaces[i]`` holds the interface positions at ``times[i]`` in the
anchored end member's frame, original interface at 0 for a fresh couple.
``markers`` has one column per marker. ``consumed`` lists (time, name) for
every end member used up during the run.


### `def integrated_diffusivity(left: 'LineCompound', layer: 'LineCompound', right: 'LineCompound', T: 'float') -> 'float'`

Wagner's D_int (van Loo eq. 17) of ``layer`` between its neighbours at T, in m^2/s.

### `def parabolic_growth(phases: 'Sequence[LineCompound]', temperature: 'float', anchor: 'str' = 'left') -> 'ParabolicGrowth'`

Parabolic growth constants of every layer in a fresh couple at constant T.

All layers start together (van Loo p. 61, item c). With d_k = delta_k sqrt(t)
every velocity scales as 1/sqrt(t), so delta_k = 2 (V_k - V_(k-1)) where V are
the interface velocity coefficients computed with thicknesses delta. That
nonlinear system is solved in log(delta).

### `def solve_reaction_layers(phases: 'Sequence[LineCompound]', t_end: 'float', temperature: 'Temperature', thickness: 'Sequence[float] | None' = None, end_thickness: 'tuple[float | None, float | None]' = (None, None), markers: 'Sequence[float]' = (0.0,), save_times=None, anchor: 'str | None' = None, rtol: 'float' = 1e-09) -> 'ReactionResult'`

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

### `class ScheilResult(temperatures: 'np.ndarray', fraction_solid: 'np.ndarray', liquid_composition: 'np.ndarray', solids: 'list', components: 'tuple') -> None`

Temperatures (K), fraction solid, liquid mole fractions (n_T, n) and the solid formed at each step:
for every temperature step the phases that formed with their amounts (fraction of the alloy) and
mole fractions.

- `increments(self) -> 'tuple[np.ndarray, np.ndarray]'`: (fraction-solid edges, independent mole fractions of the solid laid down in each step), the input
- `solid_phase_fractions(self) -> 'dict'`: Amount of each solid phase formed over the whole path, as a fraction of the alloy.

### `def kobayashi(k: 'float', gamma: 'float', fs, x=None, terms: 'int' = 10000) -> 'np.ndarray'`

Exact solution of the Brody-Flemings model over C0 (Kobayashi eqs. 37-40): the interface solid
composition at fraction solid ``fs``, or with ``x`` (x* = x / X, 0 at the dendrite spine) the profile
in the solid. The series converges for 0 < k < 2 and gamma > 0; ``terms`` must grow as f_s
approaches 1 with k small, where the terms fall off slowly.

### `def scheil_gulliver(db: 'Database', components: 'Sequence[str]', phases: 'Sequence[str]', composition: 'Sequence[float]', start: 'float', step: 'float' = 1.0, liquid: 'str' = 'LIQUID', stop_fraction: 'float' = 0.0001, t_min: 'float' = 300.0) -> 'ScheilResult'`

Scheil-Gulliver solidification by the algorithm Bocklund et al., Materialia 11 (2020) 100689, give
(section 2): from ``start`` (K, above the liquidus) the remaining liquid is equilibrated at each
temperature, the solid that forms is set aside and never changes again, the liquid's fraction is
reduced by the solid's, and its composition becomes that of the equilibrium liquid, until less than
``stop_fraction`` liquid is left. ``composition`` is the n-1 independent mole fractions in the order of
``components`` (the last element dependent).

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
