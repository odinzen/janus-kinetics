# API reference

Generated from the docstrings by `tools/gen_api.py`; do not edit by hand.

## `calphad_kinetics`

### `R`

Constant, value `8.3145`.

## `calphad_kinetics.datasets`

### `def jonsson1995_bcc_crfeni() -> 'Database'`

Atomic mobilities of Cr, Fe and Ni in bcc Cr-Fe-Ni, paramagnetic part.

Transcribed from the Appendix (p. 1421) of B. Jonsson, ISIJ Int. 35 (1995)
1415. It holds mobility (MQ/MF) parameters only, to be combined with a
thermodynamic description through ``SolutionPhase(..., mobility_db=...)``. The ferromagnetic
factor of the paper's eq. (3) is not included. It is largest below the Curie
temperature but still lowers Fe-rich mobilities some way above it, so results
for Fe-rich alloys within a few hundred kelvin of Tc are approximate.

## `calphad_kinetics.diffusivity`

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

## `calphad_kinetics.errors`

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


## `calphad_kinetics.grain_boundary`

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

## `calphad_kinetics.homogenization`

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

### `def average_kinetic_factor(gamma: 'np.ndarray', fraction: 'np.ndarray', how: 'Averaging') -> 'np.ndarray'`

Gamma* from per-phase Gamma (P, n) and volume fractions (P,), shape (n,).

Larsson and Hoglund (2009): Wiener upper and lower bounds, eqs. (6)-(7), and the
Hashin-Shtrikman bounds of eqs. (8)-(11), where the reference phase alpha is the
one with the lowest (lower bound) or highest (upper bound) Gamma_k present.

### `def solve_homogenized(z_faces: 'np.ndarray', x0: 'np.ndarray', table: 'LocalEquilibriumTable', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, tol: 'float' = 1e-10, max_iter: 'int' = 60) -> 'DiffusionResult'`

Integrate planar multiphase diffusion with the homogenization model.

Backward Euler on cell compositions. Each iteration linearizes the potentials
in the composition change (Newton) and holds Gamma* and the frame correction
at the current iterate (Picard). Face fluxes use the potential difference
between neighbouring cells, so transport across a phase boundary is driven by
the true potential jump rather than by a diffusivity at a mean composition.

## `calphad_kinetics.interstitial`

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

## `calphad_kinetics.mobility`

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

## `calphad_kinetics.moving_boundary`

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

### `class MovingBoundaryResult(times: 'np.ndarray', interface: 'np.ndarray', z_left: 'np.ndarray', x_left: 'np.ndarray', z_right: 'np.ndarray', x_right: 'np.ndarray', interface_left: 'np.ndarray', interface_right: 'np.ndarray') -> None`

MovingBoundaryResult(times: 'np.ndarray', interface: 'np.ndarray', z_left: 'np.ndarray', x_left: 'np.ndarray', z_right: 'np.ndarray', x_right: 'np.ndarray', interface_left: 'np.ndarray', interface_right: 'np.ndarray')

- `inventory(self) -> 'np.ndarray'`: Integral of each independent mole fraction over both phases, shape (n_t, m).

### `class PhaseDissolvedError`

The interface reached an outer boundary, so one phase has vanished.


### `class StackResult(times: 'np.ndarray', interfaces: 'np.ndarray', faces: 'list', x: 'list', interface_left: 'np.ndarray', interface_right: 'np.ndarray', names: 'tuple' = ()) -> None`

Saved state of a layered run; index [t] is a saved time, [j] a layer, [i] an interface.

- `inventory(self) -> 'np.ndarray'`: Integral of each independent mole fraction over the whole stack, shape (n_t, m).
- `profile(self, index: 'int' = -1) -> 'tuple[np.ndarray, np.ndarray]'`: Cell centres (m) and compositions through the whole stack at one saved time.
- `thickness(self) -> 'np.ndarray'`: Layer thicknesses [t, j] in m.

### `def solve_layers(positions, layers: 'list[Layer]', conditions: 'list', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, tol: 'float' = 1e-10, picard_tol: 'float' = 1e-10, max_newton: 'int' = 30, max_picard: 'int' = 50) -> 'StackResult'`

Integrate planar diffusion through a stack of phase layers with moving interfaces.

``positions`` are the layer boundaries from 0 to the stack thickness (len(layers)
+ 1 values); the inner ones are the initial interfaces. ``conditions[i]`` holds
interface i, between layers i and i+1, at fixed compositions or in local
equilibrium. Every interface moves to conserve every element (the flux balance
of the module docstring), and all interface velocities and operating
tie-lines are solved together each step. A reaction layer must be present from
the start, if only a few nanometres thick. If a layer is consumed,
:class:`PhaseDissolvedError` is raised; it and any :class:`ConvergenceError`
carry the results up to the last saved time as ``.result``.

### `def solve_moving_boundary(length: 'float', interface: 'float', x0_left, x0_right, d_left: 'Diffusivity', d_right: 'Diffusivity', condition: 'FixedInterface | LocalEquilibrium', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, tol: 'float' = 1e-10, picard_tol: 'float' = 1e-10, max_newton: 'int' = 30, max_picard: 'int' = 50) -> 'MovingBoundaryResult'`

Integrate two-phase planar diffusion with a moving interface, as two layers of :func:`solve_layers`.

``x0_left`` (N_L, n-1) and ``x0_right`` (N_R, n-1) are the initial profiles on
uniform grids over [0, interface] and [interface, length]; a binary may pass
1D arrays. ``d_left``/``d_right`` are constant matrices or callables like those
of :func:`solve_planar`, for example ``lambda x: alpha.interdiffusion_matrix(x, T)``.
``left``/``right`` are the conditions at z = 0 and z = length.

## `calphad_kinetics.phase`

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
- `interdiffusion_function(self, temperature: 'Temperature', frame: 'Frame' = 'number', partial_volumes=None)`: D~ as a function of (x, t) for the solvers, at a constant T or along a schedule.
- `interdiffusion_matrix(self, x, T, frame: 'Frame' = 'number', partial_volumes: 'np.ndarray | None' = None, magnetic_factor=1.0) -> 'np.ndarray'`: Reduced interdiffusion matrix D~ in m^2/s, shape (..., n-1, n-1).
- `intrinsic_matrix(self, x, T, magnetic_factor=1.0) -> 'np.ndarray'`: Lattice-frame diffusivities D^I_kj in m^2/s, shape (..., n, n-1).
- `lattice_velocity(self, x, dx_dz, T, magnetic_factor=1.0) -> 'np.ndarray'`: Kirkendall (marker) velocity relative to the number-fixed frame, m/s.
- `lattice_velocity_function(self, temperature: 'Temperature')`: Marker velocity as a function of (x, dx_dz, t), for ``solve_planar(..., lattice_velocity=...)``.
- `mobility(self, x, T, magnetic_factor=1.0) -> 'np.ndarray'`: Atomic mobilities in m^2 mol / (J s), including any short-circuit paths.
- `reduced_hessian(self, x, T) -> 'np.ndarray'`: d2G/dx_k dx_j with the dependent element eliminated, J/mol, shape (..., n-1, n-1).
- `thermodynamic_factor(self, x, T) -> 'np.ndarray'`: Dimensionless (x_i/RT) dmu_i/dx_j, shape (..., n, n-1).

## `calphad_kinetics.precipitation`

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

## `calphad_kinetics.schedule`

### `class TemperatureSchedule(times, temperatures)`

Piecewise-linear temperature T(t) in K through (time, temperature) points.

Held at the first temperature before the first point and at the last one after
it, so a single point is an isothermal hold. Ramps, spikes and multi-step anneals
are lists of points, e.g. ``TemperatureSchedule([0, 10, 70, 80], [300, 1300, 1300, 300])``.


## `calphad_kinetics.solver`

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

### `class DiffusionResult(times: 'np.ndarray', z_faces: 'np.ndarray', x: 'np.ndarray', picard_iterations: 'list[int]' = <factory>, markers: 'np.ndarray | None' = None) -> None`

Saved profiles: ``x[t_index, cell, component]`` for the independent components.

- `inventory(self) -> 'np.ndarray'`: Integral of each mole fraction over the domain at every saved time, shape (n_t, n-1).

### `def solve_planar(z_faces: 'np.ndarray', x0: 'np.ndarray', diffusivity: 'Diffusivity', t_end: 'float', dt: 'float', left: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), right: 'Boundary' = Boundary(kind='no_flux', composition=None, held=None), save_times=None, picard_tol: 'float' = 1e-10, max_picard: 'int' = 50, markers=None, lattice_velocity: 'LatticeVelocity | None' = None) -> 'DiffusionResult'`

Integrate planar interdiffusion from ``x0`` to ``t_end``.

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
picard_tol : convergence threshold on the max change in mole fraction
    between Picard iterates.
markers : initial positions (m) of inert Kirkendall markers to follow.
lattice_velocity : callable (x, dx_dz) or (x, dx_dz, t) -> v giving the lattice
    velocity at the interior faces, such as ``lambda x, g: phase.lattice_velocity(x, g, T)``.
    Required with ``markers``.

Returns profiles at t=0 and at every save time (always including ``t_end``).

### `def uniform_faces(length: 'float', n_cells: 'int', origin: 'float' = 0.0) -> 'np.ndarray'`

Face positions (m) of ``n_cells`` equal cells spanning [origin, origin + length].

## `calphad_kinetics.thermo`

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
