# Roadmap

Phases 1 to 3 (this version) cover single-phase and two-phase multicomponent
diffusion and precipitation. Each phase is built on the published method named for it and is accepted only when
its listed validation passes. Bibliographic details below are working
references; they will be checked against Crossref before the release in phase 4.

## Carried over from phase 1

- Ferromagnetic mobility factor of Jonsson (Z. Metallkd. 85, 1994, 498 and 502;
  Z. Metallkd. 83, 1992, 349). The hook exists (`magnetic_factor`); the formula
  goes in once the papers are read, with a test against Jonsson, ISIJ Int. 35
  (1995) 1415, Figs. 5 to 8, which are in the ferromagnetic range.
- Benchmark against measured Cu-Ni-Zn fcc interdiffusion matrices (Kim, 1982, as
  named in the plan; citation to be confirmed) once a thermodynamic and mobility
  description is chosen.
- Interstitial solutes (C, N) on a vacancy sublattice, with u-fractions in place
  of mole fractions.
- Volume-fixed frame in the solver, with a composition-dependent molar volume.
- Adaptive time stepping.

## Phase 2. Moving boundaries and microstructure-scale transport (done)

Each item is validated against the published source named in its docstring.

- Kirkendall marker velocity from the intrinsic fluxes (van Loo 1990, eqs. 4,
  5, 9, 11). Validated by recovering both intrinsic diffusivities from a
  simulated couple with van Loo's eq. 11.
- Short-circuit diffusion by Hart's average (Hart 1957, eq. 5). Hart's regime
  only; isolated boundaries (Fisher 1951, type B) are not modelled.
- Planar two-phase moving interface with local equilibrium and the operating
  tie-line (Zener 1949, eq. 8). Validated against Zener's eqs. 10-12, a
  two-sided similarity solution and pycalphad tie-lines. The interface gradient
  is one-sided, so the position converges at first order in the grid step.
- Homogenization model (Larsson and Hoglund 2009). Validated against the phase 1
  solver and the sharp-interface solver; the paper's own examples need
  commercial databases and are not reproduced. Local equilibrium is tabulated
  once per temperature, so it is limited to binaries and ternaries.

Left for later: graded grids next to a moving interface, more than two phases
in the sharp-interface solver, and an adaptive (on-demand) equilibrium table
for four or more elements.

## Phase 3. Precipitation (done for binaries)

Built on Perez, Dumont and Acevedo-Reyes, Acta Mater. 56 (2008) 2119, with the
corrigendum in Acta Mater. 57 (2009) 1318.

- Kampmann-Wagner numerical method: fixed size classes with the class fluxes of
  their eq. 24, taken implicitly; adaptive steps per their section 3.4.1.
- Classical nucleation: eqs. 3-7 (Zeldovich factor, condensation rate,
  incubation time, nucleation radius) as corrected.
- Zener growth with the Gibbs-Thomson interface composition, eqs. 9-10, and the
  solute mass balance, eqs. 21-22.
- Matrix solubility and precipitate composition from `pycalphad.equilibrium`.

Validation: each formula against a hand evaluation; Zener parabolic growth at two
volume ratios; the mean radius meeting the critical radius after growth; the
coarsening rate falling toward the LSW value of eq. 11; size-grid and step
convergence; solute balance. Their Fig. 7 is not used: its plateau
(R_mean = R* = 4.8 nm, X = 0.0142) needs 2 gamma v/kT = 1.7 nm through eq. 9,
while Table 1 gives 0.70 nm.

Left for later, each needing a source to be opened first:

- driving force and Gibbs-Thomson shift from the full CALPHAD Gibbs energies
  rather than the dilute form of eq. 9;
- multicomponent precipitates and growth (Perez et al. eq. 40 is one route;
  Svoboda, Fischer, Fratzl and Kozeschnik, Mater. Sci. Eng. A 385 (2004) 166,
  is another);
- heterogeneous nucleation and non-isothermal treatments;
- the mean-radius approach, whose coarsening fraction (eq. 17) is defined only
  for 0.99 R* < R_mean < 1.01 R* in the paper.

## Phase 4. Release

- Worked examples (diffusion couple, carburizing-type surface boundary, a
  precipitation case), each tied to a test.
- API documentation and a short theory note citing the equations used.
- References verified against Crossref.
- Zenodo-ready release with CITATION.cff and a versioned DOI.
