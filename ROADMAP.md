# Roadmap

Phases 1 and 2 (this version) cover single-phase and two-phase multicomponent
diffusion. Each phase is built on the published method named for it and is accepted only when
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

## Phase 3. Precipitation

**Kampmann-Wagner numerical (KWN) method**, which evolves a size-class
distribution by nucleation, growth and coarsening.

- Kampmann and Wagner, in Decomposition of Alloys: the Early Stages (Pergamon,
  1984) 91.
- Wagner, Kampmann and Voorhees, in Phase Transformations in Materials
  (Wiley-VCH, 2001) 309.

**Classical nucleation** with the Zeldovich factor, the atomic attachment rate
from the phase 1 mobilities, and the incubation time.

- Russell, Adv. Colloid Interface Sci. 13 (1980) 205.

**Gibbs-Thomson** capillarity shift of the interface compositions, from pycalphad.

**Multicomponent growth rate**, either the thermodynamic extremal principle model
of Svoboda, Fischer, Fratzl and Kozeschnik, Mater. Sci. Eng. A 385 (2004) 166, or
the approach of Morral and Purdy, Scripta Metall. Mater. 30 (1994) 905. The choice
will be made on which one the phase 1 thermodynamic factor supports without new
approximations.

Validation:

- long-time coarsening approaches the LSW t^(1/3) law and its size distribution
  (Lifshitz and Slyozov, J. Phys. Chem. Solids 19 (1961) 35; Wagner, Z.
  Elektrochem. 65 (1961) 581);
- single-particle growth reproduces the Zener parabolic solution;
- total solute is conserved between matrix and precipitates.

## Phase 4. Release

- Worked examples (diffusion couple, carburizing-type surface boundary, a
  precipitation case), each tied to a test.
- API documentation and a short theory note citing the equations used.
- References verified against Crossref.
- Zenodo-ready release with CITATION.cff and a versioned DOI.
