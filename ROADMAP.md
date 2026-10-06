# Roadmap

Phase 1 (this version) covers single-phase multicomponent diffusion. Each later
phase is built on the published method named for it and is accepted only when
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

## Phase 2. Moving boundaries and microstructure-scale transport

**Planar sharp-interface moving boundary**, for a two-phase diffusion couple and
for layer growth. Each interface carries local equilibrium, solved together with
the flux balance across it (the Stefan condition), on grids that follow the
interface.

- Agren, J. Phys. Chem. Solids 43 (1982) 385, numerical treatment of diffusional
  reactions in multicomponent alloys.
- Crusius, Inden, Knoop, Hoglund and Agren, Z. Metallkd. 83 (1992) 673, on the
  numerical treatment of moving boundary problems.
- Validation against the exact parabolic growth solution of Zener, J. Appl. Phys.
  20 (1949) 950, for a binary planar interface, and against mass balance of the
  whole couple.

**Kirkendall marker velocity** from the intrinsic (lattice-frame) fluxes.

- Darken, Trans. AIME 175 (1948) 184; Smigelskas and Kirkendall, Trans. AIME 171
  (1947) 130.
- Validation against the closed-form marker shift of a binary couple with
  constant intrinsic diffusivities.

**Grain-boundary short-circuit diffusion** through an effective diffusivity.

- Hart, Acta Metall. 5 (1957) 597.
- Validation with the limiting cases f -> 0 (lattice only) and f -> 1.

**Homogenization model** for multiphase diffusion couples, which is the supported
route for compositions that phase 1 rejects as inside a two-phase field.

- Larsson and Engstrom, Acta Mater. 54 (2006) 2431.
- Validation against the diffusion couples reported in that paper if the required
  databases are available, otherwise against a sharp-interface phase 2
  calculation on the same couple.

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
