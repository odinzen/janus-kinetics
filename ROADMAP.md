# Roadmap

Phases 1 to 4 make up version 0.1.0: single-phase and two-phase multicomponent
diffusion, precipitation, worked examples and documentation. Each model is built
on the published method named for it and is accepted only when its validation
passes. Every reference is checked against Crossref by tools/check_references.py.

## Carried over from phase 1

- Ferromagnetic mobility factor of Jonsson (Z. Metallkd. 83, 1992, 349). The hook exists (`magnetic_factor`); the formula
  goes in once the papers are read, with a test against the Fe-rich data of
  Jonsson, ISIJ Int. 35 (1995) 1415, Figs. 5 to 8, where the magnetic factor matters.
- Benchmark against measured ternary interdiffusion matrices. Campbell et al.
  (2002) Table 7 tabulates measured fcc Ni-Al-Cr matrices at 1100 C, and their
  mobilities ship with the package; an open fcc Ni-Al-Cr thermodynamic
  description (Dupin, Ansara and Sundman, Calphad 25 (2001) 279) is still needed.
- Interstitial solutes (C, N) on a vacancy sublattice, with u-fractions in place
  of mole fractions.
- Volume-fixed frame in the solver, with a composition-dependent molar volume.
- Adaptive time stepping.

## Phase 2. Moving boundaries and microstructure-scale transport (done)

Each item is validated against the published source named in its docstring.

- Kirkendall marker velocity from the intrinsic fluxes (van Loo 1990, eqs. 4,
  5, 9, 11). Validated by recovering both intrinsic diffusivities from a
  simulated couple with van Loo's eq. 11.
- Short-circuit diffusion by Hart's average (Hart 1957, eq. 5, derived for
  dislocations; its use for grain boundaries and multicomponent mobilities is an
  extension). Diffusion along isolated boundaries from a surface source follows
  Fisher (1951), solved exactly and checked against a Laplace-transform solution.
- Planar two-phase moving interface with local equilibrium and the operating
  tie-line (Zener's eq. 8 generalized). Validated against Zener's eqs. 10-12, a
  two-sided similarity solution and binary pycalphad tie-lines; the ternary
  operating tie-line is checked for local equilibrium, constancy in time and
  conservation, not against an outside reference. The interface gradient
  is one-sided, so the position converges at first order in the grid step.
- Homogenization model (Larsson and Hoglund 2009). Validated against the phase 1
  solver and the sharp-interface solver; the paper's multiphase examples use
  commercial databases and are not reproduced. Local equilibrium is tabulated
  once per temperature, so it is limited to binaries and ternaries.

Stoichiometric reaction layers (added after phase 4): binary couples of line
compounds with Wagner's integrated diffusivity (van Loo 1990, eq. 17), lattice
and marker motion, finite films and anneal schedules. Validated against van Loo's
Ni3Sn4/Ni3Sn couple (pp. 64-65) and wustite couples (p. 78). Wagner's general
multilayer formula (Acta Metall. 17 (1969) 99) has not been opened, so the
multilayer case is checked by conservation and against the single-layer limits
only. Interface-reaction control (Dybkov, cited by van Loo) and first-phase
selection by nucleation are not modelled.

Left for later: graded grids next to a moving interface, layers that nucleate
during a run (today a reaction layer must exist from the start), and an adaptive
(on-demand) equilibrium table for four or more elements.

## Phase 3. Precipitation (done for binaries)

Built on Perez, Dumont and Acevedo-Reyes, Acta Mater. 56 (2008) 2119, with the
corrigendum in Acta Mater. 57 (2009) 1318.

- Kampmann-Wagner numerical method: fixed size classes with the class fluxes of
  their eq. 24, taken implicitly; steps limited by the change in R* as in their
  section 3.4.1, and by the change in X and the mean radius.
- Classical nucleation: eqs. 3-7 (Zeldovich factor, condensation rate,
  incubation time, nucleation radius) as corrected.
- Zener growth with the Gibbs-Thomson interface composition, eqs. 9-10, and the
  solute mass balance, eqs. 21-22.
- Matrix solubility and precipitate composition from `pycalphad.equilibrium`.

Validation: each formula against a hand evaluation; Zener parabolic growth at two
volume ratios; the mean radius meeting the critical radius after growth; the
coarsening rate falling toward the LSW value of eq. 11; insensitivity to the
size grid and step; complete dissolution returning all solute. Their Fig. 7 is
not used. Its plateau (R_mean = R* = 4.8 nm, X = 0.0142) needs
2 gamma v/kT = 1.7 nm through eq. 9, while Table 1 gives 0.70 nm.

Left for later, each needing a source to be opened first:

- driving force and Gibbs-Thomson shift from the full CALPHAD Gibbs energies
  rather than the dilute form of eq. 9;
- multicomponent precipitates and growth (Perez et al. eq. 40 is one route;
  Svoboda, Fischer, Fratzl and Kozeschnik, Mater. Sci. Eng. A 385 (2004) 166,
  is another);
- heterogeneous nucleation and non-isothermal treatments;
- the mean-radius approach, whose coarsening fraction (eq. 17) is defined only
  for 0.99 R* < R_mean < 1.01 R* in the paper.

## Phase 4. Examples, documentation and release preparation (done)

- Worked examples in examples/, each run at coarse resolution by tests/test_examples.py.
- docs/theory.md (equations with sources), docs/api.md (generated from the
  docstrings, kept current by a test) and docs/references.md (written from Crossref).
- CITATION.cff, .zenodo.json and CHANGELOG.md for version 0.1.0, with a test that
  the version and authorship agree across them.

The release itself (public repository, Zenodo DOI, optionally PyPI) is left to the
maintainer.
