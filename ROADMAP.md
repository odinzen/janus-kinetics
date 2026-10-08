# Roadmap

Phases 1 to 4 make up version 0.1.0: single-phase and two-phase multicomponent
diffusion, precipitation, worked examples and documentation. Each model is built
on the published method named for it and is accepted only when its validation
passes. Every reference is checked against Crossref by tools/check_references.py.

## Carried over from phase 1

- Ferromagnetic mobility factor of Jonsson (Z. Metallkd. 83, 1992, 349). The hook exists (`magnetic_factor`); the formula
  goes in once the papers are read, with a test against the Fe-rich data of
  Jonsson, ISIJ Int. 35 (1995) 1415, Figs. 5 to 8, where the magnetic factor matters.
- Benchmark against measured ternary interdiffusion matrices: done for Campbell
  et al. (2002) Table 7 (fcc Ni-Al-Cr, 1100 C), with the fcc of Dupin, Ansara and
  Sundman, Calphad 25 (2001) 279 (pycalphad's alcrni.tdb, checked parameter by
  parameter against their appendix) because Kattner's is not public. The matrices
  agree with Campbell's calculated ones within 6 %.
- Interstitial solutes (C, N, O) on a vacancy sublattice, with u-fractions in place
  of mole fractions: done (InterstitialPhase, Andersson and Agren eqs. 62-65).
- Volume-fixed frame in the solver, with a composition-dependent molar volume: done
  for constant partial molar volumes (planar solver, in concentrations). Partial
  volumes that vary with composition, and the other solvers, still use one V_m.
- Adaptive time stepping: done for the planar solver (solve_planar, tolerance=);
  the moving-interface, homogenization and reaction-layer solvers still take fixed steps.
- Oxides: cation tracer and chemical diffusion done (OxideCationDiffusion,
  Hallstrom et al. 2011). Magnetite checked against their Figs. 3-4 (the Fig. 4
  level is 0.2 decades off their own eq. 7, a gap inside the paper). Open: their
  Figs. 7-8, 0.4-0.7 decades above our hematite with the same
  Kjellqvist et al. (2008) thermodynamics, cause not found; the general sublattice
  formalism of Agren (1982).
- Oxide scale growth: done for a compact scale of one oxide on its metal
  (parabolic_scale_growth, Cr2O3 after Hallstrom et al., Solid State Ionics 240
  (2013) 41). Open: their Fig. 6 Cr tracer lines, 0.2-0.75 decades below their own
  eq. (11), which their thicknesses do follow; the oxygen share of growth, about
  10 % in their Fig. 11 against 18 % from their eqs. (11)-(12); scales of more than
  one oxide, alloys, and the start-up transient their numerical runs show.
- Ordered B2 phases: done for mobilities and diffusivities at a point (B2Phase,
  Helander and Agren 1999). Open: the Al-rich vacancy range (0.3 decades off their
  Fig. 2 with Ansara's own B2 model as well as Dupin's reformulation, so the cause
  is their vacancy treatment, which the paper does not spell out), Al-Fe (their eq. 8 for
  the magnetic term is garbled as printed), and use inside the solvers.

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
  once per temperature for binaries and ternaries, or computed on demand (their
  section 2.5) for any number of elements, checked in a quaternary against the
  phase 1 solver.

Stoichiometric reaction layers (added after phase 4): binary couples of line
compounds with Wagner's integrated diffusivity (van Loo 1990, eq. 17), lattice
and marker motion, finite films and anneal schedules. Validated against van Loo's
Ni3Sn4/Ni3Sn couple (pp. 64-65) and wustite couples (p. 78), and multilayer
growth against Wagner's general formula (Acta Metall. 17 (1969) 99, eq. 20),
which recovers each layer's integrated diffusivity from all the layer widths to
round-off. Plates, cylinders and spheres converting through one such layer follow van
Loo's eqs. 25-27, checked against an independent derivation and his Figs.
2.20-2.21. Interface-reaction control (Dybkov, cited by van Loo) and first-phase
selection by nucleation are not modelled.

Microsegregation: done (solidification module; Kobayashi 1988 exact solution,
Bocklund et al. 2020 Scheil-Gulliver). Open: back-diffusion with CALPHAD tie-lines
and several elements (the DICTRA solidification case of Lippard et al. 1998, whose
AerMet100 data need commercial databases), and measured homogenization data.

Cylindrical and spherical geometry: done for solve_planar and solve_layers
(Zener 1949 eq. 12 with lambda = 2, 3; series solutions), and for solve_homogenized
(checked against both). Not yet in the reaction-layer solver.

Left for later: graded grids next to a moving interface, layers that nucleate
during a run (today a reaction layer must exist from the start), and an adaptive
(on-demand) equilibrium table for four or more elements (done: OnDemandEquilibriumTable,
isothermal only, as in the paper).

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

- driving force and multicomponent growth from the full CALPHAD Gibbs energies:
  done (multicomponent_precipitation, Svoboda et al. 2004 eq. 45, checked against
  LSW, Sudbrack et al. 2006 and Booth-Morrison et al. 2008), with precipitate
  compositions that evolve after nucleation (part I eqs. 27-39), misfit energy
  (Appendix A), nucleation on grain boundaries, edges and corners (Clemm and Fisher
  1955), temperature schedules and several phases or sites competing for one matrix
  (part II section 3.5). Site counts from grain size (Cahn 1956 eq. 1) and dislocation
  density: done. Open there: the Thermo-Calc curvatures of Sudbrack
  et al. (2007) Table 4 (ours 10-30 % higher for the same database);
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
