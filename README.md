# calphad-kinetics

Diffusion, moving-boundary and precipitation kinetics on top of
[pycalphad](https://pycalphad.org), built from the published models with every
equation cited to the paper it comes from. Atomic mobilities are read from the
MQ and MF parameters of a Thermo-Calc style database, the thermodynamics come
from pycalphad's Gibbs energy models, and the two combine into multicomponent
diffusion, two-phase and precipitation simulations.

Version 0.1.0 (not yet released). Tested with Python 3.12 and pycalphad 0.11.2.

## What it does

- **Atomic mobility** with composition dependence by end members, Redlich-Kister
  and ternary terms (Andersson and Agren 1992; Jonsson 1995).
- **Thermodynamic factor** from the exact symbolic Hessian of pycalphad's molar
  Gibbs energy. A composition inside a spinodal stops the single-phase models
  with `UnstableCompositionError`.
- **Interdiffusion matrices** in the number-fixed and volume-fixed frames, and
  intrinsic (lattice-frame) diffusivities.
- **Plates, cylinders and spheres** for the diffusion and moving-interface solvers
  (concentric shells, as in DICTRA), checked against series solutions and Zener's
  growth coefficients for each geometry.
- **Planar diffusion solver**, implicit and conservative, with no-flux or
  fixed-composition boundaries and **Kirkendall marker** tracking (van Loo 1990),
  in the number-fixed frame or, with partial molar volumes, in the volume-fixed
  frame in concentrations, so the molar volume may change with composition.
- **Reading measured couples**: the interdiffusion coefficient from a binary
  penetration curve without a Matano plane, and the intrinsic diffusivities at
  the marker plane, both allowing a composition-dependent molar volume (van Loo
  1990, eqs. 3 and 12).
- **Short-circuit diffusion** by Hart's time-fraction average (Hart 1957, derived
  for dislocations; applied here also to grain boundaries and to multicomponent
  mobilities), and **diffusion down isolated grain boundaries** from a surface
  (Fisher 1951), with the penetration-curve slope used to read D'delta from
  sectioning data. Fisher's closed form overestimates deep penetration; the
  package also solves his equations exactly.
- **Interstitial elements** (C, N, O, H on a vacancy-bearing sublattice) in
  u-fractions (Andersson and Agren 1992, eqs. 62-65), and **surfaces held at a
  chemical potential**, as a carburizing or oxidizing atmosphere imposes, with the
  other elements passing no flux.
- **Moving interfaces in layered stacks**: any number of phase layers, each
  interface in local equilibrium or at fixed compositions, with every interface
  velocity and operating tie-line solved together. Reaction layers grow or are
  consumed; a consumed layer stops the run with the history returned.
- **Stoichiometric reaction layers** (silicides, intermetallics, oxide scales):
  a binary couple of line compounds grown from mobilities and Gibbs energies or
  from measured integrated diffusivities (Wagner's D_int, van Loo 1990), with
  Kirkendall markers, finite films that are used up in turn, and anneal schedules.
  Plates, wires and powder particles reacting through such a layer follow van
  Loo's conversion relations, which also turn measured conversion into the
  planar growth constant.
- **Non-isothermal anneals** along piecewise-linear temperature schedules
  (ramps, holds, spikes) for the diffusion, layered-stack and (binary)
  homogenization solvers.
- **Ordered B2 phases** (NiAl, FeAl): the ordering contribution to the activation
  energy from the equilibrium site fractions (Helander and Agren 1999), with tracer
  and chemical diffusivities that reproduce their Al-Ni figures on the Ni-rich side.
- **Cation diffusion in non-stoichiometric oxides** (magnetite, hematite): tracer
  and chemical diffusivities of the metal as functions of temperature and oxygen
  pressure, from vacancy jumps on the sublattices of a Calphad oxide description
  (Hallstrom, Hoglund and Agren 2011).
- **Oxide scale growth**: the parabolic rate constant of a compact scale from Cr and
  O transport through bulk and grain boundaries of Cr2O3 (Hallstrom et al. 2013),
  with local equilibrium at the metal/oxide interface. It reproduces their 24 h
  scale thicknesses at 625 and 700 C within 1.5 % and their late-time growth rates
  within 8 %.
- **Homogenization model** for diffusion through multiphase mixtures (Larsson and
  Hoglund 2009), with local equilibrium tabulated up front for binaries and
  ternaries or computed on demand for any number of elements (their section 2.5).
- **Precipitation**: classical nucleation, growth and coarsening on a size-class
  distribution (Perez, Dumont and Acevedo-Reyes 2008, with its 2009 corrigendum),
  for precipitates of fixed composition in a binary matrix.
- **Multicomponent precipitation** with driving forces and nucleus compositions from
  the full CALPHAD Gibbs energies (Svoboda, Fischer, Fratzl and Kozeschnik 2004):
  classes that keep their nucleation composition, ordered precipitates such as
  gamma', compositions that evolve after nucleation, misfit energy, nucleation on
  grain boundaries, edges and corners (Clemm and Fisher 1955) and temperature
  schedules; checked against the LSW limit and against atom-probe kinetics of gamma'
  in two Ni-Al-Cr alloys (Sudbrack et al. 2006; Booth-Morrison et al. 2008).

Three published mobility databases ship with the package:
`jonsson1995_bcc_crfeni()` (bcc Cr-Fe-Ni, Jonsson 1995), `campbell2002_fcc_ni()`
(fcc Ni-Al-Co-Cr-Hf-Mo-Re-Ta-Ti-W for Ni-base superalloys, Campbell, Boettinger
and Kattner 2002) and `helander1999_bcc_alni()` (B2-ordered Al-Ni, Helander and
Agren 1999), each checked against numbers or figures its paper prints. A Cr-O
thermodynamic description, `kjellqvist2008_cro()` (from Kjellqvist, Selleby and
Sundman 2008), serves the Cr2O3 scale model.

The equations are collected in [docs/theory.md](docs/theory.md), the API in
[docs/api.md](docs/api.md), and the references, checked against Crossref, in
[docs/references.md](docs/references.md). What is not done yet is listed in
[ROADMAP.md](ROADMAP.md); the main gaps are the ferromagnetic mobility factor,
scales of several oxides or on alloys, and multicomponent precipitates.

## Install

```
conda create -n calphad-kinetics -c conda-forge python=3.12 pycalphad numpy scipy pytest matplotlib
conda activate calphad-kinetics
pip install -e . --no-deps
pytest
```

The test suite (285 tests, plus 5 skipped for data that is not available) takes
about five minutes. On Windows run inside the activated environment, since
numpy's MKL libraries are only found on the activated path.

## Example

A bcc Fe-30Cr / Fe-70Cr couple at 1200 C with the published mobilities of
Jonsson (1995), which ship with the package, and an ideal-solution Gibbs energy:

```python
import numpy as np
from pycalphad import Database
from calphad_kinetics import SolutionPhase, jonsson1995_bcc_crfeni, solve_planar, uniform_faces

ideal_bcc = Database("""
ELEMENT VA VACUUM 0 0 0 !
ELEMENT CR BCC_A2 51.996 0 0 !
ELEMENT FE BCC_A2 55.847 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE BCC_A2 % 2 1 3 !
CONSTITUENT BCC_A2 :CR,FE:VA: !
PARAMETER G(BCC_A2,CR:VA;0) 298.15 0; 6000 N !
PARAMETER G(BCC_A2,FE:VA;0) 298.15 0; 6000 N !
""")
phase = SolutionPhase(ideal_bcc, "BCC_A2", ["CR", "FE"], dependent="FE", mobility_db=jonsson1995_bcc_crfeni())
T = 1473.15
print(phase.interdiffusion_matrix([0.5], T))  # m^2/s

faces = uniform_faces(2e-3, 400, origin=-1e-3)
z = 0.5 * (faces[1:] + faces[:-1])
x0 = np.where(z < 0, 0.30, 0.70)[:, None]
result = solve_planar(faces, x0, lambda x: phase.interdiffusion_matrix(x, T), t_end=36000.0, dt=180.0,
                      markers=[0.0], lattice_velocity=lambda x, g: phase.lattice_velocity(x, g, T))
print(result.markers[-1])  # Kirkendall marker position after 10 h, m
```

More complete scripts, each tested, are in [examples/](examples): the couple
above with a figure, chromium loss to a surface in a ternary, a precipitation
run, an alpha/beta couple solved by both two-phase models, and a reaction layer
growing between two thin films during a rapid thermal anneal, with both
interfaces following the tie-lines as the temperature ramps, holds and cools,
oxygen dissolving into a metal film from an atmosphere during an anneal, a metal
film reacting into two line compounds, and van Loo's Ni-Sn Kirkendall analysis.

Conventions worth knowing:

- Compositions are mole fractions in the order you list the elements, with the
  dependent element moved last (see `phase.elements`); functions take the
  independent fractions or all of them. Fractions outside [0, 1] raise.
- Interaction parameters follow pycalphad's ordering, with constituents sorted
  alphabetically, so order 1 of `MQ(FCC_A1&NI,ZN,CU;1)` multiplies (x_CU - x_ZN).
  Wildcard parameters such as `MQ(FCC_A1&NI,*:VA;0)` apply at every composition.
- The thermodynamic and mobility databases may declare a phase with or without
  vacancy-only sublattices, as `FCC_A1 :AL,CR,NI:` against `:AL,CR,NI:VA:`; each
  is read in its own layout.
- Outside a parameter's temperature ranges pycalphad extends the nearest range;
  nothing warns that T is outside the assessed range.
- The planar solver runs in the number-fixed frame with one molar volume for
  all substitutional elements unless partial molar volumes are given, in which
  case its variables are concentrations (mol/m^3); the other solvers use one
  molar volume. Temperatures are in kelvin throughout.
- A thermodynamic description with a magnetic contribution triggers a warning,
  because the mobilities omit Jonsson's ferromagnetic factor.

## How to cite

Cite the software through [CITATION.cff](CITATION.cff), pycalphad (Otis and Liu
2017), and the papers named in the docstrings of the functions you use.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Odinzen LLC.
