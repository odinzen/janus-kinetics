# janus-kinetics

Diffusion, moving-boundary and precipitation kinetics on top of
[pycalphad](https://pycalphad.org), built from the published models with every
equation cited to the paper it comes from. Atomic mobilities are read from the
MQ and MF parameters of a Thermo-Calc style database, the thermodynamics come
from pycalphad's Gibbs energy models, and the two combine into multicomponent
diffusion, two-phase and precipitation simulations.

Version 0.1.0, released 2026-10-14. Tested with Python 3.12 and pycalphad 0.11.2.

## What it does

- **Atomic mobility** with composition dependence by end members, Redlich-Kister
  and ternary terms (Andersson and Agren 1992; Jonsson 1995).
- **Thermodynamic factor** from the exact symbolic Hessian of pycalphad's molar
  Gibbs energy. A composition inside a spinodal stops the single-phase models
  with `UnstableCompositionError`.
- **Interdiffusion matrices** in the number-fixed and volume-fixed frames, and
  intrinsic (lattice-frame) diffusivities.
- **Plates, cylinders and spheres** for the diffusion, moving-interface and homogenization solvers
  (concentric shells, as in DICTRA), checked against series solutions and Zener's
  growth coefficients for each geometry.
- **Planar diffusion solver**, implicit and conservative, with no-flux or
  fixed-composition boundaries and **Kirkendall marker** tracking (van Loo 1990),
  in the number-fixed frame or, with partial molar volumes, in the volume-fixed
  frame in concentrations, so the molar volume may change with composition.
  Optional adaptive time steps (also in the homogenization and layered-stack solvers) follow profiles
  saved over decades of time and retry steps that fail to converge.
- **Reading measured couples**: the interdiffusion coefficient from a binary
  penetration curve without a Matano plane, and the intrinsic diffusivities at
  the marker plane, both allowing a composition-dependent molar volume (van Loo
  1990, eqs. 3 and 12).
- **Short-circuit diffusion** by Hart's time-fraction average (Hart 1957, derived
  for dislocations; applied here also to grain boundaries and to multicomponent
  mobilities), and **diffusion down isolated grain boundaries** from a surface
  (Fisher 1951), with the penetration-curve slope used to read D'delta from
  sectioning data. Fisher's closed form overestimates deep penetration; the
  package also solves his equations exactly, and evaluates Whipple's exact solution
  with Le Claire's (1963) y^6/5 reading of sectioning data, which corrects Fisher's
  threefold underestimate of D'delta.
- **Interstitial elements** (C, N, O, H on a vacancy-bearing sublattice) in
  u-fractions (Andersson and Agren 1992, eqs. 62-65), and **surfaces held at a
  chemical potential**, as a carburizing or oxidizing atmosphere imposes, with the
  other elements passing no flux.
- **Moving interfaces in layered stacks**: any number of phase layers, each
  interface in local equilibrium or at fixed compositions, with every interface
  velocity and operating tie-line solved together. Reaction layers grow or are
  consumed; a consumed layer stops the run with the history returned. Grids can be
  graded toward the interfaces, where accuracy is decided: 300 graded cells beat
  1200 uniform ones on Zener's growing sphere.
- **Stoichiometric reaction layers** (silicides, intermetallics, oxide scales):
  a binary couple of line compounds grown from mobilities and Gibbs energies or
  from measured integrated diffusivities (Wagner's D_int, van Loo 1990), with
  Kirkendall markers, finite films that are used up in turn, and anneal schedules.
  Plates, wires and powder particles reacting through such a layer follow van
  Loo's conversion relations, which also turn measured conversion into the
  planar growth constant; several layers can grow around a wire or particle.
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
- **Competing compound layers in thin films**: why a thin-film couple grows one
  compound where the phase diagram has several (Gosele and Tu 1982): interfacial
  reaction barriers, linear-to-parabolic changeover, and the critical thickness at which
  a second compound can appear; and the nucleation barrier of a new phase at an interface
  (d'Heurle 1988), whose small driving forces make some silicides nucleation-controlled;
  a layered run can let such a phase nucleate at an interface once its barrier is low enough.
- **Thermal oxidation**: Deal and Grove's linear-parabolic law (1965), from the
  physical constants or fitted as they fit it, along temperature schedules, checked
  against their tables and figures.
- **Electromigration in interconnect lines**: stress build-up along a confined line
  (Korhonen et al. 1993) with blocked, void or pad ends, stress-dependent diffusion and
  yielding, checked against their exact solutions and their fit to measured stresses, and
  Blech's threshold current density-length product (Blech 1976).
- **Microsegregation and homogenization**: Scheil-Gulliver paths from pycalphad for
  any number of elements (Bocklund et al. 2020), the closed-form segregation limits,
  Kobayashi's exact Brody-Flemings solution, a back-diffusion solidification model
  in a plate, cylinder or sphere, solidification with back-diffusion and CALPHAD
  tie-lines for any number of elements (between the Scheil and lever-rule limits,
  which it reproduces) and on into a eutectic or other second solid, the dendrite tip's
  radius, liquid and temperature for any number of solutes with CALPHAD partition ratios
  and slopes (Kattner et al. 1996), and the anneal that removes the segregation.
- **Homogenization model** for diffusion through multiphase mixtures (Larsson and
  Hoglund 2009), with local equilibrium tabulated up front for binaries and
  ternaries or computed on demand for any number of elements (their section 2.5).
- **Precipitation**: classical nucleation, growth and coarsening on a size-class
  distribution (Perez, Dumont and Acevedo-Reyes 2008, with its 2009 corrigendum),
  for precipitates of fixed composition in a binary matrix, and the same paper's
  mean radius approach, which follows only the number and mean size.
- **Multicomponent precipitation** with driving forces and nucleus compositions from
  the full CALPHAD Gibbs energies (Svoboda, Fischer, Fratzl and Kozeschnik 2004):
  classes that keep their nucleation composition, ordered precipitates such as
  gamma', compositions that evolve after nucleation, misfit energy, nucleation on
  grain boundaries, edges and corners (Clemm and Fisher 1955) with site counts from
  the grain size (Cahn 1956) or the dislocation density, temperature schedules, and
  several phases or nucleation sites competing for one matrix; checked against the
  LSW limit, a three-phase equilibrium, and atom-probe kinetics of gamma'
  in two Ni-Al-Cr alloys (Sudbrack et al. 2006; Booth-Morrison et al. 2008). The
  evolution rates also take general composition constraints, such as a sublattice
  stoichiometry, and a finite interface mobility (Kozeschnik, Svoboda and Fischer 2004).
- **Interfacial energy estimates** for coherent precipitates from the enthalpy of
  solution (the generalized broken-bond model of Sonderegger and Kozeschnik 2009),
  for any fcc or bcc interface orientation or the orientation average, with the
  enthalpy of solution taken from a pycalphad database.
- **Late-stage coarsening in multicomponent alloys**: Philippe and Voorhees' (2013)
  rate constant, matrix supersaturation, number density and precipitate-composition
  drift, with the full mobility matrix; the precipitation solver reproduces them in a
  ternary whose solutes diffuse at different rates, and the drift is compared with
  atom-probe trajectories of gamma' (Mao et al. 2012).
- **Precipitation along heat treatments** in dilute alloys (Myhr and Grong 2000):
  nucleation, growth, dissolution and coarsening of a size distribution through
  ageing, up-quenching and heating ramps, reproducing their AA 6082 figures.
- **Growth rate of a particle with local equilibrium** in a multicomponent matrix
  (Chen, Jeppsson and Agren 2008): the operating tie-line and growth rate of a sphere
  of given radius, with each element's own effective diffusion distance and the
  Gibbs-Thomson effect, exact against the sharp-interface solver where it should be.
  The multicomponent precipitation solver can grow its size classes by this law
  (`growth="local_equilibrium"`) instead of the thermodynamic extremal principle.

Three published mobility databases ship with the package:
`jonsson1995_bcc_crfeni()` (bcc Cr-Fe-Ni, Jonsson 1995), `campbell2002_fcc_ni()`
(fcc Ni-Al-Co-Cr-Hf-Mo-Re-Ta-Ti-W for Ni-base superalloys, Campbell, Boettinger
and Kattner 2002) and `helander1999_bcc_alni()` (B2-ordered Al-Ni, Helander and
Agren 1999), each checked against numbers or figures its paper prints. A Cr-O
thermodynamic description, `kjellqvist2008_cro()` (from Kjellqvist, Selleby and
Sundman 2008), serves the Cr2O3 scale model.

The equations are collected in [docs/theory.md](docs/theory.md), what each part is
checked against and how closely in [docs/validation.md](docs/validation.md), including
where the papers could not be reproduced and the misprints found in them, the API in
[docs/api.md](docs/api.md), and the references, checked against Crossref, in
[docs/references.md](docs/references.md). What is not done yet is listed in
[ROADMAP.md](ROADMAP.md); the main gaps are the ferromagnetic mobility factor
and scales of several oxides or on alloys.

## Install

```
git clone https://github.com/odinzen/janus-kinetics
cd janus-kinetics
conda create -n janus-kinetics -c conda-forge python=3.12 pycalphad numpy scipy pytest matplotlib
conda activate janus-kinetics
pip install -e . --no-deps
pytest
```

The full test suite takes about 15 minutes on a desktop; `pytest -m "not slow"`
leaves out the runs of 5 s or more and takes about 4. Ten tests are skipped, each
with the reason a paper's figure or table cannot be reproduced
([docs/validation.md](docs/validation.md) lists them). On Windows run inside the
activated environment, since numpy's MKL libraries are only found on the activated
path.

## Example

A bcc Fe-30Cr / Fe-70Cr couple at 1200 C with the published mobilities of
Jonsson (1995), which ship with the package, and an ideal-solution Gibbs energy:

```python
import numpy as np
from pycalphad import Database
from janus_kinetics import SolutionPhase, jonsson1995_bcc_crfeni, solve_planar, uniform_faces

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
run, an alpha/beta couple solved by both two-phase models, a reaction layer
growing between two thin films during a rapid thermal anneal, with both
interfaces following the tie-lines as the temperature ramps, holds and cools,
oxygen dissolving into a metal film from an atmosphere during an anneal, a metal
film reacting into two line compounds, van Loo's Ni-Sn Kirkendall analysis, a
Cr2O3 scale growing on chromium, solidification segregation in a Ni-Al-Cr alloy
and its removal by an anneal, the dendrite tip of the same alloy from measured
interfacial energy and liquid diffusivity of nickel, electromigration stress building up in a
passivated aluminium line, Deal and Grove's thermal oxidation of silicon,
including a dry-then-wet two-step oxidation, and the ageing and reheating of an
AA 6082 aluminium alloy.

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
