# Changelog

## 0.2.0 (2026-10-15)

- Optional bridge to kawin, `janus_kinetics.kawin_bridge` (`pip install janus-kinetics[kawin]`): merges a
  separate mobility file into the thermodynamic database, builds kawin's thermodynamics and precipitation
  model from janus's inputs, and returns kawin's results in janus's composition order. Checked against
  janus on gamma' ageing in Ni-5.2Al-14.2Cr.
- A notebook, notebooks/CastToStrength: casting with back-diffusion, homogenization, then gamma' ageing and
  relative strength across a dendrite arm in kawin, built by tools/build_cast_to_strength.py.
- README credits kawin as the seed of this work.
- A Jupyter notebook, notebooks/KineticsWithJanus: solidification beyond Scheil in Ni-18Al-2Cr with pycalphad
  and janus-kinetics.

## 0.1.0 (2026-10-14)

First version.

- Atomic mobilities from MQ/MF database parameters, the thermodynamic factor
  from pycalphad's exact Gibbs energy Hessian, and interdiffusion matrices in the
  number- and volume-fixed frames.
- Implicit 1D solver for single-phase multicomponent diffusion, with Kirkendall
  marker tracking.
- Short-circuit diffusion by Hart's average.
- Moving interfaces in stacks of any number of phase layers, with local
  equilibrium and the operating tie-line at each interface.
- Non-isothermal anneals along piecewise-linear temperature schedules.
- Stoichiometric reaction layers in binary couples (Wagner's integrated
  diffusivity, van Loo 1990), checked against van Loo's Ni-Sn and Fe-O analyses.
- Volume-fixed planar diffusion in concentrations with constant but unequal
  partial molar volumes, with marker tracking, and analysis of measured binary
  couples for D~ and the intrinsic diffusivities (van Loo 1990, eqs. 3, 9, 10, 12).
- Conversion of plates, cylinders and spheres through a reaction layer (van Loo
  1990, eqs. 25-27), including anneal schedules and the inverse for k_p.
- Homogenization model for multiphase diffusion (binaries and ternaries).
- Cylindrical and spherical geometry for the diffusion, moving-interface, homogenization and
  reaction-layer solvers, the last with several layers growing around a wire or particle.
- Microsegregation: Scheil-Gulliver from pycalphad, closed-form limits, Kobayashi's exact
  solution, a back-diffusion solidification model, and segregation profiles for homogenization.
- Multicomponent precipitation with CALPHAD driving forces and nucleus compositions
  (Svoboda et al. 2004), checked against the LSW limit and against gamma' precipitation
  in Ni-5.2Al-14.2Cr at 873 K (Sudbrack et al. 2006, 2007) and Ni-7.5Al-8.5Cr
  (Booth-Morrison et al. 2008), with precipitate compositions that evolve after nucleation,
  misfit energy, grain-junction nucleation (Clemm and Fisher 1955), temperature schedules,
  several phases or nucleation sites competing for one matrix, and nucleation site counts
  from the grain size (Cahn 1956) or the dislocation density.
- Whipple's grain-boundary solution and Le Claire's y^6/5 analysis of sectioning data (Le Claire 1963),
  Suzuoka's (1961) thin-source case, and Harrison's (1961) A/B/C regimes.
- Coherent interfacial energy from the enthalpy of solution by the generalized broken-bond model
  (Sonderegger and Kozeschnik 2009), with the enthalpy of solution from pycalphad.
- Closed forms for late-stage multicomponent coarsening (Philippe and Voorhees 2013), checked
  against the precipitation solver in a ternary.
- Precipitation along heat treatments in dilute alloys (Myhr and Grong 2000), with their AA 6082
  inputs, checked against all five of their simulation figures.
- Worked example: ageing and reheating of AA 6082.
- Particle growth rate with local equilibrium in a multicomponent matrix (Chen, Jeppsson and Agren
  2008), checked against Zener's sphere solution and the sharp-interface solver, and as a growth law for
  the multicomponent precipitation solver (`growth="local_equilibrium"`).
- Worked example: electromigration stress in a passivated aluminium line.
- Worked example: thermal oxidation of silicon with Deal and Grove's constants, including a two-step run.
- Linear-parabolic thermal oxidation (Deal and Grove 1965).
- Two compound layers with interfacial reaction barriers in thin-film couples (Gosele and Tu 1982).
- Electromigration stress in confined metal lines (Korhonen et al. 1993): blocked, void and
  reservoir ends, stress-dependent diffusivity, ideal plasticity, and Blech's threshold.
- Phase layers that nucleate during a layered run (`solve_layers_with_nucleation`), inserted where the
  d'Heurle interface barrier falls to a threshold, conserving every element.
- Nucleation of a new phase at an interface (d'Heurle 1988): specific thickness and the critical lens for any
  three interface energies. Blech's 1977 erratum, Clyne and Kurz (1981) and Purdy and Kirkaldy (1971) checked
  against the package.
- Solidification with back-diffusion in the solid and CALPHAD tie-lines for any number of elements
  (`solve_solidification`), checked against the Scheil-Gulliver and lever-rule limits;
  `SolutionPhase(mobilities=False)` for a phase used for its thermodynamics alone.
- Scheil-Gulliver and the eutectic hand-off tell ordered from disordered composition sets of one model
  (`disordered={"L12_FCC": "FCC_A1"}`), so gamma' is found as the second solid of a Ni-Al-Cr casting.
- The Perez et al. Table 1 example draws the mean radius approach beside the size classes, as their Fig. 7 does.
- Worked example: the dendrite tip of Ni-5.2Al-14.2Cr from Turnbull's interfacial energy and Meyer et al.'s
  liquid diffusivity of nickel.
- docs/validation.md, a page of what each part is checked against, the papers not reproduced and the
  misprints found; slow tests marked so `pytest -m "not slow"` runs a quick check.
- Precipitate evolution rates under general composition constraints and a finite interface mobility
  (Kozeschnik, Svoboda and Fischer 2004, `constrained_evolution_rates`).
- The relaxed Hessian of an ordered phase (`PhaseMinimizer.hessian`), and the precipitate composition drift of
  Philippe and Voorhees checked against the atom-probe trajectories of Mao et al. (2012).
- Layers nucleating during a run around an axis or a centre, not just in plates.
- The mean radius approach of Perez et al. (2008), section 3.1 (`solve_mean_radius`), checked against the class
  method and the LSW rate.
- Two candidate layers reaching the nucleation threshold at one check: the second was read at its old interface
  index and came a check late; both now nucleate in that check.
- Back-diffusion solidification that goes on into a second solid (`solve_solidification_to_end`), handing the
  last liquid to Scheil-Gulliver at its onset as Kattner et al. (1996) do; `stop` hooks on `solve_layers` and
  `solve_solidification`.
- Dendrite tip operating point for any number of solutes (Kattner, Boettinger and Coriell 1996), with
  partition ratios and liquidus slopes from the solid-liquid equilibrium; Whipple's (1954) contour angles,
  penetration ratios and asymptote checked against his paper.
- Grids graded toward moving interfaces in the layered-stack and two-phase solvers (`graded_faces`,
  `Layer(faces=...)`, `solve_moving_boundary(grading=...)`).
- Adaptive time steps in the planar, homogenization and layered-stack solvers, with recovery
  from steps that fail to converge.
- Parabolic growth of a compact oxide scale from cation and anion transport with
  grain-boundary averaging, checked against the Cr2O3 thicknesses and figures of
  Hallstrom et al. (2013), with the Cr-O description of Kjellqvist et al. (2008)
  shipped as `kjellqvist2008_cro()`.
- Precipitation by classical nucleation, growth and coarsening on a size-class
  distribution.
- The fcc Ni-superalloy mobility database of Campbell, Boettinger and Kattner
  (2002), transcribed by a generator and checked against their Tables 3-4,
  printed values, Figs. 4-5 and the Ni-Al-Cr matrices of Table 7.
- Homogenization with local equilibrium computed on demand (Larsson and Hoglund
  2009, section 2.5), so multiphase diffusion runs with four or more elements,
  and their optional ideal flux term against oscillations (section 2.4, eq. 13).
- Cation diffusion in non-stoichiometric oxides (Hallstrom, Hoglund and Agren
  2011), with their magnetite and hematite mobilities.
- Ordered B2 phases (Helander and Agren 1999) with the Al-Ni parameters, checked
  against their Figs. 2 and 7.
- Mobility and thermodynamic databases may declare vacancy-only sublattices
  differently; each is read in its own layout.
- Worked examples (including a thin-film stack through a rapid thermal anneal),
  a theory note, a generated API reference and a
  Crossref-checked reference list.
- Input checks from an adversarial review: compositions outside [0, 1],
  misspelled boundaries, unit slips in precipitation inputs and compositions
  beyond a homogenization table now raise instead of running on silently.
