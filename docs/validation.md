# Validation

What each part of janus-kinetics is checked against, and how closely. Every row is an assertion in the
test suite (the file is named under each heading); the agreement given is what the test requires, or
what it found where that is tighter and stated in its docstring. The equations themselves are in
[theory.md](theory.md), the references in [references.md](references.md).

Three kinds of check are used. Against a paper: a table, a worked number or a figure (figures digitized
or read by eye, with the reading error stated). Against an exact solution: a closed form, a series, or an
independent calculation built in the test. Against a planted fault: for each feature a handful of
deliberate errors were introduced during development and every one made a test fail; that is how the
tests were judged able to fail, and is not repeated in the suite.

Where a paper cannot be reproduced the test is skipped with the reason and the DOI; those ten, and every
misprint found along the way, are listed at the end.

## Mobilities and diffusivities

Tests: test_mobility.py, test_jonsson1995.py, test_campbell2002.py, test_b2_ordering.py,
test_interstitial.py, test_thermo.py, test_diffusivity.py.

| Check | Source | Agreement |
|---|---|---|
| Mobilities from the shipped bcc Cr-Fe-Ni file against the paper's eqs. 3-6 evaluated independently | Jonsson 1995, Appendix | 2e-4 relative |
| Cr and Fe tracer lines in pure Cr | Jonsson 1995, Figs. 1-2 | 0.2 decades (read by eye) |
| Fe tracer in Cr-rich Cr-Fe, 960-1250 C | Jonsson 1995, Fig. 5 | 0.1 decades |
| Self-diffusion activation energies | Campbell et al. 2002, Table 4 | 5 J/mol |
| Mo in Ni at 1382 K | Campbell et al. 2002, p. 781 | 1.5 % (the paper's composition is not stated) |
| Cr in Co activation energy, 247 kJ/mol | Campbell et al. 2002, p. 781 | 0.2 % |
| Hf in Ni; Ni and W tracers in three Ni-W alloys | Campbell et al. 2002, Figs. 4-5 | 0.15 and 0.05 decades |
| Ni-Al-Cr interdiffusion matrices at 1100 C | Campbell et al. 2002, Table 7 | 8 % of their calculation (6 % found); within a factor 1.7 of the measured, as theirs is within 1.52 |
| Ni tracer and chemical diffusivity in B2 Al-Ni, x_Al <= 0.5 | Helander and Agren 1999, Figs. 7 and 2 | 0.08 and 0.06 decades (readings carry 0.03) |
| Ordering energy reduces to Girifalco's S^2 form and to eq. 9 | Helander and Agren 1999, App. B, eq. 14 | round-off |
| Interstitial diffusivity of an ideal solution; fluxes from eqs. 62-63; carbon potential in austenite | Andersson and Agren 1992; pycalphad | 1e-10; 1e-6; 0.5 J/mol |
| Mobility ordering convention equals pycalphad's excess-energy convention | pycalphad | round-off |

## Diffusion solvers and couples

Tests: test_solver.py, test_geometry.py, test_kirkendall.py, test_volume_frame.py, test_boundaries.py,
test_short_circuit.py, test_schedule.py.

| Check | Source | Agreement |
|---|---|---|
| Every element conserved under no-flux ends | exact | round-off |
| Sphere and cylinder with a held surface | series solutions | 0.03 % of the change |
| Kirkendall marker shift parabolic; velocity equals eq. 9 at the marker | van Loo 1990, eqs. 5, 9 | 0.2 % and 0.5 % |
| Interdiffusion and both intrinsic diffusivities read back from a simulated couple with V_B/V_A = 1.8 | van Loo 1990, eqs. 3, 12 | 0.5 % and 1 % |
| Sinusoid decay, ternary two-mode decay, and a fast solute slaved to a slow one | Purdy and Kirkaldy 1971, eqs. 6, 11-16, 21 | eq. 21 within D_22/D_11, as derived |
| Adaptive steps over decades of time | time-converged run | 1.5e-3 worst error in 107 steps |

## Moving interfaces, layers and reactions

Tests: test_moving_boundary.py, test_layers.py, test_geometry.py, test_reaction_layers.py,
test_particle_reaction.py, test_layer_competition.py, test_oxidation.py.

| Check | Source | Agreement |
|---|---|---|
| Cylinder and sphere growth from zero size | Zener 1949, eq. 12 | 0.24 % and 0.35 % on uniform grids; 0.11 % and 0.17 % on graded grids with a quarter of the cells |
| Ternary local-equilibrium couple on 40 graded cells a side | 400-cell uniform run | 0.4 % in position, 1e-4 in tie-line |
| Ni3Sn2 analysis: consumption ratio, marker split, marker shift; wustite couples | van Loo 1990, pp. 64-65, 78 | as printed |
| Several line-compound layers between pure end members | Wagner 1969, eq. 20 | round-off |
| Plate, cylinder and sphere conversion through one layer | van Loo 1990, eqs. 25-27 | 1e-9 |
| Two layers in a sphere against the layered solver with a narrow composition range | independent model | 1.9 nm in 400 nm |
| Competing compound layers: window ratios, critical thickness, diffusion-controlled ratio | Gosele and Tu 1982, eqs. 19-34 | 0.6 % in the thickness ratio |
| Linear-parabolic oxidation: B/A columns, activation energies, Fig. 4 lines | Deal and Grove 1965, Tables I-II, Figs. 4, 9, 11 | 4 % in B, 0.005 um in A |

## Grain boundaries

Tests: test_grain_boundary.py.

| Check | Source | Agreement |
|---|---|---|
| Fisher's equations solved numerically against the exact Laplace-transform solution | Fisher 1951 | 0.5 % |
| Whipple's contour angles | Whipple 1954, Figs. 1-2 | 0.06 degrees at beta = 0.1; 0.15-1.1 degrees at beta = 1 |
| Penetration ratios 1.39 and 2.88 over the lattice depth 0.742 | Whipple 1954, Fig. 4 | 0.05 |
| Steepest-descents asymptote and its coefficients | Whipple 1954, eq. 5.3 | within 6 % by eta = 32 |
| D' delta read back from an exact profile at beta = 100 by the y^6/5 method | Le Claire 1963, eq. 15 | 3.4 % (Fisher's reading gives a third, as Le Claire says) |
| Thin-source 6/5 slope and coefficient | Suzuoka 1961; Le Claire 1963, eq. 16 | 2 % in D' delta |

## Oxides and scales

Tests: test_oxides.py, test_chromia.py.

| Check | Source | Agreement |
|---|---|---|
| Magnetite defect fractions, 1e-16 to 1e-2 atm at 1273 K | Hallstrom et al. 2011, Fig. 3 | about 5 % |
| Pressure of lowest Fe tracer diffusivity, 1273-1623 K | Hallstrom et al. 2011, Fig. 4 | 0.25 decades |
| Cr2O3 scale thickness after 24 h at 625 and 700 C | Hallstrom et al. 2013, Figs. 8-9 | 1.5 % |
| Late-time scale growth rate | Hallstrom et al. 2013 | 5 % and 8 % |
| Oxygen tracer lines | Hallstrom et al. 2013, Fig. 7 | 0.07 decades |

## Thin films and electromigration

Tests: test_electromigration.py, test_interface_nucleation.py, test_layer_nucleation.py.

| Check | Source | Agreement |
|---|---|---|
| Stress in a semi-infinite line, a line blocked at both ends, a void end | Korhonen et al. 1993, eqs. 9, 12, 13, 15 | first-order convergence; steady state to 1e-9 |
| Yield front speed and the stress ahead of it | Korhonen et al. 1993, eq. 11 | 0.3 % and 1 % |
| Measured stress fit, digitized | Korhonen et al. 1993, Fig. 9 | 0.006 in log10 stress |
| Threshold current density-length product; 1.1e5 A/cm^2 in 115 um | Blech 1976 with the 1977 erratum | as printed |
| Specific thicknesses; silicide molar volumes; NiSi2 driving force | d'Heurle 1988, Tables II-III | as printed, two worked numbers excepted (below) |
| Equal-energy lens; hemisphere condition; symmetric lens against the grain-boundary lens | d'Heurle 1988, eqs. 14-19; Clemm and Fisher 1955 | round-off |
| A layer nucleating in a heated couple, in a plate, cylinder and sphere | independent barrier calculation | first check after the threshold (900 K against 897.5 K); content kept to 1e-12 |

## Precipitation

Tests: test_precipitation.py, test_multicomponent_precipitation.py, test_coarsening.py,
test_particle_growth.py, test_heat_treatment.py, test_interfacial_energy.py.

| Check | Source | Agreement |
|---|---|---|
| Gibbs-Thomson composition, nucleation rate by hand, Zener growth | Perez et al. 2008, eqs. 1-10 | round-off; Zener within 2e-3 |
| Coarsening rate approaching LSW | Perez et al. 2008, eq. 11 | within 20 % from above, falling |
| Mean radius approach against the size classes, 1e3 to 1e5 s | Perez et al. 2008, section 3.5.1 | R 4 %, N 15 %, X 1.5 % |
| Mean radius approach coarsening at the LSW rate | Perez et al. 2008, eq. 11 | 2 % |
| gamma/gamma' equilibrium and critical nucleus in Ni-5.2Al-14.2Cr | Sudbrack et al. 2007, Table 3 (Thermo-Calc, same database) | 0.1 at.% |
| State at 4 h: volume fraction, number density, radius | Sudbrack et al. 2006, Table 1 | 5.3 against 5.2 %; 3.2e24 against 3.2e24; 1.54 against 1.27 nm |
| Mean radius at 1 h in Ni-7.5Al-8.5Cr | Booth-Morrison et al. 2008, Table 1 | 1.24 nm within 10 % |
| Grain-junction barriers for the bulk, boundary and edge | Clemm and Fisher 1955, Fig. 6 | 0.5 % |
| Boundary area, edge length and corner count of the grain model | Cahn 1956, eq. 1 | printed digits; exact solid by Voronoi cell |
| Misfit energy with and without yielding | Svoboda et al. 2004, Appendix A | the printed values of f(S) |
| Seeds of two phases settling on the three-phase equilibrium | Gibbs-Thomson condition, lever rule | 1e-3; 22.13 against 22.26 % |
| Evolving compositions satisfy the extremum principle | Svoboda et al. 2004, eqs. 27-28 | numerical derivative |
| Constrained evolution equals part I; keeps a sublattice stoichiometry; dG/dt + Q = 0 | Kozeschnik et al. 2004, eqs. 17-32 | round-off |
| Multicomponent coarsening rate, number density and supersaturation in a ternary | Philippe and Voorhees 2013, eqs. 42-44 | 2 % |
| Precipitate composition drift direction against measured trajectories | Philippe and Voorhees 2013, eq. 45; Mao et al. 2012, Table 4 | Al sign agrees; Cr sign does not (below) |
| Growth with local equilibrium against the sharp-interface solver | Chen et al. 2008, eqs. 2, 10 | 0.1 % in d(R^2)/dt, 2e-6 in tie-line |
| Local-equilibrium growth as the class law, coarsening in a ternary | Philippe and Voorhees 2013, eq. 42 | 3 % |
| Broken-bond factors, 132 entries | Sonderegger and Kozeschnik 2009, Table II | 0.1 %, one misprint excepted |
| Interfacial energy of gamma' in Ni-Al | Sonderegger and Kozeschnik 2009, Table VI | inside the measured 14-80 mJ/m^2 |
| Heat treatment of AA 6082: ageing, up-quench, reversion, heating ramps | Myhr and Grong 2000, Figs. 3, 5, 7, 8, 9 | 3 % in radius and 5 % in number (Fig. 3); 1.5 % in fraction (Fig. 5); 0.0003 in fraction (Fig. 9) |

## Solidification and homogenization

Tests: test_solidification.py, test_dendrite_tip.py, test_homogenization.py.

| Check | Source | Agreement |
|---|---|---|
| Closed forms and the exact Brody-Flemings series | Kobayashi 1988 | series converged to 4.513 at f_s = 1 |
| Numerical back-diffusion against the exact solution | Kobayashi 1988, eqs. 37-40 | 0.2 % to f_s = 0.95 |
| Clyne-Kurz parameter and Table 1 | Clyne and Kurz 1981, eq. 21, Table 1 | to the printed digits |
| Scheil-Gulliver path on a constant-k binary | Scheil equation; Bocklund et al. 2020 | 1.5 % at 4 K steps, falling with the step |
| Back-diffusion with CALPHAD tie-lines: frozen solid and fast solid | Scheil-Gulliver; lever rule | 1e-3 and 3e-3 in fraction solid |
| Into a eutectic: frozen solid; fast solid | closed-form Scheil eutectic; lever rule | 1 %; no second solid forms |
| Solute kept through back-diffusion and the hand-off to Scheil | exact | 1e-9 (1e-8 around a cylinder) |
| gamma' told from gamma under one ordered model; hand-off at gamma' in Ni-18Al-2Cr | pycalphad equilibrium on Dupin's description; Scheil-Gulliver | 0.05 at.% Al; within 1 K and 2 % of the Scheil gamma' |
| Dendrite tip: Ivantsov function, eqs. 2-5 rebuilt by hand, low-speed limit | Kattner et al. 1996 | 1e-10, 1e-9, exact relation |
| Tip liquid equals the lever-rule liquid at the tip temperature when diffusivities are equal | Kattner et al. 1996, p. 527 | 1e-6 against pycalphad |
| Homogenization model against the single-phase solver, in a plate and a sphere | phase-1 solver | 1.7e-5 |
| Homogenization in a quaternary as the grid is refined | phase-1 solver | 2.1e-3, 4.8e-4, 1.4e-4 |
| Slowest decay mode of an anneal in a plate, cylinder and sphere | exact eigenvalues | 1 % |

## Not reproduced, and why

### Paper inconsistent with itself (skipped tests)

- Perez et al. 2008, Fig. 7: its plateau (R = 4.8 nm, X = 0.0142) needs 2 gamma v/kT = 1.7 nm by its own
  eq. 9, where Table 1 gives 0.70 nm.
- Clemm and Fisher 1955, Fig. 6: the corner barrier printed, 0.0096e-8 erg, is 13 % below what the paper's own
  coefficients give (0.0109e-8); those coefficients pass every internal check.
- Hallstrom et al. 2013, Fig. 6: the Cr tracer lines lie 0.2-0.75 decades below the paper's eq. 11 with the
  same thermodynamics, and since the defect fraction cancels in D*_Cr / D*_O its Figs. 6 and 7 cannot both
  follow eqs. 11-12.
- Hallstrom et al. 2013, Fig. 11: oxygen carries about 10 % of the rate constant there and 18 % here, a share
  fixed by eqs. 11-12 alone.

### Model against measurement (skipped tests)

- Sudbrack et al. 2006: classical nucleation at a 5 kT barrier gives 2.6e25 m^-3 s^-1 against the measured
  5.9e21, a comparison the authors warn against.
- Sudbrack et al. 2006: the measured equilibrium gamma' fraction is 15.6 %, Dupin's description gives 11.8 %,
  so the late volume fraction cannot reach the measurement.
- Hallstrom et al. 2011, Figs. 7-8: hematite comes out 0.4-0.7 decades below the paper's curves with the
  thermodynamics the paper used; not explained.

### Data or a paper not available (skipped tests)

- Sudbrack et al. 2007, Table 4: pycalphad's curvatures of Dupin's gamma are 10-34 % above the Thermo-Calc
  values printed for the same description, while the equilibria agree within 0.08 at.%; the cause is not found.
- Jonsson 1995, Fig. 4: needs the 1994 SGTE solution database and the ferromagnetic factor.
- Jonsson 1995, ferromagnetic factor: needs Jonsson 1992, which defines it; the restatement in Helander and
  Agren 1999, eq. 8, is not dimensionally consistent as printed.

### Gaps recorded in passing tests

- Coarsening of gamma' in Ni-Al-Cr runs 4.4-6.5 times faster than measured, close to the papers' own
  estimates with these mobilities.
- Mao et al. 2012: the measured gamma' Cr is above its long-time value and falling in two alloys; the mean-field
  drift has it below. The drift direction holds no mobility, so flux coupling cannot reverse it in the
  late stage.
- B2 Al-Ni above x_Al = 0.5: chemical diffusivity 0.3-0.5 decades above Helander and Agren's Fig. 2, with
  either thermodynamic description; the module warns there.
- Fe-rich bcc Fe-Cr: 0.12 decades above Jonsson's Fig. 5 at 1250 C and growing toward the Curie temperature,
  the size of the omitted ferromagnetic factor.
- Myhr and Grong 2000, Fig. 9: regrowth after the sigma = 0.8 dip runs about 8 C behind the figure.
- Hallstrom et al. 2011: the magnetite level sits 0.1-0.25 decades below Fig. 4; the paper's own eq. 7 with
  its eqs. 10-11 and Fig. 3 fractions lies 0.21-0.30 decades below the same figure.

### Commercial or unpublished data

Kattner et al. 1996 (IN718 and Rene N4, a preliminary database), Lippard et al. 1998 (AerMet100), Chen et al.
2008 (Fe-Cr-C with TCFE4 and MOB2), Larsson and Hoglund 2009 (multiphase examples) and the 36.2 mJ/m^2 of
Sonderegger and Kozeschnik 2009 (Ni-Data) are not reproduced.

## Misprints and slips found in the papers

Each is recorded where it is used, with the reading taken.

- Blech 1976: eq. 5 lacks the resistivity and eq. 6 is inverted on the left, as his 1977 erratum corrects.
- Korhonen et al. 1993: eq. 15 prints its leading term as -G xi where -G L xi is needed; the diffusivity
  inferred from their Fig. 9 fit is 50 times what their own kappa gives, as if hours were read as minutes.
- Gosele and Tu 1982: eq. 33 carries a stray T before D_gamma that eq. 34, derived from it, does not.
- Deal and Grove 1965: Table II's tau differs from eq. 12c with its own A and B by up to 27 % (estimated
  graphically, the text says); eq. 12c is used.
- d'Heurle 1988: d_s for Ni2Si is 0.137 nm from his inputs (printed 0.15); the NiSi driving force is 300
  cal/cm^3 (printed 285).
- Whipple 1954: eqs. 5.1-5.2 print the lattice term as erfc eta where eq. 3.12 and Fig. 4 have erfc(eta/2).
- Kobayashi 1988: 4.48 at f_s = 1 is a series stopped near 1000 terms; it converges to 4.513.
- Svoboda et al. 2004: eq. 32 prints 5/(3 c_01 D_01) where eqs. 25, 33 and the dissipation give
  5/(c_01 D_01).
- Sonderegger and Kozeschnik 2009: Table II's [4,4,3] bcc entry at r^-8 is 0.2969, printed 0.3297; the bcc
  maximum below index 10 is at [6,1,1], 0.1 % above the [10,2,1] named; z_L,eff on p. 503 differs slightly
  from the converged sums.
- Myhr and Grong 2000: the distributions drawn in Fig. 6 sit about 10 % low in radius against the text.
- Kattner et al. 1996: eq. 4's solute gradient has the wrong sign (eq. 3 then has no real root); eq. 5's
  curvature term is a quarter of 2 Gamma/R.
- Mao et al. 2012: Table 4 prints gamma' Al of alloy C at 16 h as 11.0 at.%; the row sums to 100 only with 18.0.
- Jonsson 1995: Fig. 1's bottom axis is labelled 10^4/T but scales as 10^5/T.
- Sundman 1991, Table 1: Fe+2:Fe+2:Va:O-2 = 7G - B where Kjellqvist et al. 2008 and Hallstrom et al.'s
  results need 7G + B.
- Perez et al. 2008: eq. 17 is left undefined outside 0.99-1.01 R*, and eq. 18 compares rates of opposite
  sign; read as 0 outside the band and as comparing sizes.
