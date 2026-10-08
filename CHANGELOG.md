# Changelog

## 0.1.0 (unreleased)

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
