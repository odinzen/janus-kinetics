# calphad-kinetics

Multicomponent diffusion kinetics on top of [pycalphad](https://pycalphad.org).
Atomic mobilities come from the MQ and MF parameters of a Thermo-Calc style
database, the thermodynamic factor comes from pycalphad's Gibbs energy model,
and the two combine into the reduced interdiffusion matrix used by a 1D
implicit diffusion solver.

This is version 0.1.0.dev0, the first development phase. It is not yet released.

## What is in this version

Implemented and tested:

- **Atomic mobility** (Andersson and Agren, J. Appl. Phys. 72, 1992). Composition
  dependence by end-member values, Redlich-Kister binary terms and the ternary
  term of Jonsson (ISIJ Int. 35, 1995), read directly from MQ and MF parameters
  in a pycalphad `Database`. M = exp((MF + MQ)/RT)/RT.
- **Thermodynamic factor** from the exact symbolic Hessian of pycalphad's molar
  Gibbs energy (no finite-difference step).
- **Reduced interdiffusion matrix** D = P Lambda Phi in the number-fixed frame,
  and in the volume-fixed frame from partial molar volumes.
- **Planar 1D solver**, backward Euler finite volume with composition-dependent D
  (Picard iteration), no-flux or fixed-composition boundaries, uniform or graded
  grids.
- **Stability check.** If the Gibbs energy curvature is not positive definite at a
  node, the calculation stops with `UnstableCompositionError`. That composition is
  inside a miscibility gap or two-phase field and needs a multiphase model, which
  this version does not have.

Validation (`pytest`, 36 tests pass, 3 skipped):

- error-function solutions for a binary step couple, a fixed-surface
  semi-infinite medium, a graded grid, a diagonal ternary, and a coupled
  (off-diagonal) ternary with uphill diffusion (Crank 1975; Toor 1964);
- mass conservation under no-flux boundaries, shown to be round-off only;
- Boltzmann similarity of a strongly composition-dependent binary profile;
- ideal solution gives the identity-like thermodynamic factor and the Darken form
  of D;
- curvature against closed-form regular solutions and against finite differences
  of `pycalphad.calculate`;
- volume- and number-fixed fluxes against a direct lattice-frame flux calculation;
- mobilities against the equations and Appendix parameters of Jonsson,
  ISIJ Int. 35 (1995) 1415, and against the assessed lines of its Figs. 1 and 2.

Not yet in this version:

- the ferromagnetic mobility factor of Jonsson (1994). `mobility()` accepts a
  `magnetic_factor` multiplier, but its formula is not implemented until the
  source is checked;
- benchmarks against measured interdiffusion matrices (Cu-Ni-Zn) and against
  Jonsson's chemical diffusivities, which need data not yet in hand;
- phases with interstitial elements or vacancies on the substitutional sublattice;
- the volume-fixed frame in the solver (it runs in the number-fixed frame with a
  constant molar volume);
- moving boundaries, precipitation and coarsening. See [ROADMAP.md](ROADMAP.md).

## Install

```
conda create -n calphad-kinetics -c conda-forge python=3.12 pycalphad numpy scipy pytest
conda activate calphad-kinetics
pip install -e . --no-deps
pytest
```

On Windows run inside the activated environment (or through `conda run`), since
numpy's MKL libraries are only found on the activated path.

## Minimal example

```python
import numpy as np
from pycalphad import Database
from calphad_kinetics import SolutionPhase, solve_planar, uniform_faces

db = Database("alloy.tdb")  # G parameters plus MQ/MF mobility parameters
phase = SolutionPhase(db, "FCC_A1", ["CU", "NI", "ZN"], dependent="CU")
print(phase.elements)  # ('NI', 'ZN', 'CU'), the dependent element is last

T = 1048.0
print(phase.interdiffusion_matrix([0.30, 0.10], T))  # m^2/s, number-fixed frame

faces = uniform_faces(1e-3, 200, origin=-5e-4)
z = 0.5 * (faces[1:] + faces[:-1])
x0 = np.where(z[:, None] < 0, [0.30, 0.10], [0.10, 0.25])
result = solve_planar(faces, x0, lambda x: phase.interdiffusion_matrix(x, T), t_end=3.6e5, dt=600.0)
profile = result.x[-1]  # mole fractions of NI and ZN in each cell
```

Mobility parameters can sit in a separate file, passed as
`SolutionPhase(..., mobility_db=Database("mobility.tdb"))`.

Conventions worth knowing:

- Compositions are mole fractions ordered as `phase.elements`; functions take
  either the independent fractions or all of them.
- Interaction parameters follow pycalphad's ordering, with constituents sorted
  alphabetically, so order 1 of `MQ(FCC_A1&NI,ZN,CU;1)` multiplies (x_CU - x_ZN).
- A TDB parameter evaluates to zero outside its temperature range, as in
  pycalphad. Check that the ranges cover your temperature.
- Compositions are floored at 1e-12 before evaluating D, where the ideal-mixing
  curvature diverges but D stays finite.

## How to cite

Cite the software through [CITATION.cff](CITATION.cff), and cite pycalphad
(Otis and Liu, J. Open Res. Softw. 5, 2017) and the papers named in each
function's docstring.

## License

MIT, see [LICENSE](LICENSE).
