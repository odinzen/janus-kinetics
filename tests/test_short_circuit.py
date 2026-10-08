"""Short-circuit diffusion by Hart's average, Acta Metall. 5 (1957) 597, eq. (5)."""

import numpy as np
import pytest
from pycalphad import Database

from janus_kinetics import R, ShortCircuit, SolutionPhase
from conftest import HEADER

T = 900.0
X = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.05, 0.9, 0.05]])

# Path mobilities: the lattice values of conftest with activation energies lowered by
# about 40%, the usual order of grain-boundary enhancement. Invented for testing.
PATH_TDB = HEADER + """
PHASE SOL % 2 1 1 !
CONSTITUENT SOL :A,B,C:VA: !
PARAMETER MQ(SOL&A,A:VA;0) 298.15 -90000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(SOL&A,B:VA;0) 298.15 -100000+R*T*LN(2E-4); 6000 N !
PARAMETER MQ(SOL&A,C:VA;0) 298.15 -95000+R*T*LN(5E-5); 6000 N !
PARAMETER MQ(SOL&B,A:VA;0) 298.15 -85000+R*T*LN(3E-5); 6000 N !
PARAMETER MQ(SOL&B,B:VA;0) 298.15 -108000+R*T*LN(1E-4); 6000 N !
PARAMETER MQ(SOL&B,C:VA;0) 298.15 -93000+R*T*LN(8E-5); 6000 N !
PARAMETER MQ(SOL&C,A:VA;0) 298.15 -87000+R*T*LN(6E-5); 6000 N !
PARAMETER MQ(SOL&C,B:VA;0) 298.15 -99000+R*T*LN(1.5E-4); 6000 N !
PARAMETER MQ(SOL&C,C:VA;0) 298.15 -91000+R*T*LN(9E-5); 6000 N !
"""


@pytest.fixture(scope="module")
def phases(regular_db):
    path_db = Database(PATH_TDB)
    lattice = SolutionPhase(regular_db, "SOL", ["A", "B", "C"])
    path_only = SolutionPhase(regular_db, "SOL", ["A", "B", "C"], mobility_db=path_db)
    with_f = lambda f: SolutionPhase(regular_db, "SOL", ["A", "B", "C"], short_circuit=ShortCircuit(f, path_db))
    return lattice, path_only, with_f


def test_limits_recover_lattice_and_path(phases):
    lattice, path_only, with_f = phases
    np.testing.assert_allclose(with_f(0.0).mobility(X, T), lattice.mobility(X, T), rtol=1e-14)
    np.testing.assert_allclose(with_f(1.0).mobility(X, T), path_only.mobility(X, T), rtol=1e-14)


def test_tracer_diffusivity_follows_hart_eq5(phases):
    lattice, path_only, with_f = phases
    f = 2e-3
    d_eff = R * T * with_f(f).mobility(X, T)
    expected = f * R * T * path_only.mobility(X, T) + (1 - f) * R * T * lattice.mobility(X, T)
    np.testing.assert_allclose(d_eff, expected, rtol=1e-13)
    assert np.all(d_eff > 1.5 * R * T * lattice.mobility(X, T))  # the paths matter at this T


def test_every_matrix_is_the_same_average(phases):
    """Thermodynamics is shared, so intrinsic and interdiffusion matrices are linear in f."""
    lattice, path_only, with_f = phases
    f = 0.3
    for name in ("intrinsic_matrix", "interdiffusion_matrix"):
        mixed = getattr(with_f(f), name)(X, T)
        expected = (1 - f) * getattr(lattice, name)(X, T) + f * getattr(path_only, name)(X, T)
        np.testing.assert_allclose(mixed, expected, rtol=1e-12, atol=1e-30)


@pytest.mark.parametrize("f", [-0.01, 1.5])
def test_fraction_outside_unit_interval_is_refused(f):
    with pytest.raises(ValueError, match="fraction"):
        ShortCircuit(f, Database(PATH_TDB))
