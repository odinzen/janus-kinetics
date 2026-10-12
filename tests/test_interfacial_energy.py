"""Generalized broken-bond interfacial energy, against Sonderegger and Kozeschnik, Metall. Mater. Trans. A 40 (2009)
499: Table II (structural factors of 22 orientations for r^-6 to r^-8 bonds), Table III (extremes and means),
the first-neighbour limits and coordination numbers of pp. 504 and 506, and Table VI for Ni-Al."""

from importlib.resources import files

import numpy as np
import pytest
from pycalphad import Database, equilibrium, variables as v

from janus_kinetics.interfacial_energy import (
    N_A,
    broken_bond_energy,
    broken_bond_factor,
    effective_coordination,
    solution_enthalpy,
)

# Table II: orientation -> z_S,eff / z_L,eff for (fcc, bcc) at r^-6, r^-7, r^-8, 100 shells
TABLE_II = {
    (1, 0, 0): (0.3105, 0.3332, 0.2848, 0.3113, 0.2746, 0.3041),
    (1, 1, 0): (0.3271, 0.2905, 0.3019, 0.2617, 0.2918, 0.2488),
    (1, 1, 1): (0.2871, 0.3351, 0.2581, 0.3096, 0.2454, 0.2986),
    (2, 1, 0): (0.3394, 0.3283, 0.3152, 0.3029, 0.3056, 0.2925),
    (2, 1, 1): (0.3222, 0.3224, 0.2953, 0.2956, 0.2839, 0.2837),
    (2, 2, 1): (0.3182, 0.3281, 0.2907, 0.3012, 0.2790, 0.2893),
    (3, 1, 0): (0.3372, 0.3355, 0.3126, 0.3116, 0.3028, 0.3025),
    (3, 2, 0): (0.3384, 0.3174, 0.3138, 0.2904, 0.3039, 0.2790),
    (3, 1, 1): (0.3275, 0.3361, 0.3019, 0.3111, 0.2913, 0.3007),
    (3, 2, 1): (0.3322, 0.3198, 0.3063, 0.2921, 0.2955, 0.2796),
    (3, 2, 2): (0.3118, 0.3319, 0.2838, 0.3056, 0.2717, 0.2939),
    (3, 3, 1): (0.3244, 0.3198, 0.2977, 0.2921, 0.2866, 0.2798),
    (3, 3, 2): (0.3094, 0.3328, 0.2812, 0.3066, 0.2691, 0.2951),
    (4, 1, 0): (0.3337, 0.3379, 0.3088, 0.3144, 0.2988, 0.3057),
    (4, 3, 0): (0.3365, 0.3110, 0.3116, 0.2835, 0.3017, 0.2716),
    (4, 1, 1): (0.3288, 0.3398, 0.3030, 0.3158, 0.2924, 0.3064),
    (4, 2, 1): (0.3360, 0.3314, 0.3107, 0.3053, 0.3004, 0.2941),
    (4, 3, 1): (0.3340, 0.3150, 0.3083, 0.2869, 0.2977, 0.2742),
    (4, 3, 2): (0.3228, 0.3290, 0.2957, 0.3021, 0.2842, 0.2901),
    (4, 3, 3): (0.3057, 0.3340, 0.2773, 0.3079, 0.2651, 0.2965),
    (4, 4, 1): (0.3266, 0.3141, 0.3002, 0.2861, 0.2893, 0.2736),
    (4, 4, 3): (0.3043, 0.3344, 0.2758, 0.3083, 0.2635, 0.3297),
}
COLUMNS = [(m, s) for m in (6, 7, 8) for s in ("fcc", "bcc")]


def test_structural_factors_reproduce_table_ii():
    """All 132 entries to 0.1 % but one: [4,4,3] bcc at r^-8 is printed 0.3297, while we get 0.2969. Every other
    row falls from r^-7 to r^-8, and the printed value would rise above its own r^-7 entry (0.3083), so it is
    taken as a misprint."""
    for normal, row in TABLE_II.items():
        for (m, s), ref in zip(COLUMNS, row):
            ours = broken_bond_factor(s, normal, m)
            if normal == (4, 4, 3) and (m, s) == (8, "bcc"):
                assert ours == pytest.approx(0.2969, abs=5e-5)
                assert ours < broken_bond_factor(s, normal, 7) and ref > row[3]
                continue
            assert ours == pytest.approx(ref, rel=1e-3), (normal, m, s)
            if m > 6:
                assert ours < broken_bond_factor(s, normal, m - 1)


def test_extremes_and_anisotropy_of_table_iii():
    """Searching every orientation with indices up to 10, as the paper does: fcc runs from [111] (0.2871) to
    [10,5,1] (0.3398), 18.4 %. In bcc the minimum is [110] (0.2905) and the anisotropy 17.6 %, but the largest
    value we find is at [6,1,1], 0.3418, 0.1 % above the [10,2,1] the paper names, whose 0.3415 we reproduce."""
    for s, lo, hi, (vlo, vhi), aniso in (("fcc", (1, 1, 1), (10, 5, 1), (0.2871, 0.3398), 0.184),
                                         ("bcc", (1, 1, 0), (6, 1, 1), (0.2905, 0.3415), 0.176)):
        vals = {(h, k, l): broken_bond_factor(s, (h, k, l)) for h in range(1, 11) for k in range(h + 1) for l in range(k + 1)}
        assert min(vals, key=vals.get) == lo and max(vals, key=vals.get) == hi
        assert vals[lo] == pytest.approx(vlo, rel=1e-3)
        assert vals[hi] / vals[lo] - 1 == pytest.approx(aniso, abs=1e-3)
    assert broken_bond_factor("fcc", (10, 5, 1)) == pytest.approx(0.3398, rel=1e-3)
    assert broken_bond_factor("bcc", (10, 2, 1)) == pytest.approx(0.3415, rel=1e-3)
    assert broken_bond_factor("bcc", (6, 1, 1)) == pytest.approx(0.3418, abs=1e-4)


def test_orientation_mean():
    """The closed-form average (|v . n| -> |v| / 2) against an average over 4000 random orientations, and against
    Table III's r^-6 means, 0.3291 (fcc) and 0.3280 (bcc), within 0.2 %. Table II's r^-7 and r^-8 means differ
    from the closed form by up to 0.7 % with alternating sign, which we put down to the paper's numerical
    integration over interpolated orientations (p. 503)."""
    normals = np.random.default_rng(1).normal(size=(4000, 3))
    for s, ref in (("fcc", 0.3291), ("bcc", 0.3280)):
        mean = broken_bond_factor(s)
        assert np.mean([broken_bond_factor(s, n) for n in normals]) == pytest.approx(mean, rel=1e-3)
        assert mean == pytest.approx(ref, rel=2e-3)
    for (m, s), ref in zip(COLUMNS[2:], (0.3020, 0.3032, 0.2909, 0.2926)):
        assert broken_bond_factor(s, None, m) == pytest.approx(ref, rel=8e-3)


def test_first_neighbour_limit():
    """As m grows only the first shell counts, and the minima become 0.2291 for fcc [111] and 0.2227 for bcc
    [110] (p. 506): Turnbull's three bonds per (111) atom over twelve, rescaled to n_S = n_V^(2/3), which is
    4^(1/3) 3^(1/2) / 12 and, for bcc [110], 2^(1/3) 2^(1/2) / 8."""
    fcc, bcc = broken_bond_factor("fcc", (1, 1, 1), 200, shells=2), broken_bond_factor("bcc", (1, 1, 0), 200, shells=2)
    assert fcc == pytest.approx(4 ** (1 / 3) * np.sqrt(3) / 12, rel=1e-9)
    assert bcc == pytest.approx(2 ** (1 / 3) * np.sqrt(2) / 8, rel=1e-9)
    assert fcc == pytest.approx(0.2291, abs=1e-4) and bcc == pytest.approx(0.2227, abs=1e-4)


def test_effective_coordination():
    """z_L,eff converges with shells (100 against 500 to 1e-5); first neighbours alone underestimate it by 20 %
    (fcc) and 53 % (bcc), p. 504. The paper quotes 14.442 and 12.234 (p. 503), 0.08 % and 0.16 % below the
    converged lattice sums we get, 14.4539 and 12.2537; it does not say how many shells those used."""
    for s, z1, under, quoted in (("fcc", 12, 0.20, 14.442), ("bcc", 8, 0.53, 12.234)):
        z = effective_coordination(s, 6, 500)
        assert effective_coordination(s, 6, 100) == pytest.approx(z, rel=1e-5)
        assert z / z1 - 1 == pytest.approx(under, abs=0.005)
        assert z == pytest.approx(quoted, rel=2e-3)
    assert effective_coordination("fcc") == pytest.approx(14.4539, abs=1e-4)
    assert effective_coordination("bcc") == pytest.approx(12.2537, abs=1e-4)


def test_energy_reduces_to_turnbull():
    """With first neighbours only, eq. (9) is the classic count, eqs. (1)-(2): three bonds cut per atom of a
    (111) plane holding 4 / (3^(1/2) a^2) atoms per area, each worth Delta E_sol / (12 N)."""
    vm, dh = 7.1e-6, 12e3
    a = (4 * vm / N_A) ** (1 / 3)
    factor = broken_bond_factor("fcc", (1, 1, 1), 200, shells=2)
    gamma = broken_bond_energy(dh, vm, factor=factor)
    assert gamma == pytest.approx(3 * 4 / (np.sqrt(3) * a * a) * dh / (12 * N_A), rel=1e-9)
    assert broken_bond_energy(dh, vm, "bcc", (1, 1, 0)) == pytest.approx(
        (N_A / vm) ** (2 / 3) * broken_bond_factor("bcc", (1, 1, 0)) * dh / N_A, rel=1e-12)


TOY = """ELEMENT VA VACUUM 0 0 0 !
ELEMENT CU FCC_A1 63.546 0 0 !
ELEMENT NI FCC_A1 58.69 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE FCC_A1 % 1 1 !
CONSTITUENT FCC_A1 :CU,NI: !
PARAMETER G(FCC_A1,CU;0) 298.15 -2000+4*T; 6000 N !
PARAMETER G(FCC_A1,NI;0) 298.15 1000-2*T; 6000 N !
PARAMETER L(FCC_A1,CU,NI;0) 298.15 15000-3*T; 6000 N !
PHASE PREC % 1 1 !
CONSTITUENT PREC :NI: !
PARAMETER G(PREC,NI;0) 298.15 -5000+3*T+0.001*T**2; 6000 N !
"""


def test_solution_enthalpy_of_a_regular_solution():
    """Pure Ni precipitate in a Cu-Ni regular solution with a temperature-dependent interaction: the partial
    enthalpy of Ni is 1000 + 15000 (1 - x)^2 and the precipitate's enthalpy -5000 - 0.001 T^2, so Appendix I's
    Delta E_sol is their difference, whatever the entropy terms."""
    db = Database(TOY)
    for x in (0.02, 0.1, 0.3):
        h = solution_enthalpy(db, ["CU", "NI"], "FCC_A1", [1 - x, x], "PREC", [0, 1], 800.0)
        assert h == pytest.approx(1000 + 15000 * (1 - x) ** 2 + 5000 + 0.001 * 800**2, rel=1e-5)


ALCRNI = files("pycalphad") / "tests" / "databases" / "alcrni.tdb"


def _alcrni():
    if not ALCRNI.is_file():
        pytest.skip("pycalphad's alcrni.tdb test database is not installed")
    return Database(str(ALCRNI))


def test_gamma_prime_in_nickel_aluminium():
    """Table VI, first row: gamma' in binary Ni-Al between 500 and 715 C, 14 to 80 mJ/m^2 from coarsening
    experiments, 36.2 from the model with the commercial Ni-Data database. With Dupin et al.'s description
    (alcrni.tdb) at the equilibrium compositions and the r^-6 mean factor we get 40-45 mJ/m^2, inside the measured
    range; the 20 % gap to 36.2 is a different database's enthalpy of solution. Naming the ordered phase for the
    matrix gives the same answer: held homogeneous, it stays disordered, where pycalphad's equilibrium at fixed
    composition would split it into gamma + gamma' half a kelvin below the solvus."""
    db = _alcrni()
    for T, lo, hi in ((773.0, 40.0, 41.0), (873.0, 43.0, 43.5), (973.0, 45.0, 46.0)):
        eq = equilibrium(db, ["AL", "NI", "VA"], ["FCC_A1", "L12_FCC"], {v.T: T, v.P: 101325, v.N: 1, v.X("AL"): 0.15})
        xm, xp = sorted(eq.X.values.reshape(-1, 2)[:2], key=lambda r: r[0])
        h = solution_enthalpy(db, ["AL", "NI"], "FCC_A1", xm, "L12_FCC", xp, T)
        assert h == pytest.approx(solution_enthalpy(db, ["AL", "NI"], "FCC_A1", xm, "L12_FCC", xp, T, dT=0.1), rel=1e-5)
        gamma = 1e3 * broken_bond_energy(h, 6.6e-6, "fcc")
        assert lo < gamma < hi and 14 < gamma < 80
        assert solution_enthalpy(db, ["AL", "NI"], "L12_FCC", xm, "L12_FCC", xp, T) == pytest.approx(h, rel=1e-8)


def test_input_checks():
    with pytest.raises(ValueError, match="fcc"):
        broken_bond_factor("hcp")
    with pytest.raises(ValueError, match="exceed 4"):
        effective_coordination("fcc", 4.0)
    with pytest.raises(ValueError, match="three indices"):
        broken_bond_factor("fcc", (0, 0, 0))
    with pytest.raises(ValueError, match="shell"):
        broken_bond_factor("bcc", (1, 0, 0), shells=0)
    with pytest.raises(ValueError, match="molar volume"):
        broken_bond_energy(1e4, 0.0)
    db = Database(TOY)
    with pytest.raises(ValueError, match="present in the matrix"):
        solution_enthalpy(db, ["CU", "NI"], "FCC_A1", [1.0, 0.0], "PREC", [0, 1], 800.0)
    with pytest.raises(ValueError, match="summing to 1"):
        solution_enthalpy(db, ["CU", "NI"], "FCC_A1", [0.5, 0.4], "PREC", [0, 1], 800.0)
