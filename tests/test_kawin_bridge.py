"""The optional bridge to kawin (Ury et al., Acta Mater. 255 (2023) 118988).

The database merge is checked without kawin. Everything that runs kawin is skipped when kawin is not
installed (pip install kawin).

The case is gamma' (L1_2) in Ni-5.2Al-14.2Cr at.% at 873 K on pycalphad's alcrni.tdb, Dupin, Ansara and
Sundman, Calphad 25 (2001) 279, with the mobilities of Campbell, Boettinger and Kattner (2002), as in
test_multicomponent_precipitation.py.
"""

import functools
import importlib.util
import warnings
from importlib.resources import files

import numpy as np
import pytest
from pycalphad import Database, variables as v
from tinydb import where

from janus_kinetics import SolutionPhase, campbell2002_fcc_ni
from janus_kinetics.kawin_bridge import independent_copy, kawin_elements, local_compositions, merge_mobilities
from janus_kinetics.multicomponent_precipitation import (
    N_A,
    NucleationSettings,
    PrecipitateThermo,
    solve_multicomponent_precipitation,
)

ALCRNI = files("pycalphad") / "tests" / "databases" / "alcrni.tdb"
ELEMENTS = ["AL", "CR", "NI"]
T = 873.0
ALLOY = np.array([0.052, 0.142])
# Sudbrack, Noebe and Seidman, Acta Mater. 55 (2007) 119, section 4.1 and Table 5, as in
# test_multicomponent_precipitation.py
SETTINGS = NucleationSettings(gamma=0.0225, molar_volume=6.7584e-6, lattice_parameter=0.3552e-9)
needs_kawin = pytest.mark.skipif(importlib.util.find_spec("kawin") is None,
                                 reason="kawin is not installed (pip install kawin)")


@functools.lru_cache(maxsize=None)
def _db():
    if not ALCRNI.is_file():
        pytest.skip("pycalphad's alcrni.tdb test database is not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database(str(ALCRNI))


@functools.lru_cache(maxsize=None)
def _merged():
    return merge_mobilities(_db(), campbell2002_fcc_ni(), "FCC_A1", ELEMENTS)


def _phase(db, mobility_db=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SolutionPhase(db, "FCC_A1", ELEMENTS, mobility_db=mobility_db)


def test_kawin_elements_put_the_solvent_first():
    assert kawin_elements(["AL", "CR", "NI"]) == ["NI", "AL", "CR", "VA"]
    assert kawin_elements(["NI", "AL", "CR", "VA"], dependent="NI") == ["NI", "AL", "CR", "VA"]
    with pytest.raises(ValueError, match="not among"):
        kawin_elements(["AL", "NI"], dependent="CR")


def test_merged_database_gives_the_same_mobilities():
    """One merged file read by janus gives the mobilities of the two separate files exactly, at the
    alloy, near each corner and across temperatures, so nothing was dropped, doubled or re-keyed wrongly."""
    apart, together = _phase(_db(), campbell2002_fcc_ni()), _phase(_merged())
    xs = np.array([[0.052, 0.142], [0.98, 0.01], [0.01, 0.98], [0.01, 0.01], [0.3, 0.3]])
    for temp in (873.0, 1273.0, 1650.0):
        a, b = apart.mobility(xs, temp), together.mobility(xs, temp)
        assert np.all(a > 0)
        np.testing.assert_array_equal(a, b)


def test_merge_keys_parameters_onto_the_thermodynamic_lattice():
    """Campbell writes fcc as (..)(Va) and alcrni.tdb as one sublattice, so every copied parameter has one
    sublattice and only Al, Cr and Ni, at least 3 elements x 3 end members."""
    params = _merged().search((where("phase_name") == "FCC_A1") & (where("parameter_type") == "MQ"))
    source = campbell2002_fcc_ni().search((where("phase_name") == "FCC_A1") & (where("parameter_type") == "MQ"))
    allowed = {v.Species(el) for el in ELEMENTS}
    kept = [p for p in source if p["diffusing_species"].name in ELEMENTS and set(p["constituent_array"][0]) <= allowed]
    assert len(params) == len(kept) >= 9
    for p in params:
        assert len(p["constituent_array"]) == 1
        assert set(p["constituent_array"][0]) <= allowed
    # the thermodynamic database itself is left as it was
    assert not _db().search(where("parameter_type") == "MQ")


def test_merge_refuses_a_second_copy():
    with pytest.raises(ValueError, match="already holds MQ"):
        merge_mobilities(_merged(), campbell2002_fcc_ni(), "FCC_A1", ELEMENTS)


def test_merge_leaves_out_other_elements():
    binary = merge_mobilities(_db(), campbell2002_fcc_ni(), "FCC_A1", ["AL", "NI"])
    names = {sp.name for p in binary.search(where("parameter_type") == "MQ") for sp in p["constituent_array"][0]}
    assert names == {"AL", "NI"}
    assert {p["diffusing_species"].name for p in binary.search(where("parameter_type") == "MQ")} == {"AL", "NI"}


def test_independent_copy_shares_nothing():
    db = _db()
    c = independent_copy(db)
    assert c == db
    c.phases["EXTRA"] = c.phases["FCC_A1"]
    c.symbols["EXTRA"] = 1.0
    assert "EXTRA" not in db.phases and "EXTRA" not in db.symbols


def test_local_compositions_interpolate_the_profile():
    z = np.array([1.0, 2.0, 4.0]) * 1e-6
    x = np.array([[0.04, 0.13], [0.05, 0.14], [0.07, 0.18]])
    got = local_compositions(z, x, [1e-6, 1.5e-6, 3e-6, 4e-6])
    np.testing.assert_allclose(got, [[0.04, 0.13], [0.045, 0.135], [0.06, 0.16], [0.07, 0.18]], rtol=1e-12)
    with pytest.raises(ValueError, match="between the first and last"):
        local_compositions(z, x, [0.5e-6])


# --- with kawin ----------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _kawin_thermo():
    from janus_kinetics.kawin_bridge import kawin_thermodynamics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kawin_thermodynamics(_db(), "FCC_A1", ["L12_FCC"], ELEMENTS, mobility_db=campbell2002_fcc_ni())


def _janus_nucleus(db):
    matrix = _phase(db, campbell2002_fcc_ni())
    mu = np.asarray(matrix.chemical_potentials(np.append(ALLOY, 1 - ALLOY.sum()), T), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PrecipitateThermo(db, "L12_FCC", ELEMENTS).nucleus(mu, T)


@needs_kawin
def test_kawin_sees_janus_thermodynamics():
    """Both find the same gamma' nucleus against the as-quenched matrix by the parallel tangent: 16.37 Al,
    8.25 Cr at.% and 368.8 J/mol (Sudbrack et al. (2007) Table 3, row [28]: 16.43 Al, 8.21 Cr)."""
    x, dg = _janus_nucleus(_db())
    dg_k, x_k = _kawin_thermo().getDrivingForce(list(ALLOY), T)
    assert float(dg_k) == pytest.approx(dg, rel=1e-5)
    np.testing.assert_allclose(np.ravel(x_k), x[:2], atol=1e-5)
    assert dg == pytest.approx(369.0, abs=2.0)


@needs_kawin
def test_kawin_sees_janus_mobilities():
    """The interdiffusion matrices (Ni the dependent element) agree to rounding; the tracer diffusivities
    differ only by kawin's gas constant, 8.314 against 8.314462618 J/(mol K)."""
    matrix = _phase(_db(), campbell2002_fcc_ni())
    for x, temp in ((ALLOY, T), ((0.07, 0.178), T), ((0.10, 0.085), 1073.0)):
        x = np.asarray(x)
        d_janus = np.asarray(matrix.interdiffusion_matrix(x, temp), dtype=float)
        d_kawin = np.asarray(_kawin_thermo().getInterdiffusivity(list(x), temp), dtype=float)
        np.testing.assert_allclose(d_kawin, d_janus, rtol=1e-7)
        tracer = 8.314462618 * temp * np.asarray(matrix.mobility(x, temp), dtype=float)  # Al, Cr, Ni
        tracer_k = np.asarray(_kawin_thermo().getTracerDiffusivity(list(x), temp), dtype=float)  # Ni, Al, Cr
        np.testing.assert_allclose(tracer_k[[1, 2, 0]], tracer * 8.314 / 8.314462618, rtol=1e-6)


@needs_kawin
def test_kawin_leaves_the_database_alone():
    """kawin adds a DIS_FCC_A1 phase to the database it reads; on the caller's database that would turn
    janus's gamma' nucleus into nonsense, so the bridge must hand kawin a copy."""
    db = independent_copy(_db())
    before = _janus_nucleus(db)
    from janus_kinetics.kawin_bridge import kawin_thermodynamics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kawin_thermodynamics(db, "FCC_A1", ["L12_FCC"], ELEMENTS)
    assert sorted(db.phases) == sorted(_db().phases)
    after = _janus_nucleus(db)
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1] == before[1]


@needs_kawin
def test_kawin_model_carries_janus_settings():
    """Same molar volume (to kawin's Avogadro constant), cell edge, sites, interfacial energy and alloy."""
    from janus_kinetics.kawin_bridge import kawin_model

    model = kawin_model(_kawin_thermo(), "L12_FCC", ALLOY, T, SETTINGS, ["AL", "CR"])
    for vol in (model.matrix.volume, model.precipitates[0].volume):
        assert vol.Vm == pytest.approx(SETTINGS.molar_volume, rel=1e-4)
        assert vol.a == pytest.approx(SETTINGS.lattice_parameter, rel=1e-12)
    assert model.matrix.nucleationSites.bulkN0 == pytest.approx(N_A / SETTINGS.molar_volume, rel=1e-12)
    assert model.precipitates[0].gamma == SETTINGS.gamma
    np.testing.assert_array_equal(np.ravel(model.matrix.initComposition), ALLOY)
    with pytest.raises(NotImplementedError, match="bulk nucleation"):
        kawin_model(_kawin_thermo(), "L12_FCC", ALLOY, T,
                    NucleationSettings(0.0225, 6.7584e-6, 0.3552e-9, misfit=1e6), ["AL", "CR"])


@needs_kawin
@pytest.mark.slow
def test_history_is_kawin_own_record():
    """After a 10 min ageing the history holds kawin's own arrays, the matrix composition comes back in
    janus's order, and the strength is what kawin's strength model gives for the solved run."""
    from kawin.precipitation.coupling import APBContribution, DislocationParameters, StrengthModel

    from janus_kinetics.kawin_bridge import kawin_history, kawin_model

    model = kawin_model(_kawin_thermo(), "L12_FCC", ALLOY, T, SETTINGS, ["AL", "CR"])
    # arbitrary strength inputs, since only the plumbing is tested here
    strength = StrengthModel(["L12_FCC"], [APBContribution(yAPB=0.1)], DislocationParameters(G=60e9, b=0.25e-9))
    model.addCouplingModel(strength)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.solve(600.0, verbose=False)
    h = kawin_history(model, strength)
    data = model.toDict()
    np.testing.assert_array_equal(h.times, data["time"])
    np.testing.assert_array_equal(h.mean_radius, np.asarray(data["Ravg"])[:, 0])
    np.testing.assert_array_equal(h.volume_fraction, np.asarray(data["volFrac"])[:, 0])
    np.testing.assert_array_equal(h.number_density, np.asarray(data["precipitateDensity"])[:, 0])
    np.testing.assert_array_equal(h.strength, strength.totalStrength(model))
    assert h.times[-1] == pytest.approx(600.0, rel=1e-9)
    assert h.volume_fraction[-1] > 0.01 and h.strength[-1] > 0
    assert h.matrix_composition[0, 0] == ALLOY[0] and h.matrix_composition[-1, 0] < ALLOY[0]


@needs_kawin
@pytest.mark.slow
def test_kawin_and_janus_age_the_same_alloy_alike():
    """Sudbrack's ageing, 10 min to 1024 h at 873 K, in both packages from the same database, mobilities and
    settings, janus with evolving precipitate compositions (gamma' tracer diffusivities as in
    test_multicomponent_precipitation.py) since kawin's default keeps each particle at its interface
    composition. They end on the same state, volume fraction 11.05 against 10.97 % and matrix Al 4.02
    against 4.03 at.% at 1024 h. The path differs by model. kawin nucleates half as many particles,
    which grow larger (radius up to 31 % above janus at 16 h) and coarsen faster, 13.3 nm at 1024 h. janus
    gives 11.5 to 12.6 nm there depending on the saved times and on 1-4 times tighter step limits, a
    scatter from merging and splitting size classes that does not shrink with the step; its volume
    fraction moves by 0.1 %. With janus's default, frozen nucleus compositions, its gamma' holds 16.4 % Al
    instead of 14.7 % and the volume fraction stays near 9.8 %, as the lever rule says it must."""
    from janus_kinetics.kawin_bridge import kawin_history, kawin_model

    hours = np.array([0.167, 1, 4, 16, 64, 256, 1024])
    model = kawin_model(_kawin_thermo(), "L12_FCC", ALLOY, T, SETTINGS, ["AL", "CR"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.solve(hours[-1] * 3600, verbose=False)
        janus = solve_multicomponent_precipitation(_phase(_db(), campbell2002_fcc_ni()),
                                                   PrecipitateThermo(_db(), "L12_FCC", ELEMENTS),
                                                   np.append(ALLOY, 1 - ALLOY.sum()), T, hours[-1] * 3600,
                                                   SETTINGS, hours * 3600,
                                                   precipitate_tracer=[2.7e-23, 1.4e-23, 2.7e-23])
    k = kawin_history(model)

    def at(values):
        return np.interp(np.log(hours * 3600), np.log(k.times[1:]), values[1:])

    assert k.times[-1] == pytest.approx(hours[-1] * 3600, rel=1e-9)
    late = slice(-2, None)  # 256 and 1024 h
    np.testing.assert_allclose(at(k.volume_fraction)[late], janus.volume_fraction[late], rtol=0.02)
    np.testing.assert_allclose(k.matrix_composition[-1], janus.matrix_composition[-1][:2], atol=3e-4)
    ratio_r = at(k.mean_radius)[1:] / janus.mean_radius[1:]
    assert np.all((ratio_r > 1.0) & (ratio_r < 1.35))
    assert ratio_r[-1] < 1.2
    ratio_n = at(k.number_density) / janus.number_density
    assert np.all((ratio_n > 0.4) & (ratio_n < 0.85))
