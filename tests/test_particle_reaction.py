"""Plate, cylinder and sphere conversion, checked against van Loo, Prog. Solid State Chem. 20 (1990) 47, pp. 74-77.

Fig. 2.20 and 2.21 use z = 2, r0 = 100 and k_p = 50 in arbitrary units.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from janus_kinetics import LineCompound, TemperatureSchedule, parabolic_growth
from janus_kinetics.particle_reaction import (
    conversion,
    conversion_integral,
    growth_constant_from_conversion,
    layer_thickness,
)

KP, R0, Z = 50.0, 100.0, 2.0


@pytest.mark.parametrize("geometry", ["cylinder", "sphere"])
@pytest.mark.parametrize("z", [0.7, 1.0, 2.0])
def test_eq_26_follows_from_steady_diffusion_across_the_shell(geometry, z):
    """Re-derivation: a planar layer of thickness d uses up A at k_p / (z d) per unit
    area and time, so a shell between core radius a and outer radius b uses it up
    at k_p/z times the shell conductance, 2 pi / ln(b/a) per length of cylinder or
    4 pi a b / (b - a) for a sphere. Integrating gives eq. 26."""
    n = 2 if geometry == "cylinder" else 3

    def rhs(t, y):
        a = y[0]
        b = (a**n + z * (R0**n - a**n)) ** (1 / n)
        if geometry == "cylinder":
            return [-KP / (z * a * np.log(b / a))]
        return [-KP * b / (z * a * (b - a))]

    times = np.array([2.0, 6.0, 12.0])  # short of complete conversion for every z here
    sol = solve_ivp(rhs, (1e-12, times[-1]), [R0 * (1 - 1e-12)], t_eval=times, method="Radau", rtol=1e-11, atol=1e-12)
    alpha = 1 - (sol.y[0] / R0) ** n
    assert conversion_integral(alpha, z, geometry) == pytest.approx(2 * KP * times / R0**2, rel=1e-6)


@pytest.mark.parametrize("geometry, factor", [("plate", 1.0), ("cylinder", 4.0), ("sphere", 9.0)])
def test_eq_27_small_conversion_limits(geometry, factor):
    """(z alpha)^2 = 2, 8 and 18 k_p t / r0^2 for small alpha (eq. 27)."""
    alpha = 1e-5
    assert factor * conversion_integral(alpha, Z, geometry) == pytest.approx((Z * alpha) ** 2, rel=1e-4)


def test_printed_z_equal_one_forms():
    a = np.linspace(0.0, 1.0, 11)
    cyl = 0.5 * a + 0.5 * np.where(a < 1, (1 - a) * np.log(np.where(a < 1, 1 - a, 1.0)), 0.0)
    assert conversion_integral(a, 1.0, "cylinder") == pytest.approx(cyl, rel=1e-12, abs=1e-15)
    assert conversion_integral(a, 1.0, "sphere") == pytest.approx((1 - 2 * a / 3) - (1 - a) ** (2 / 3), rel=1e-12, abs=1e-15)
    for g in ("cylinder", "sphere"):  # continuity across the z = 1 special case
        assert conversion_integral(a, 1 + 1e-9, g) == pytest.approx(conversion_integral(a, 1.0, g), rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("geometry, t_half, d_end", [("plate", 20.0, 200.0), ("cylinder", 12.0, 140.0), ("sphere", 9.0, 125.0)])
def test_end_points_of_figs_2_20_and_2_21(geometry, t_half, d_end):
    """Complete conversion at t^(1/2) of about 20, 12 and 9, with final layers of about
    200, 140 and 125 (figure readings, so +-0.3 and +-3)."""
    t_full = conversion_integral(1.0, Z, geometry) * R0**2 / (2 * KP)
    assert np.sqrt(t_full) == pytest.approx(t_half, abs=0.3)
    assert layer_thickness(1.0, Z, geometry) * R0 == pytest.approx(d_end, abs=3.0)
    assert conversion(t_full * (1 - 1e-9), KP, R0, Z, geometry)[0] < 1.0
    assert conversion(t_full * 1.01, KP, R0, Z, geometry)[0] == 1.0


def test_initial_slopes_are_one_two_and_three():
    """Fig. 2.20: alpha against t^(1/2) starts with slopes 1, 2, 3 times (2 k_p)^(1/2) / (z r0)."""
    t = 1e-6
    unit = np.sqrt(2 * KP * t) / (Z * R0)
    for geometry, slope in (("plate", 1), ("cylinder", 2), ("sphere", 3)):
        assert conversion(t, KP, R0, Z, geometry)[0] / unit == pytest.approx(slope, rel=1e-3)


@pytest.mark.parametrize("geometry, alpha", [("cylinder", 0.95), ("sphere", 0.90)])
def test_ten_percent_thicker_than_planar(geometry, alpha):
    """p. 76: the layer is 10 % thicker than in a planar couple at alpha of about 0.95
    (cylinder) and 0.90 (sphere). Eq. 25 puts those layers at 1.17 and 0.77 r0,
    which the text rounds to 'about 1.1' and '0.7'."""
    planar = np.sqrt(conversion_integral(alpha, Z, geometry))  # sqrt(2 k_p t) / r0 at the same time
    assert layer_thickness(alpha, Z, geometry) / planar == pytest.approx(1.10, abs=0.01)


def test_growth_constant_round_trip():
    t = np.array([10.0, 50.0, 80.0])
    for g in ("plate", "cylinder", "sphere"):
        alpha = conversion(t, KP, R0, Z, g)
        assert growth_constant_from_conversion(alpha, t, R0, Z, g) == pytest.approx(KP, rel=1e-9)


def test_schedule_uses_the_time_integral_of_kp():
    schedule = TemperatureSchedule([0.0, 30.0, 90.0], [600.0, 900.0, 900.0])
    kp = lambda T: 1e-14 * np.exp(-90e3 / (8.314 * T)) / np.exp(-90e3 / (8.314 * 900.0))
    from scipy.integrate import quad
    exposure = quad(lambda s: kp(schedule(s)), 0, 90.0, points=[30.0], epsabs=0, epsrel=1e-12)[0]
    got = conversion(90.0, kp, 1e-6, 1.6, "sphere", temperature=schedule)[0]
    same = conversion(exposure / 1e-14, 1e-14, 1e-6, 1.6, "sphere")[0]
    assert got == pytest.approx(same, rel=1e-9)


def test_z_and_kp_from_line_compounds_reproduce_a_consumed_film():
    """With k_p and z from a planar line-compound couple, the plate result matches the
    time at which reaction_layers uses up a film of thickness r0."""
    a = LineCompound("A", 0.0, 1.0e-5)
    ab = LineCompound("AB", 0.5, 1.3e-5, integrated_diffusivity=1e-16, tracer_ratio=0.3)
    b = LineCompound("B", 1.0, 1.1e-5)
    g = parabolic_growth([a, ab, b], 1000.0)
    z = (ab.molar_volume / 0.5) / a.molar_volume
    r0 = 1e-6
    t_film = (r0 / g.consumed["A"]) ** 2
    t_plate = conversion_integral(1.0, z, "plate") * r0**2 / (2 * g.growth_constants["AB"])
    assert t_plate == pytest.approx(t_film, rel=1e-10)


def test_input_checks():
    with pytest.raises(ValueError, match="geometry"):
        conversion_integral(0.5, 2.0, "cube")
    with pytest.raises(ValueError, match="positive"):
        layer_thickness(0.5, -1.0, "sphere")
    with pytest.raises(ValueError, match="needs a temperature"):
        conversion(1.0, lambda T: 1e-15, 1e-6, 2.0, "sphere")
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        conversion_integral(1.2, 2.0, "plate")
