"""Nucleation of a new phase at the interface between two others, after d'Heurle, J. Mater. Res. 3 (1988) 167.

A new phase AB forming between A and B replaces the A/B interface (energy sigma_1) by two, A/AB and AB/B (sigma_2,
sigma_3), so its interfacial energy rises by Delta sigma = sigma_2 + sigma_3 - sigma_1 (eq. 11) while its formation
releases Delta h per volume. A planar layer pays for itself once thicker than the specific thickness
d_s = Delta sigma / Delta h (eq. 12), d'Heurle's gauge of how hard a phase is to nucleate. A discrete nucleus at the
interface is a lens of two spherical caps meeting the old interface along a circle; with all three energies equal
its caps meet at 120 degrees and it adds (5 pi / 4) R^2 of interface and holds (5 pi / 12) R^3 for caps of radius R,
so R* = 2 sigma / Delta h (eqs. 14-16).

For unequal energies the rim obeys Neumann's triangle (p. 177): sigma_2 cos theta_2 + sigma_3 cos theta_3 = sigma_1
and sigma_2 sin theta_2 = sigma_3 sin theta_3, theta being the angle between a cap and the old interface. The lens
exists only if each energy is less than the sum of the other two (eqs. 17-18); a cap is a hemisphere when
sigma_1^2 + sigma_2^2 = sigma_3^2 (eq. 19). Both caps have the same Laplace pressure, so at the critical size each
has radius 2 sigma_i / Delta g and the barrier is a third of the interfacial energy added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_A = 6.02214076e23  # 1/mol


def specific_thickness(interface_energy_change: float, driving_force: float) -> float:
    """d_s = Delta sigma / Delta h (m), eq. (12), for Delta sigma in J/m^2 and the formation energy released per
    volume of the new phase Delta h in J/m^3."""
    if driving_force <= 0:
        raise ValueError("the new phase must release energy as it forms (driving force > 0)")
    return interface_energy_change / driving_force


def molar_volume(cell_volume: float, formula_units: int) -> float:
    """Volume per mole of formula units, N_A V_cell / Z (m^3/mol), as d'Heurle obtains it from unit cells."""
    if cell_volume <= 0 or formula_units < 1:
        raise ValueError("need a positive cell volume and at least one formula unit per cell")
    return N_A * cell_volume / formula_units


@dataclass(frozen=True)
class InterfaceNucleus:
    """The critical lens at an interface: the angles (rad) its caps make with the old interface on the A and B
    sides, their radii, the radius of the rim, its height normal to the interface, its volume (m^3), the interfacial
    energy it adds (J) and the barrier (J)."""

    angle_a: float
    angle_b: float
    cap_radius_a: float
    cap_radius_b: float
    rim_radius: float
    height: float
    volume: float
    area_energy: float
    barrier: float


def _cap(radius, angle):
    c = np.cos(angle)
    return 2 * np.pi * radius**2 * (1 - c), np.pi * radius**3 * (2 - 3 * c + c**3) / 3


def interface_nucleus(sigma_old: float, sigma_a: float, sigma_b: float, driving_force: float) -> InterfaceNucleus:
    """The critical nucleus of a new phase at the A/B interface (energy ``sigma_old``), bounded by A/new and new/B
    interfaces of energies ``sigma_a`` and ``sigma_b`` (J/m^2), for a driving force per volume ``driving_force``
    (J/m^3). Raises ValueError where no lens forms: if sigma_a + sigma_b <= sigma_old the new phase wets the
    interface and there is no barrier, and if one of the new interfaces costs more than the other two together
    (eqs. 17-18) the nucleus leaves the interface."""
    if min(sigma_old, sigma_a, sigma_b) <= 0 or driving_force <= 0:
        raise ValueError("interface energies and the driving force must be positive")
    if sigma_a + sigma_b <= sigma_old:
        raise ValueError("sigma_a + sigma_b <= sigma_old: the new phase wets the interface, so it needs no nucleus")
    if sigma_old + sigma_a <= sigma_b or sigma_old + sigma_b <= sigma_a:
        raise ValueError("eqs. (17)-(18) fail: one new interface costs more than the other two, so the nucleus "
                         "does not sit on the interface")
    # Neumann's triangle: the three tensions balance at the rim
    angle_a = np.arccos((sigma_a**2 + sigma_old**2 - sigma_b**2) / (2 * sigma_a * sigma_old))
    angle_b = np.arccos((sigma_b**2 + sigma_old**2 - sigma_a**2) / (2 * sigma_b * sigma_old))
    radius_a, radius_b = 2 * sigma_a / driving_force, 2 * sigma_b / driving_force
    rim = radius_a * np.sin(angle_a)
    area_a, volume_a = _cap(radius_a, angle_a)
    area_b, volume_b = _cap(radius_b, angle_b)
    added = sigma_a * area_a + sigma_b * area_b - sigma_old * np.pi * rim**2
    volume = volume_a + volume_b
    height = radius_a * (1 - np.cos(angle_a)) + radius_b * (1 - np.cos(angle_b))
    return InterfaceNucleus(float(angle_a), float(angle_b), radius_a, radius_b, float(rim), float(height),
                            float(volume), float(added), float(added - driving_force * volume))
