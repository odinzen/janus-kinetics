"""Ageing and reheating of an Al-Mg-Si alloy (AA 6082) with Myhr and Grong's model.

The inputs are their Table 2 (Acta Mater. 48 (2000) 1605): Mg2Si particles from 0.63 wt% Mg, solvus 970 exp(-47175/RT)
wt%, Mg diffusivity 2.2e-4 exp(-130000/RT) m^2/s, interfacial energy 0.2 J/m^2. The left panel ages the alloy at 180 C:
particles nucleate for about 1e4 s, after which the matrix is depleted and they coarsen. The right panel heats under-,
peak- and over-aged material (5e3, 1e4 and 4e4 s at 180 C) at 1 C/s; the volume fractions part at first and meet near
300 C, then follow the equilibrium fraction down to the 500 C solvus.
Run ``python examples/aluminium_heat_treatment.py`` to write examples/output/aluminium_heat_treatment.png.
"""

import numpy as np

from janus_kinetics import TemperatureSchedule
from janus_kinetics.heat_treatment import aa6082, solve_heat_treatment, stretched_radii

from _common import figure_path

AGEING = 453.15  # K
TEMPERS = {"under-aged": 5e3, "peak-aged": 1e4, "over-aged": 4e4}  # s at 180 C


def run(coarse: bool = False) -> dict:
    alloy = aa6082()
    radii = stretched_radii(0.04e-9 if coarse else 0.02e-9, 15e-9, 1.01 if coarse else 1.005, 2e-6)
    times = np.logspace(0, 6.5, 27 if coarse else 131)
    ageing = solve_heat_treatment(alloy, AGEING, times[-1], radii=radii, save_times=times)
    heat = TemperatureSchedule([0, 330], [AGEING, AGEING + 330])
    temps = np.arange(180.0, 511.0, 10.0 if coarse else 2.0)
    heating = {}
    for name, t in TEMPERS.items():
        start = solve_heat_treatment(alloy, AGEING, t, radii=radii).state()
        res = solve_heat_treatment(alloy, heat, temps[-1] - 180, radii=radii, save_times=temps - 180, initial=start)
        heating[name] = res.volume_fraction[:temps.size]
    return {"ageing": ageing, "temps": temps, "heating": heating,
            "equilibrium": alloy.equilibrium_fraction(temps + 273.15)}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run()
    ageing = out["ageing"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    a.semilogx(ageing.times, ageing.mean_radius * 1e9, "k-", label="mean radius")
    a.semilogx(ageing.times, np.minimum(ageing.critical_radius, 1e3) * 1e9, "k:", label="critical radius")
    a.set_ylim(0, 8)
    a.set_xlabel("Time at 180 C (s)")
    a.set_ylabel("Radius (nm)")
    a2 = a.twinx()
    a2.loglog(ageing.times, ageing.number_density, color="0.6", label="number density")
    a2.set_ylabel("Particles per m^3", color="0.4")
    lines = a.get_lines() + a2.get_lines()
    a.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8, loc="upper left")
    for (name, f), style in zip(out["heating"].items(), ("--", "-", "-.")):
        b.plot(out["temps"], f, "k" + style, label=name)
    b.plot(out["temps"], out["equilibrium"], color="0.6", ls=":", lw=2, label="equilibrium")
    b.set_xlabel("Temperature, heating at 1 C/s (C)")
    b.set_ylabel("Particle volume fraction")
    b.legend(frameon=False, fontsize=8)
    fig.suptitle("Precipitation in AA 6082 during Ageing and Reheating")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(figure_path("aluminium_heat_treatment.png"), dpi=150)
    k = np.argmin(np.abs(ageing.times - 1e4))
    print(f"after 1e4 s at 180 C: f = {ageing.volume_fraction[k]:.4f}, mean radius {ageing.mean_radius[k] * 1e9:.2f} nm, "
          f"{ageing.number_density[k]:.2e} particles per m^3")
    for name, f in out["heating"].items():
        print(f"{name}: f = {f[0]:.4f} at 180 C, {f[out['temps'] == 300][0]:.4f} at 300 C, max {f.max():.4f}")


if __name__ == "__main__":
    main()
