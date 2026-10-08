"""Nucleation, growth and coarsening with the parameters of Perez et al. (2008), Table 1.

Perez, Dumont and Acevedo-Reyes, Acta Mater. 56 (2008) 2119, Table 1 (p. 2125):
X^P = 1, X^0 = 0.06, X^i = 0.01, D = 5e-20 m^2/s, gamma = 0.13 J/m^2,
v_at = 1.6e-29 m^3, a = 4.04e-10 m, T = 160 C. The paper does not give the number
of nucleation sites; every lattice site is counted here. The run reproduces the
sequence of their Fig. 7 but not its values, which do not follow from Table 1
(see ROADMAP.md).
Run ``python examples/precipitation_table1.py`` to write examples/output/precipitation_table1.png.
"""

import numpy as np

from janus_kinetics import DiluteSolution, Kinetics, solve_kwn

from _common import figure_path

V_AT = 1.6e-29
THERMO = DiluteSolution(x_eq=0.01, x_p=1.0, v_at_p=V_AT, v_at_m=V_AT, gamma=0.13, T=160 + 273.15)
KINETICS = Kinetics(diffusivity=5e-20, lattice_parameter=4.04e-10, sites=1 / V_AT)


def run(coarse: bool = False) -> dict:
    edges = np.arange(0.2e-9, 60e-9, 0.1e-9 if coarse else 0.05e-9)
    saves = np.logspace(1, 7, 25 if coarse else 61)[:-1]
    return {"result": solve_kwn(THERMO, KINETICS, 0.06, 1e7, edges, save_times=saves)}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = run()["result"]
    t = res.times[1:]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    (a, b), (c, d) = axes
    a.loglog(t, res.mean_radius[1:] * 1e9, "k-", label="mean radius")
    a.loglog(t, np.where(np.isfinite(res.critical_radius[1:]), res.critical_radius[1:] * 1e9, np.nan), "k--", label="critical radius")
    a.set_ylabel("Radius (nm)")
    a.legend(frameon=False)
    b.semilogx(t, res.number_density[1:], "k-")
    b.set_ylabel("Number density (m^-3)")
    c.semilogx(t, res.matrix_fraction[1:], "k-")
    c.set_ylabel("Matrix solute fraction")
    d.semilogx(t, res.volume_fraction[1:], "k-")
    d.set_ylabel("Precipitate volume fraction")
    for ax in (c, d):
        ax.set_xlabel("Time (s)")
    fig.suptitle("Precipitation with Perez et al. Table 1 Parameters")
    fig.tight_layout()
    fig.savefig(figure_path("precipitation_table1.png"), dpi=150)
    i = int(np.argmax(res.number_density))
    print(f"peak number density {res.number_density[i]:.3e} m^-3 at {res.times[i]:.3g} s; "
          f"mean radius after 1e7 s {res.mean_radius[-1] * 1e9:.2f} nm")


if __name__ == "__main__":
    main()
