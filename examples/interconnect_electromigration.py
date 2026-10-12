"""Stress build-up by electromigration in a passivated aluminium line.

A 50 um line between two blocking boundaries (tungsten studs, say) carries about 1 MA/cm^2 at 200 C. The
numbers are Korhonen, Borgesen, Tu and Li's (J. Appl. Phys. 73 (1993) 3790): G = 16.5 MPa/um for that current
(p. 3793) and kappa = 2.6 um^2/h, their fit to Hemmert and Costa's measured stresses (Fig. 9). Atoms leave the
cathode end and pile up at the anode, building tension and compression until the stress gradient balances the
electron wind: Blech's steady state, sigma = G L (1/2 - x/L), with +-412 MPa at the ends. The line sustains that
only if its passivation and yield strength do, which is Blech's threshold read the other way.
Run ``python examples/interconnect_electromigration.py`` to write examples/output/interconnect_electromigration.png.
"""

import numpy as np

from janus_kinetics import finite_line_stress, semi_infinite_stress, solve_line_stress, steady_stress

from _common import figure_path

G = 16.5e12  # Pa/m
KAPPA = 2.6e-12 / 3600  # m^2/s
LENGTH = 50e-6
HOURS = np.array([10.0, 50.0, 200.0, 1000.0])


def run(coarse: bool = False) -> dict:
    res = solve_line_stress(LENGTH, 100 if coarse else 400, G, KAPPA, HOURS[-1] * 3600, 3600.0 if coarse else 600.0,
                            save_times=HOURS[:-1] * 3600)
    return {"result": res}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = run()["result"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    for i, style in zip(range(1, len(res.times)), (":", "-.", "--", "-")):
        a.plot(res.x * 1e6, res.stress[i] / 1e6, "k" + style, label=f"{res.times[i] / 3600:.0f} h")
    a.plot(res.x * 1e6, steady_stress(res.x, LENGTH, G) / 1e6, color="0.6", lw=3, alpha=0.5, label="steady state")
    a.set_xlabel("Distance along the line (um)")
    a.set_ylabel("Stress (MPa)")
    a.legend(frameon=False, fontsize=8)
    t = np.linspace(0.0, HOURS[-1], 400) * 3600
    b.plot(t / 3600, finite_line_stress(0.0, t, LENGTH, KAPPA, G) / 1e6, "k-", label="line of 50 um")
    b.plot(t / 3600, semi_infinite_stress(0.0, t, KAPPA, G) / 1e6, "k--", label="long line")
    b.axhline(G * LENGTH / 2 / 1e6, color="0.6", ls=":")
    b.set_xlabel("Time (h)")
    b.set_ylabel("Tensile stress at the cathode end (MPa)")
    b.legend(frameon=False, fontsize=8)
    fig.suptitle("Electromigration Stress in a Passivated Aluminium Line")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(figure_path("interconnect_electromigration.png"), dpi=150)
    end = finite_line_stress(0.0, HOURS * 3600, LENGTH, KAPPA, G)
    print("cathode-end stress (MPa) at", HOURS, "h:", np.round(end / 1e6, 1), f"; steady {G * LENGTH / 2 / 1e6:.0f}")


if __name__ == "__main__":
    main()
