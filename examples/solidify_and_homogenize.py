"""Microsegregation of a cast Ni-5.2Al-14.2Cr (at.%) alloy and its removal by a homogenization anneal.

The primary gamma solidifies by the Scheil-Gulliver path (Bocklund et al., Materialia 11 (2020)
100689) on pycalphad's alcrni.tdb, the Al-Cr-Ni description of Dupin, Ansara and Sundman (2001),
with the gamma' that would form from the last liquid left out. The solid laid down is spread across
a secondary dendrite arm, taken as a cylinder of radius 25 um (half of a 50 um arm spacing), core
first. That profile is annealed at 1200 C in the same cylinder with the mobilities of Campbell,
Boettinger and Kattner (2002). The arm spacing and the anneal are illustrative choices.
The thin Al- and Cr-rich shell left by the last liquid goes within minutes; the Al profile then turns
over for a while, Al pushed toward the axis by the Cr gradient (cross-diffusion), before both flatten.
Run ``python examples/solidify_and_homogenize.py`` to write examples/output/solidify_and_homogenize.png.
"""

import warnings
from importlib.resources import files

import numpy as np
from pycalphad import Database

from janus_kinetics import SolutionPhase, campbell2002_fcc_ni, solve_planar
from janus_kinetics.solidification import scheil_gulliver, segregation_index, segregation_profile

from _common import figure_path

ALLOY = [0.052, 0.142]  # Al, Cr; Ni balance
ARM_RADIUS = 25e-6
ANNEAL_T = 1473.0
HOURS = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])


def run(coarse: bool = False) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(str(files("pycalphad") / "tests" / "databases" / "alcrni.tdb"))
        scheil = scheil_gulliver(db, ["AL", "CR", "NI"], ["LIQUID", "FCC_A1"], ALLOY, 1700.0,
                                 1.0 if coarse else 0.25, stop_fraction=1e-3, t_min=1500.0)
        edges, solid = scheil.increments()
        faces, x0 = segregation_profile(edges, solid, ARM_RADIUS, 40 if coarse else 120, "cylindrical")
        gamma = SolutionPhase(db, "FCC_A1", ["AL", "CR", "NI"], mobility_db=campbell2002_fcc_ni())
        hours = HOURS[:4] if coarse else HOURS
        anneal = solve_planar(faces, x0, gamma.interdiffusion_function(ANNEAL_T), hours[-1] * 3600,
                              30.0 if coarse else 10.0, save_times=hours * 3600, geometry="cylindrical")
    return {"scheil": scheil, "faces": faces, "profile": x0, "anneal": anneal}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    out = run()
    res = out["anneal"]
    r = res.z * 1e6
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for i, style in zip((0, 3, 5, len(res.times) - 1), ("-", "--", "-.", ":")):
        label = "as cast" if i == 0 else f"{res.times[i] / 3600:g} h at {ANNEAL_T - 273.15:.0f} C"
        a.plot(r, 100 * res.x[i, :, 0], "k" + style, label=label)
    a.set_xlabel("Distance from the dendrite axis (um)")
    a.set_ylabel("Al (at.%)")
    a.legend(frameon=False)
    for j, (name, marker) in enumerate((("Al", "o"), ("Cr", "s"))):
        b.semilogx(res.times[1:] / 3600, segregation_index(res.x[:, :, j])[1:], "k" + marker + "-", label=name,
                   mfc="white")
    b.xaxis.set_major_formatter(ScalarFormatter())
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xlabel("Anneal time (h)")
    b.set_ylabel("Residual segregation index")
    b.legend(frameon=False)
    fig.suptitle("Scheil Microsegregation and Homogenization of Ni-5.2Al-14.2Cr")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(figure_path("solidify_and_homogenize.png"), dpi=150)
    idx = segregation_index(res.x[:, :, 0])
    print(f"as cast Al {100 * res.x[0, 0, 0]:.2f} at the axis, {100 * res.x[0, -1, 0]:.2f} at the arm edge; "
          f"residual Al segregation after {HOURS[-1]:g} h: {idx[-1]:.3f}")


if __name__ == "__main__":
    main()
