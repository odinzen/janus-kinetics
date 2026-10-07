"""Chromium loss to the surface of a bcc Fe-30Cr-2Ni (at.%) alloy at 1200 C.

The surface is held at 20 at.% Cr, as when a growing chromia scale consumes Cr,
and the ternary interdiffusion matrix makes Ni respond to the Cr gradient through
its off-diagonal terms. Mobilities: Jonsson, ISIJ Int. 35 (1995) 1415.
Thermodynamics: ideal solution, and the alloy is assumed to stay single-phase bcc.
Run ``python examples/surface_depletion.py`` to write examples/output/surface_depletion.png.
"""

import numpy as np

from calphad_kinetics import Boundary, SolutionPhase, solve_planar, uniform_faces
from calphad_kinetics.datasets import jonsson1995_bcc_crfeni

from _common import figure_path, ideal_bcc

T = 1473.15
HOURS = 10.0
BULK = np.array([0.30, 0.02])  # x_Cr, x_Ni; Fe is the balance
SURFACE = np.array([0.20, 0.02])


def run(coarse: bool = False) -> dict:
    phase = SolutionPhase(ideal_bcc(), "BCC_A2", ["CR", "FE", "NI"], dependent="FE",
                          mobility_db=jonsson1995_bcc_crfeni())
    # sqrt(D t) is about 100 um at 10 h; a 1.5 mm slab stays semi-infinite
    n = 150 if coarse else 750
    faces = uniform_faces(1.5e-3, n)
    x0 = np.tile(BULK, (n, 1))  # phase.elements is (CR, NI, FE), so the columns are x_Cr, x_Ni
    t_end = HOURS * 3600
    res = solve_planar(faces, x0, lambda x: phase.interdiffusion_matrix(x, T), t_end, t_end / (100 if coarse else 400),
                       left=Boundary.fixed(SURFACE), save_times=[t_end / 4])
    return {"phase": phase, "result": res, "elements": phase.elements}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run()
    res = out["result"]
    z = res.z * 1e6
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))
    for i, style in zip(range(1, len(res.times)), ("--", "-")):
        a1.plot(z, res.x[i, :, 0], "k" + style, label=f"{res.times[i] / 3600:.1f} h")
        a2.plot(z, res.x[i, :, 1], "k" + style)
    a1.set_ylabel("Mole fraction Cr")
    a2.set_ylabel("Mole fraction Ni")
    for a in (a1, a2):
        a.set_xlabel("Depth (um)")
        a.set_xlim(0, 500)
    a1.legend(frameon=False)
    fig.suptitle("Surface Cr Depletion at 1200 C")
    fig.tight_layout()
    fig.savefig(figure_path("surface_depletion.png"), dpi=150)
    lost = (res.inventory()[0, 0] - res.inventory()[-1, 0]) / 1e-6
    print(f"Cr removed through the surface after {HOURS:.0f} h: {lost:.2f} um of pure Cr equivalent")


if __name__ == "__main__":
    main()
