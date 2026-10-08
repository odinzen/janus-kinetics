"""Fe-30Cr / Fe-70Cr (at.%) diffusion couple in bcc at 1200 C, with Kirkendall markers.

Mobilities: Jonsson, ISIJ Int. 35 (1995) 1415, paramagnetic part, valid here
because both alloys are well above their Curie temperatures. Thermodynamics:
ideal solution (see _common.py), so the result shows the mobility side only.
Run ``python examples/fe_cr_couple.py`` to write examples/output/fe_cr_couple.png.
"""

import numpy as np

from janus_kinetics import SolutionPhase, solve_planar, uniform_faces
from janus_kinetics.datasets import jonsson1995_bcc_crfeni

from _common import figure_path, ideal_bcc

T = 1473.15
HOURS = 10.0


def run(coarse: bool = False) -> dict:
    phase = SolutionPhase(ideal_bcc(), "BCC_A2", ["CR", "FE"], dependent="FE", mobility_db=jonsson1995_bcc_crfeni())
    # sqrt(D t) is about 100 um at 10 h, so 1 mm each side keeps the couple semi-infinite
    n = 200 if coarse else 1000
    half = 1e-3
    faces = uniform_faces(2 * half, n, origin=-half)
    z = 0.5 * (faces[1:] + faces[:-1])
    x0 = np.where(z < 0, 0.30, 0.70)[:, None]
    t_end = HOURS * 3600
    res = solve_planar(faces, x0, lambda x: phase.interdiffusion_matrix(x, T), t_end, t_end / (100 if coarse else 400),
                       save_times=[t_end / 4], markers=[0.0],
                       lattice_velocity=lambda x, g: phase.lattice_velocity(x, g, T))
    d_mid = phase.interdiffusion_matrix(np.array([[0.5]]), T)[0, 0, 0]
    return {"phase": phase, "result": res, "z": z, "d_mid": d_mid}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run()
    res, z = out["result"], out["z"]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for i, style in zip(range(len(res.times)), (":", "--", "-")):
        ax.plot(z * 1e6, res.x[i, :, 0], "k" + style, label=f"{res.times[i] / 3600:.0f} h")
        ax.axvline(res.markers[i, 0] * 1e6, color="0.6", lw=0.8, label="Kirkendall markers" if i == 0 else None)
    ax.set_xlabel("Distance (um)")
    ax.set_xlim(-400, 400)
    ax.set_ylabel("Mole fraction Cr")
    ax.set_title("Fe-Cr Couple at 1200 C")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path("fe_cr_couple.png"), dpi=150)
    print(f"interdiffusion coefficient at x_Cr = 0.5: {out['d_mid']:.3e} m^2/s")
    print(f"Kirkendall marker shift after {HOURS:.0f} h: {res.markers[-1, 0] * 1e6:.2f} um")


if __name__ == "__main__":
    main()
