"""An alpha/beta diffusion couple solved two ways: sharp interface and homogenization.

The database is invented for illustration: two regular solutions A-B-C, with a
two-phase field between them at 1000 K. In a binary at constant temperature no
two-phase zone can grow, so the homogenization model (Larsson and Hoglund,
Calphad 33 (2009) 495) must reproduce the sharp-interface result (local
equilibrium, Zener's flux balance); this example shows that it does.
Run ``python examples/two_phase_couple.py`` to write examples/output/two_phase_couple.png.
"""

import numpy as np
from pycalphad import Database

from janus_kinetics import (
    LocalEquilibrium,
    LocalEquilibriumTable,
    SolutionPhase,
    solve_homogenized,
    solve_moving_boundary,
    uniform_faces,
)

from _common import figure_path

T = 1000.0
TDB = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT A BLANK 1 0 0 !
ELEMENT B BLANK 1 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE ALPHA % 1 1 !
CONSTITUENT ALPHA :A,B: !
PHASE BETA % 1 1 !
CONSTITUENT BETA :A,B: !
PARAMETER G(ALPHA,A;0) 298.15 0; 6000 N !
PARAMETER G(ALPHA,B;0) 298.15 0; 6000 N !
PARAMETER G(BETA,A;0) 298.15 5000; 6000 N !
PARAMETER G(BETA,B;0) 298.15 -5000; 6000 N !
PARAMETER G(ALPHA,A,B;0) 298.15 -2000; 6000 N !
PARAMETER G(BETA,A,B;0) 298.15 1500; 6000 N !
""" + "".join(
    f"PARAMETER MQ({ph}&{el},{host};0) 298.15 {q + base}+R*T*LN({d0}); 6000 N !\n"
    for ph, base in (("ALPHA", 0), ("BETA", -15000))
    for el, d0 in (("A", "1E-4"), ("B", "2E-4"))
    for host, q in (("A", -150000), ("B", -160000))
)


def run(coarse: bool = False) -> dict:
    db = Database(TDB)
    alpha = SolutionPhase(db, "ALPHA", ["A", "B"])
    beta = SolutionPhase(db, "BETA", ["A", "B"])
    n = 400 if coarse else 800  # the homogenized interface converges at first order in the cell size
    half, t_end = 4e-4, 200.0
    cond = LocalEquilibrium(alpha, beta, T, [0.56], [0.27])
    sharp = solve_moving_boundary(2 * half, half, np.full(n // 2, 0.8), np.full(n // 2, 0.1),
                                  lambda x: alpha.interdiffusion_matrix(x, T), lambda x: beta.interdiffusion_matrix(x, T),
                                  cond, t_end, t_end / 200)
    table = LocalEquilibriumTable(db, {"ALPHA": alpha, "BETA": beta}, T)
    faces = uniform_faces(2 * half, n)
    z = 0.5 * (faces[1:] + faces[:-1])
    homog = solve_homogenized(faces, np.where(z < half, 0.8, 0.1)[:, None], table, t_end, t_end / 200)
    thickness = np.sum(table.phase_fractions(homog.x[-1])["ALPHA"] * np.diff(faces))
    return {"sharp": sharp, "homogenized": homog, "table": table, "half": half, "alpha_thickness": thickness}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run()
    sharp, homog, half = out["sharp"], out["homogenized"], out["half"]
    zl = 0.5 * (sharp.z_left[-1, 1:] + sharp.z_left[-1, :-1])
    zr = 0.5 * (sharp.z_right[-1, 1:] + sharp.z_right[-1, :-1])
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(np.concatenate([zl, zr]) * 1e6 - half * 1e6, np.concatenate([sharp.x_left[-1, :, 0], sharp.x_right[-1, :, 0]]),
            "k-", label="sharp interface")
    ax.plot(homog.z * 1e6 - half * 1e6, homog.x[-1, :, 0], "k:", label="homogenization")
    ax.set_xlim(-60, 60)
    ax.set_xlabel("Distance from the weld (um)")
    ax.set_ylabel("Mole fraction A")
    ax.set_title("Alpha/Beta Couple at 1000 K")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path("two_phase_couple.png"), dpi=150)
    print(f"interface shift: sharp {(sharp.interface[-1] - half) * 1e6:.2f} um, "
          f"homogenized alpha layer {(out['alpha_thickness'] - half) * 1e6:.2f} um")


if __name__ == "__main__":
    main()
