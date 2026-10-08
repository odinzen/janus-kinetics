"""Oxygen dissolving into a metal film from an atmosphere during a rapid thermal anneal.

A 200 nm film of metal M holds oxygen interstitially, M(1)(O,Va)(1). The surface
is held at a fixed oxygen chemical potential, as an atmosphere of fixed oxygen
activity imposes, while the anneal ramps from 600 K to 900 K, holds and cools.
The surface concentration follows the temperature through the potential, and
oxygen moves in by interstitial jumps into vacant sites (Andersson and Agren
1992, eq. 63).

The database is invented for illustration: an ideal interstitial solution with a
dissolution enthalpy of -100 kJ/mol O and an oxygen mobility with Q = 170 kJ/mol.
Replace it with an assessed description and mobilities for a real film.
Run ``python examples/oxygen_uptake.py`` to write examples/output/oxygen_uptake.png.
"""

import numpy as np
from pycalphad import Database

from janus_kinetics import Boundary, InterstitialPhase, TemperatureSchedule, solve_planar, uniform_faces

from _common import figure_path

SCHEDULE = TemperatureSchedule([0.0, 10.0, 70.0, 80.0], [600.0, 900.0, 900.0, 600.0])
FILM = 200e-9
MU_O = -120000.0  # J/mol, held at the surface

TDB = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT M BLANK 1 0 0 !
ELEMENT O BLANK 1 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE FILM % 2 1 1 !
CONSTITUENT FILM :M:O,VA: !
PARAMETER G(FILM,M:VA;0) 298.15 0; 6000 N !
PARAMETER G(FILM,M:O;0) 298.15 -100000; 6000 N !
PARAMETER MQ(FILM&O,M:*;0) 298.15 -170000+R*T*LN(1E-6); 6000 N !
PARAMETER MQ(FILM&M,M:*;0) 298.15 -400000+R*T*LN(1E-5); 6000 N !
"""


def run(coarse: bool = False) -> dict:
    phase = InterstitialPhase(Database(TDB), "FILM", ["M"], ["O"])
    n = 60 if coarse else 200
    faces = uniform_faces(FILM, n)
    surface = Boundary.potential(phase, {"O": MU_O}, SCHEDULE)
    res = solve_planar(faces, np.full((n, 1), 1e-6), phase.interdiffusion_function(SCHEDULE), 80.0,
                       0.5 if coarse else 0.1, left=surface, save_times=np.linspace(5.0, 75.0, 15))
    return {"result": res, "phase": phase, "surface": surface}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run()
    res = out["result"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    for i, style in zip((4, 8, len(res.times) - 1), (":", "--", "-")):
        a.plot(res.z * 1e9, res.x[i, :, 0], "k" + style, label=f"{res.times[i]:.0f} s, {SCHEDULE(res.times[i]):.0f} K")
    a.set_xlabel("Depth (nm)")
    a.set_ylabel("Oxygen u-fraction (O per M)")
    a.legend(frameon=False)
    b.plot(res.times, res.inventory()[:, 0] * 1e9, "k-")
    b.set_xlabel("Time (s)")
    b.set_ylabel("Oxygen taken up (nm of O per M)")
    b2 = b.twinx()
    b2.plot(res.times, SCHEDULE(res.times), color="0.6", ls="--")
    b2.set_ylabel("Temperature (K)", color="0.4")
    fig.suptitle("Oxygen Uptake into a Metal Film During a Rapid Thermal Anneal")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(figure_path("oxygen_uptake.png"), dpi=150)
    print(f"oxygen taken up: {res.inventory()[-1, 0] * 1e9:.2f} nm-equivalent; "
          f"surface u_O at 900 K {res.x[8, 0, 0]:.4f}, after cooling {res.x[-1, 0, 0]:.4f}")


if __name__ == "__main__":
    main()
