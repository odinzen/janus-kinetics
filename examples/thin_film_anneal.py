"""Growth of a reaction layer between two thin films during a rapid thermal anneal.

A 100 nm A-rich film (alpha) and a 100 nm B-rich film (gamma) sandwich a 2 nm seed
of the intermediate phase beta. The stack is ramped from 600 K to 900 K in 6 s,
held 60 s and cooled in 6 s. Both interfaces stay in local equilibrium, so their
compositions follow the tie-lines as the temperature moves.

The database is invented for illustration: three regular solutions with a wide
intermediate phase, and Arrhenius mobilities (D0 = 1e-5 m^2/s, Q = 220 kJ/mol in
beta and 215 kJ/mol in the terminal films) that put the growth at tens of
nanometres. Replace it with assessed descriptions for a real stack.
Run ``python examples/thin_film_anneal.py`` to write examples/output/thin_film_anneal.png.
"""

import numpy as np
from pycalphad import Database

from janus_kinetics import Layer, LocalEquilibrium, SolutionPhase, TemperatureSchedule, solve_layers
from janus_kinetics.precipitation import solubility_from_pycalphad

from _common import figure_path

SCHEDULE = TemperatureSchedule([0.0, 6.0, 66.0, 72.0], [600.0, 900.0, 900.0, 600.0])
FILM, SEED = 100e-9, 2e-9

TDB = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT A BLANK 1 0 0 !
ELEMENT B BLANK 1 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE ALPHA % 1 1 !
CONSTITUENT ALPHA :A,B: !
PHASE BETA % 1 1 !
CONSTITUENT BETA :A,B: !
PHASE GAMMA % 1 1 !
CONSTITUENT GAMMA :A,B: !
PARAMETER G(ALPHA,A;0) 298.15 0; 6000 N !
PARAMETER G(ALPHA,B;0) 298.15 5000; 6000 N !
PARAMETER G(ALPHA,A,B;0) 298.15 2000; 6000 N !
PARAMETER G(GAMMA,A;0) 298.15 5000; 6000 N !
PARAMETER G(GAMMA,B;0) 298.15 0; 6000 N !
PARAMETER G(GAMMA,A,B;0) 298.15 2000; 6000 N !
PARAMETER G(BETA,A;0) 298.15 6000; 6000 N !
PARAMETER G(BETA,B;0) 298.15 6000; 6000 N !
PARAMETER G(BETA,A,B;0) 298.15 -40000; 6000 N !
""" + "".join(
    f"PARAMETER MQ({phase}&{el},{host};0) 298.15 {-q}+R*T*LN(1E-5); 6000 N !\n"
    for phase, q in (("ALPHA", 215000), ("BETA", 220000), ("GAMMA", 215000))
    for el in "AB" for host in "AB"
)


def run(coarse: bool = False) -> dict:
    db = Database(TDB)
    # B first, so the independent composition is x_B
    alpha, beta, gamma = (SolutionPhase(db, name, ["B", "A"]) for name in ("ALPHA", "BETA", "GAMMA"))
    t0 = SCHEDULE(0.0)
    a_eq, b_lo = solubility_from_pycalphad(db, ["A", "B"], "ALPHA", "BETA", "B", t0, 0.1)
    b_hi, g_eq = solubility_from_pycalphad(db, ["A", "B"], "BETA", "GAMMA", "B", t0, 0.9)
    n_film, n_seed = (30, 12) if coarse else (80, 30)
    layers = [
        Layer(np.full(n_film, a_eq), alpha.interdiffusion_function(SCHEDULE), "alpha film"),
        Layer(np.linspace(b_lo, b_hi, n_seed), beta.interdiffusion_function(SCHEDULE), "beta reaction layer"),
        Layer(np.full(n_film, g_eq), gamma.interdiffusion_function(SCHEDULE), "gamma film"),
    ]
    conditions = [LocalEquilibrium(alpha, beta, SCHEDULE, [a_eq], [b_lo]),
                  LocalEquilibrium(beta, gamma, SCHEDULE, [b_hi], [g_eq])]
    saves = np.linspace(2.0, 70.0, 35)
    res = solve_layers([0.0, FILM, FILM + SEED, 2 * FILM + SEED], layers, conditions, 72.0, 0.4 if coarse else 0.1,
                       save_times=saves)
    return {"result": res, "db": db}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = run()["result"]
    t = res.times
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    a.plot(t, res.thickness()[:, 1] * 1e9, "k-")
    a.set_xlabel("Time (s)")
    a.set_ylabel("Beta layer thickness (nm)")
    a2 = a.twinx()
    a2.plot(t, SCHEDULE(t), color="0.6", ls="--")
    a2.set_ylabel("Temperature (K)", color="0.4")
    z, x = res.profile()
    b.plot(z * 1e9, x[:, 0], "k-")
    for s in res.interfaces[-1]:
        b.axvline(s * 1e9, color="0.6", lw=0.8)
    b.set_xlabel("Depth (nm)")
    b.set_ylabel("Mole fraction B")
    fig.suptitle("Reaction Layer Growth During a Rapid Thermal Anneal")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(figure_path("thin_film_anneal.png"), dpi=150)
    print(f"beta layer: {res.thickness()[0, 1] * 1e9:.1f} nm before, {res.thickness()[-1, 1] * 1e9:.1f} nm after the anneal")


if __name__ == "__main__":
    main()
