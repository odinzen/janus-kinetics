"""A metal film reacting with its substrate into stoichiometric compounds during an anneal.

A 50 nm film of metal M on a thick substrate S forms two line compounds, M2S next
to the metal and MS next to the substrate, as in many silicide and intermetallic
film reactions. Both grow from the start of the anneal, the film is used up, then
M2S is turned into MS. The anneal ramps from 500 K to 750 K in 10 s and holds.

The data are invented for illustration: Gibbs energies of -40 and -45 kJ per mole
of atoms, Arrhenius integrated diffusivities with Q = 160 kJ/mol (M2S) and
170 kJ/mol (MS), and molar volumes near those of transition-metal silicides.
Interface reactions and nucleation barriers, which select the first phase in
many real films, are not modelled.
Run ``python examples/film_reaction.py`` to write examples/output/film_reaction.png.
"""

import numpy as np

from janus_kinetics import LineCompound, TemperatureSchedule, solve_reaction_layers

from _common import figure_path

SCHEDULE = TemperatureSchedule([0.0, 10.0], [500.0, 750.0])
FILM = 50e-9
T_END = 3000.0


def arrhenius(d0: float, q: float):
    return lambda T: d0 * np.exp(-q / (8.314 * T))


PHASES = [
    LineCompound("M", 0.0, 6.6e-6, gibbs=0.0),
    LineCompound("M2S", 1 / 3, 8.0e-6, gibbs=-40000.0, integrated_diffusivity=arrhenius(1e-6, 160e3),
                 tracer_ratio=0.1),
    LineCompound("MS", 0.5, 9.0e-6, gibbs=-45000.0, integrated_diffusivity=arrhenius(5e-7, 170e3),
                 tracer_ratio=0.1),
    LineCompound("S", 1.0, 12.0e-6, gibbs=0.0),
]


def run(coarse: bool = False) -> dict:
    times = np.linspace(2.0, T_END, 60 if coarse else 600)
    res = solve_reaction_layers(PHASES, T_END, SCHEDULE, end_thickness=(FILM, None), save_times=times)
    return {"result": res}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = run()["result"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, label, style in (("M", "M", ":"), ("M2S", "M$_2$S", "--"), ("MS", "MS", "-")):
        ax.plot(res.times, res.thickness[name] * 1e9, "k" + style, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thickness (nm)")
    ax.legend(frameon=False)
    ax.set_title("Film Reaction into Two Line Compounds")
    fig.tight_layout()
    fig.savefig(figure_path("film_reaction.png"), dpi=150)
    for t, name in res.consumed:
        print(f"{name} used up at {t:.1f} s ({SCHEDULE(t):.0f} K)")
    print(f"final MS thickness {res.thickness['MS'][-1] * 1e9:.1f} nm")


if __name__ == "__main__":
    main()
