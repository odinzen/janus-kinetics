"""Growth of a Cr2O3 scale on pure Cr in dry O2, after Hallstrom et al., Solid State Ionics 240 (2013) 41.

Cr moves outward through interstitial sites of the corundum lattice and O inward
by vacancies, both mainly along the oxide grain boundaries (their eq. 8 with the
boundary activation energy half the bulk one). The defect fractions come from the
Kjellqvist et al. (2008) Cr-O description shipped with the package, the
mobilities from the paper's eqs. (11)-(12). The paper observed grains of 100-150 nm
at 625 C and 200-250 nm at 700 C; the smaller sizes are used here, for which the
paper's simulations give 340 nm and 660 nm after 24 h.
Run ``python examples/chromia_scale.py`` to write examples/output/chromia_scale.png.
"""

import warnings

import numpy as np

from calphad_kinetics import kjellqvist2008_cro, parabolic_scale_growth
from calphad_kinetics.oxides import chromia

from _common import figure_path

CASES = {"625 C, 100 nm grains": (898.15, 100e-9), "700 C, 200 nm grains": (973.15, 200e-9)}
P_O2 = 1.01325  # bar, dry O2 at 1 atm


def run(coarse: bool = False) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = kjellqvist2008_cro()
        return {label: parabolic_scale_growth(chromia(db, grain_size=grain), T, P_O2, metal_phase="BCC_A2",
                                              nodes=8 if coarse else 16)
                for label, (T, grain) in CASES.items()}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run()
    hours = np.linspace(0.0, 24.0, 200)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for (label, growth), style in zip(out.items(), ("-", "--")):
        ax.plot(hours, growth.thickness(hours * 3600) * 1e9, "k" + style, label=label)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Scale thickness (nm)")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, None)
    ax.legend(frameon=False)
    fig.suptitle("Chromia Scale Growth on Chromium in Dry Oxygen")
    fig.tight_layout()
    fig.savefig(figure_path("chromia_scale.png"), dpi=150)
    for label, growth in out.items():
        print(f"{label}: {growth.thickness(86400.0) * 1e9:.0f} nm after 24 h, k = {growth.k:.3g} m^2/s, "
              f"{100 * growth.inward_fraction:.0f} % grown inward by oxygen")


if __name__ == "__main__":
    main()
