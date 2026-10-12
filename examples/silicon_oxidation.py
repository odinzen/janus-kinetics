"""Thermal oxidation of silicon with Deal and Grove's rate constants.

The rate constants are those of Deal and Grove, J. Appl. Phys. 36 (1965) 3770: Table I for wet oxygen (95 C water,
no initial oxide) and Table II for dry oxygen (760 Torr), whose first ~230 A grow faster than the law describes and
are taken as an initial oxide x_i = 230 A, with tau from eq. (12c). The left panel is the thickness against time at
each tabulated temperature; the right one a two-step oxidation, dry oxygen at 1100 C for 30 min followed by wet
oxygen at 1000 C for 2 h, the second step starting from the oxide the first left.
Run ``python examples/silicon_oxidation.py`` to write examples/output/silicon_oxidation.png.
"""

import numpy as np

from janus_kinetics import oxide_thickness

from _common import figure_path

UM, HOUR = 1e-6, 3600.0
WET = {1200: (0.05, 0.720), 1100: (0.11, 0.510), 1000: (0.226, 0.287), 920: (0.50, 0.203)}  # A (um), B (um^2/h)
DRY = {1200: (0.040, 0.045), 1100: (0.090, 0.027), 1000: (0.165, 0.0117), 920: (0.235, 0.0049), 800: (0.370, 0.0011)}
X_I = 0.023  # um, the dry-oxygen initial layer of Table II


def constants(table: dict, temp: int) -> tuple[float, float]:
    a, b = table[temp]
    return a * UM, b * UM**2 / HOUR


def two_step(times: np.ndarray) -> np.ndarray:
    """Dry oxygen at 1100 C for 0.5 h, then wet oxygen at 1000 C; thickness (m) at ``times`` (s)."""
    switch = 0.5 * HOUR
    first = oxide_thickness(np.minimum(times, switch), *constants(DRY, 1100), X_I * UM)
    second = oxide_thickness(np.maximum(times - switch, 0.0), *constants(WET, 1000), first)
    return np.where(times <= switch, first, second)


def run(coarse: bool = False) -> dict:
    t = np.linspace(0.0, 10 * HOUR, 101 if coarse else 1001)
    wet = {temp: oxide_thickness(t, *constants(WET, temp)) for temp in WET}
    dry = {temp: oxide_thickness(t, *constants(DRY, temp), X_I * UM) for temp in DRY}
    t2 = np.linspace(0.0, 2.5 * HOUR, 51 if coarse else 501)
    return {"t": t, "wet": wet, "dry": dry, "t2": t2, "two_step": two_step(t2)}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run()
    t = out["t"] / HOUR
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    for temp, style in zip(sorted(WET, reverse=True), ("-", "--", "-.", ":")):
        a.plot(t, out["wet"][temp] / UM, "k" + style, label=f"wet, {temp} C")
    for temp, style in zip((1200, 1000), ("-", "--")):
        a.plot(t, out["dry"][temp] / UM, color="0.6", ls=style, label=f"dry, {temp} C")
    a.set_xlabel("Oxidation time (h)")
    a.set_ylabel("Oxide thickness (um)")
    a.legend(frameon=False, fontsize=8)
    b.plot(out["t2"] / HOUR, out["two_step"] / UM, "k-")
    b.axvline(0.5, color="0.6", ls=":")
    b.text(0.25, 0.95, "dry\n1100 C", transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    b.text(1.5, 0.95, "wet 1000 C", transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    b.set_xlabel("Time (h)")
    b.set_ylabel("Oxide thickness (um)")
    fig.suptitle("Thermal Oxidation of Silicon by the Linear-Parabolic Law")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(figure_path("silicon_oxidation.png"), dpi=150)
    i1 = np.argmin(np.abs(out["t"] - HOUR))
    print("after 1 h (um): wet", {k: round(v[i1] / UM, 3) for k, v in out["wet"].items()},
          "dry", {k: round(v[i1] / UM, 4) for k, v in out["dry"].items()})
    print(f"two-step: {out['two_step'][np.argmin(np.abs(out['t2'] - 0.5 * HOUR))] / UM:.4f} um after the dry step, "
          f"{out['two_step'][-1] / UM:.4f} um at the end")


if __name__ == "__main__":
    main()
