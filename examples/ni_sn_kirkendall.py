"""Where the Kirkendall plane sits in a Ni3Sn2 layer grown between Ni3Sn and Ni3Sn4.

Reproduces the analysis of van Loo, Prog. Solid State Chem. 20 (1990) 47, pp. 63-65:
all three phases are taken as line compounds, the layer takes Ni3Sn4 and Ni3Sn in
the volume ratio 125 : 103 whatever diffuses, and the marker split x2/x1 of the
layer (Fig. 2.8) depends only on the ratio of intrinsic diffusivities D_Ni/D_Sn,
with partial molar volumes V_Ni/V_Sn = 0.38 (his Fig. 2.9). The measured
x2/x1 = 1.85 then points to Ni as the faster diffuser, D_Ni/D_Sn = 10.3; van Loo
quotes about 9.5, which gives x2/x1 = 1.84 on this steep part of the curve.
Run ``python examples/ni_sn_kirkendall.py`` to write examples/output/ni_sn_kirkendall.png.
"""

import numpy as np

from janus_kinetics import LineCompound, parabolic_growth

from _common import figure_path

V_NI_OVER_V_SN = 0.38
MEASURED = 1.85


def marker_ratio(d_ni_over_d_sn: float) -> float:
    # Eq. 14 of van Loo with r = 1: D_Ni / D_Sn = (M_Ni / M_Sn)(V_Ni / V_Sn).
    m_sn_over_m_ni = V_NI_OVER_V_SN / d_ni_over_d_sn
    phases = [
        LineCompound("Ni3Sn", 0.25, 51.5 / 4 * 1e-6),
        LineCompound("Ni3Sn2", 0.4, 75 / 5 * 1e-6, integrated_diffusivity=1e-15, tracer_ratio=m_sn_over_m_ni),
        LineCompound("Ni3Sn4", 4 / 7, 125 / 7 * 1e-6),
    ]
    g = parabolic_growth(phases, 1065.0)
    a0, a1 = g.interfaces
    return (a1 - g.marker) / (g.marker - a0)


def run(coarse: bool = False) -> dict:
    ratios = np.geomspace(0.1, 100.0, 25 if coarse else 200)
    return {"d_ratio": ratios, "x2_over_x1": np.array([marker_ratio(r) for r in ratios])}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(out["x2_over_x1"], out["d_ratio"], "k-")
    ax.axvline(MEASURED, color="0.5", ls="--")
    ax.axvline(0.5, color="0.8", ls=":")
    ax.axvline(2.0, color="0.8", ls=":")
    ax.set_xlabel("Marker split $x_2/x_1$")
    ax.set_ylabel(r"$D_\mathrm{Ni}/D_\mathrm{Sn}$")
    ax.set_title("Kirkendall Plane in a Growing Ni$_3$Sn$_2$ Layer")
    fig.tight_layout()
    fig.savefig(figure_path("ni_sn_kirkendall.png"), dpi=150)
    at_measured = np.interp(MEASURED, out["x2_over_x1"], out["d_ratio"])
    print(f"x2/x1 = {MEASURED} gives D_Ni/D_Sn = {at_measured:.1f} for line compounds")


if __name__ == "__main__":
    main()
