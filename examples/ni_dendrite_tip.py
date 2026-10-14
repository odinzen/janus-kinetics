"""The dendrite tip of a Ni-5.2Al-14.2Cr (at.%) casting, as Kattner, Boettinger and Coriell (Z. Metallkd. 87
(1996) 522) compute it for IN718: tip radius, undercooling and tip liquid against growth speed, with partition
ratios and liquidus slopes from pycalphad's alcrni.tdb (Dupin, Ansara and Sundman 2001).

The inputs come from measurements. The solid-liquid interfacial energy of nickel, 255 erg/cm^2, is Turnbull's
(J. Appl. Phys. 21 (1950) 1022, Table II), and the Gibbs-Thomson coefficient Gamma = sigma V_m T_M / dH_f is
taken with the heat of fusion, melting point and molar volume of the same row (sigma_g = 1860 cal/g-atom,
sigma_g / dH_f = 0.444, sigma_g / T_M = 1.08, and sigma_g = N_A^(1/3) V_m^(2/3) sigma). The liquid diffusivity
is Meyer et al.'s for nickel self-diffusion (Phys. Rev. B 77 (2008) 092201, eq. 3: 77e-9 m^2/s and 0.470 eV),
taken for Al and Cr too, since no measurement in liquid Ni is at hand; with equal diffusivities the tip liquid is
the lever-rule liquid at the tip temperature (Kattner et al., p. 527). Measured interfacial energies of nickel
span 255-347 mJ/m^2 (Hoyt, Asta and Karma, Mater. Sci. Eng. R 41 (2003) 121, Table 2; 347 is Jones' value at the
melting point); the higher one is run for comparison. Run ``python examples/ni_dendrite_tip.py`` to write
examples/output/ni_dendrite_tip.png.
"""

import warnings
from importlib.resources import files

import numpy as np
from pycalphad import Database

from janus_kinetics import SolutionPhase
from janus_kinetics.dendrite_tip import dendrite_tip_calphad, liquidus

from _common import figure_path

ALLOY = [0.052, 0.142]  # Al, Cr; Ni balance
CAL, N_A, K_B_EV = 4.184, 6.02214076e23, 8.617333262e-5
GRADIENTS = (0.0, 1e4)  # K/m in the liquid ahead of the tip


def turnbull_nickel(sigma: float = 0.255) -> dict:
    """Gamma (K m) from Turnbull's Table II row for nickel, and the quantities it is built from."""
    sigma_g = 1860 * CAL  # J per gram-atom
    heat = sigma_g / 0.444  # J/mol
    t_m = 1860 / 1.08  # K, from sigma_g / T0 in cal/(g-atom K)
    v_m = (sigma_g / (0.255 * N_A ** (1 / 3))) ** 1.5  # m^3/mol, from the table's own sigma
    return {"heat": heat, "melting_point": t_m, "molar_volume": v_m, "gamma": sigma * v_m * t_m / heat}


def liquid_diffusivity(T):
    """Meyer et al. (2008), eq. 3."""
    return 77e-9 * np.exp(-0.470 / (K_B_EV * np.asarray(T, dtype=float)))


def run(coarse: bool = False) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(str(files("pycalphad") / "tests" / "databases" / "alcrni.tdb"))
        gamma = SolutionPhase(db, "FCC_A1", ["AL", "CR", "NI"], mobilities=False)
        melt = SolutionPhase(db, "LIQUID", ["AL", "CR", "NI"], mobilities=False)
        top = liquidus(gamma, melt, ALLOY, 1700.0)
        d_l = float(liquid_diffusivity(top.temperature))
        speeds = np.logspace(-5, -1, 5 if coarse else 25)
        runs = {}
        for label, sigma in (("Turnbull", 0.255), ("Jones", 0.347)):
            coefficient = turnbull_nickel(sigma)["gamma"]
            for g in GRADIENTS:
                runs[label, g] = [dendrite_tip_calphad(gamma, melt, v, g, ALLOY, d_l, coefficient, top.temperature)
                                  for v in speeds]
    return {"liquidus": top, "diffusivity": d_l, "speeds": speeds, "runs": runs}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    out = run()
    tl, v = out["liquidus"].temperature, out["speeds"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for (label, g), style in zip(out["runs"], ("-", "--", ":", "-.")):
        tips = out["runs"][label, g]
        a.loglog(v, [tl - t.temperature for t in tips], "k" + style, label=f"{label}, G = {g / 100:g} K/cm")
    a.set_xlabel("Tip velocity (m/s)")
    a.set_ylabel("Tip undercooling below the liquidus (K)")
    a.yaxis.set_major_formatter(ScalarFormatter())
    a.yaxis.set_minor_formatter(NullFormatter())
    a.set_yticks([0.5, 1, 2, 5])
    a.legend(frameon=False, fontsize=8)
    tips = out["runs"]["Turnbull", 0.0]
    b.loglog(v, [1e6 * t.radius for t in tips], "k-", label="radius")
    b.set_xlabel("Tip velocity (m/s)")
    b.set_ylabel("Tip radius (um)")
    twin = b.twinx()
    for j, (name, style) in enumerate((("Al", "--"), ("Cr", ":"))):
        twin.semilogx(v, [t.liquid[j] / ALLOY[j] for t in tips], "k" + style, label=f"{name} in the tip liquid / alloy")
    twin.set_ylabel("Tip liquid over alloy content")
    lines = b.get_lines() + twin.get_lines()
    b.legend(lines, [ln.get_label() for ln in lines], frameon=False, fontsize=8, loc="upper center")
    fig.suptitle("Dendrite Tip of Ni-5.2Al-14.2Cr")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(figure_path("ni_dendrite_tip.png"), dpi=150)
    slow, fast = tips[0], tips[-1]
    print(f"liquidus {tl:.1f} K, liquid D {out['diffusivity']:.2e} m^2/s, Gamma {turnbull_nickel()['gamma']:.2e} K m; "
          f"at {v[0]:g} m/s R = {1e6 * slow.radius:.1f} um, {tl - slow.temperature:.2f} K below the liquidus; "
          f"at {v[-1]:g} m/s R = {1e6 * fast.radius:.2f} um, {tl - fast.temperature:.2f} K below")


if __name__ == "__main__":
    main()
