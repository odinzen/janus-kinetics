"""Write the fcc mobility database of Campbell, Boettinger and Kattner, Acta Mater. 50 (2002) 775.

Table 2 (pp. 780-781) is transcribed below once. Parameters are DICTRA MQ values,
M_i = exp(MQ / RT) / RT (their eq. 4 with Theta_i = 1), with only order-0
interactions (p. 778). Their estimation rule (p. 778) fills the end members that
Table 2 does not list: the activation energy of element i in pure j, for i and j
other than Ni, equals its value in pure Ni. tests/test_campbell2002.py checks the
written file against this script and against numbers the paper prints.

    python tools/make_campbell2002.py
"""

from __future__ import annotations

from pathlib import Path

ELEMENTS = ("AL", "CO", "CR", "HF", "MO", "NI", "RE", "TA", "TI", "W")
OUT = Path(__file__).parents[1] / "src" / "calphad_kinetics" / "data" / "campbell2002_fcc_ni_mob.tdb"

# diffusing element -> ({host: (a, b)}, {(p, j): (a, b)}) with each parameter a + b*T in J/mol
TABLE2 = {
    "AL": ({"AL": (-142000, -72.1), "CR": (-235000, -82), "NI": (-284000, -59.8), "TI": (-284000, -59.8)},
           {("AL", "CR"): (335000, 0), ("AL", "NI"): (-41300, -91.2), ("CR", "NI"): (-53200, 0),
            ("AL", "TI"): (335000, 0), ("NI", "TI"): (-53200, 0)}),
    "CO": ({"CO": (-286175, -76.0), "NI": (-284169, -67.6)},
           {("CO", "NI"): (10787, -11.5)}),
    "CR": ({"AL": (-261700, -3.71), "CO": (-246904, -110), "CR": (-235000, -82), "NI": (-287000, -64.4)},
           {("AL", "CR"): (487000, 0), ("AL", "NI"): (-118000, 0), ("CR", "NI"): (-68000, 0)}),
    "HF": ({"HF": (-235350, -123), "NI": (-251956, -71.2)}, {}),
    "MO": ({"CO": (-256943, -94.2), "MO": (-254975, -81.5), "NI": (-267585, -79.5)}, {}),
    "NI": ({"AL": (-145900, -64.3), "CO": (-270348, -87.3), "CR": (-235000, -82), "HF": (-286182, -84.5),
            "MO": (-286000, -82), "NI": (-287000, -69.8), "RE": (-294401, -105), "TA": (-443736, -96),
            "TI": (-287000, -69.8), "W": (-628250, -63.5)},
           {("AL", "CR"): (-211000, 0), ("AL", "NI"): (-11300, 65.5), ("AL", "TI"): (-211000, 0),
            ("CO", "NI"): (7866, 7.65), ("CR", "NI"): (-81000, 0), ("NI", "RE"): (-157409, 0),
            ("NI", "TA"): (-668454, 0), ("NI", "TI"): (-81000, 0), ("NI", "W"): (175736, 0)}),
    "RE": ({"NI": (-278817, -105), "RE": (-382950, -109.4)}, {("NI", "RE"): (239128, 0)}),
    "TA": ({"NI": (-267729, -79.9), "TA": (-268253, -108.6)}, {("NI", "TA"): (-163380, -15.97)}),
    "TI": ({"AL": (-256900, -77.8), "NI": (-256900, -77.4), "TI": (-256900, -77.8)},
           {("AL", "NI"): (-118000, 0), ("AL", "TI"): (487000, 0), ("NI", "TI"): (-68000, 0)}),
    "W": ({"NI": (-282130, -87.2), "W": (-311423, -70.05)}, {("NI", "W"): (-97025, 0)}),
}


def expression(a: float, b: float) -> str:
    return f"{a:+g}{b:+g}*T".lstrip("+")


def render() -> str:
    lines = [
        "$ fcc atomic mobilities of the Ni-Al-Co-Cr-Hf-Mo-Re-Ta-Ti-W system.",
        "$ Campbell, Boettinger and Kattner, Acta Mater. 50 (2002) 775, Table 2 (pp. 780-781),",
        "$ https://doi.org/10.1016/S1359-6454(01)00383-4. End members that Table 2 does not",
        "$ list take the value in pure Ni (their estimation rule, p. 778); those lines say so.",
        "$ Mobility parameters only; written by tools/make_campbell2002.py.",
        "ELEMENT VA VACUUM 0 0 0 !",
        *[f"ELEMENT {el} BLANK 0 0 0 !" for el in ELEMENTS],
        "TYPE_DEFINITION % SEQ * !",
        "PHASE FCC_A1 % 2 1 1 !",
        f"CONSTITUENT FCC_A1 :{','.join(ELEMENTS)}:VA: !",
    ]
    for el in ELEMENTS:
        hosts, inter = TABLE2[el]
        lines.append(f"$ mobility of {el}")
        for host in ELEMENTS:
            if host in hosts:
                a, b = hosts[host]
                lines.append(f"PARAMETER MQ(FCC_A1&{el},{host}:VA;0) 298.15 {expression(a, b)}; 6000 N !")
            else:
                a, b = hosts["NI"]
                lines.append(f"PARAMETER MQ(FCC_A1&{el},{host}:VA;0) 298.15 {expression(a, b)}; 6000 N ! $ estimated, = in Ni")
        for (p, j), (a, b) in inter.items():
            lines.append(f"PARAMETER MQ(FCC_A1&{el},{p},{j}:VA;0) 298.15 {expression(a, b)}; 6000 N !")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT}")
