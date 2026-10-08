"""Check every reference cited in the package against Crossref and write docs/references.md.

Each entry holds the DOI and the details as the code cites them (first author,
year, volume, first page). The script fetches the Crossref record, compares those
fields, and writes the reference list from the Crossref metadata itself, so no
entry is typed by hand. Run it before a release; it needs network access.

    python tools/check_references.py
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

# (DOI, first author, year, volume, first page) as cited in the docstrings and docs
REFERENCES = [
    ("10.1063/1.351745", "Andersson", 1992, "72", "1350"),
    ("10.2355/isijinternational.35.1415", "Jonsson", 1995, "35", "1415"),
    ("10.1515/ijmr-1992-830512", "Jonsson", 1992, "83", "349"),
    ("10.1016/0079-6786(90)90007-3", "van Loo", 1990, "20", "47"),
    ("10.1016/0001-6160(57)90127-X", "Hart", 1957, "5", "597"),
    ("10.1063/1.1698258", "Zener", 1949, "20", "950"),
    ("10.1016/j.calphad.2009.06.004", "Larsson", 2009, "33", "495"),
    ("10.1016/j.actamat.2007.12.050", "Perez", 2008, "56", "2119"),
    ("10.1016/j.actamat.2008.11.020", "Perez", 2009, "57", "1318"),
    ("10.1016/S1359-6454(99)00010-5", "Helander", 1999, "47", "1141"),
    ("10.1063/1.1699825", "Fisher", 1951, "22", "74"),
    ("10.5334/jors.140", "Otis", 2017, "5", None),
    ("10.1016/j.msea.2004.06.018", "Svoboda", 2004, "385", "166"),
    ("10.1016/S1359-6454(01)00383-4", "Campbell", 2002, "50", "775"),
    ("10.1016/S0364-5916(01)00049-9", "Dupin", 2001, "25", "279"),
    ("10.1016/S0925-8388(96)02652-7", "Ansara", 1997, "247", "20"),
    ("10.1016/0001-6160(69)90131-X", "Wagner", 1969, "17", "99"),
    ("10.1016/j.actamat.2010.08.032", "Hallstr", 2011, "59", "53"),
    ("10.1016/j.ssi.2013.02.017", "Hallstr", 2013, "240", "41"),
    ("10.1007/BF02645709", "Sundman", 1991, "12", "127"),
    ("10.1016/j.calphad.2008.04.005", "Kjellqvist", 2008, "32", "577"),
]

OUT = Path(__file__).parents[1] / "docs" / "references.md"


def plain(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace(" ", "")


def fetch(doi: str) -> dict:
    req = urllib.request.Request(f"https://api.crossref.org/works/{doi}", headers={"User-Agent": "calphad-kinetics-refcheck"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)["message"]
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Crossref did not answer for {doi}")


def entry(m: dict) -> str:
    authors = ", ".join(f"{a.get('family', '')} {''.join(p[0] + '.' for p in a.get('given', '').replace('-', ' ').split())}".strip()
                        for a in m.get("author", []))
    year = m["issued"]["date-parts"][0][0]
    # house style keeps plain ASCII punctuation, so typographic dashes and quotes are folded
    title = " ".join(m["title"][0].split()).rstrip(".")
    for fancy, plain_char in (("\u2013", "-"), ("\u2014", "-"), ("\u201c", '"'), ("\u201d", '"'), ("\u2019", "'")):
        title = title.replace(fancy, plain_char)
    journal = (m.get("container-title") or [""])[0]
    vol, page = m.get("volume", ""), m.get("page", "")
    return f"{authors} ({year}). {title}. *{journal}* {vol}" + (f", {page}" if page else "") + f". https://doi.org/{m['DOI']}"


def main() -> int:
    problems, lines = [], []
    for doi, author, year, volume, page in REFERENCES:
        m = fetch(doi)
        got_author = plain(m["author"][0]["family"])
        got_year = m["issued"]["date-parts"][0][0]
        got_page = (m.get("page") or "").split("-")[0]
        checks = {
            "first author": plain(author) in got_author or got_author in plain(author),
            "year": got_year == year,
            "volume": str(m.get("volume", "")) == volume,
            "first page": page is None or got_page == page,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            problems.append(f"{doi}: {', '.join(bad)} differ (Crossref: {m['author'][0]['family']}, {got_year}, "
                            f"vol {m.get('volume')}, p {got_page})")
        lines.append(entry(m))
        time.sleep(1)
    OUT.write_text("# References\n\nEvery work cited in calphad-kinetics, written from its Crossref record by "
                   "`tools/check_references.py`.\n\n" + "\n".join(f"- {l}" for l in sorted(lines)) + "\n", encoding="utf-8")
    for p in problems:
        print("MISMATCH", p)
    print(f"{len(REFERENCES)} references checked, {len(problems)} mismatches; wrote {OUT}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
