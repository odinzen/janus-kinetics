"""Version and authorship agree across the package, CITATION.cff and .zenodo.json."""

import json
import re
from pathlib import Path

import janus_kinetics

ROOT = Path(__file__).parents[1]


def test_versions_agree():
    pyproject = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    cff = re.search(r"^version: (\S+)", (ROOT / "CITATION.cff").read_text(), re.M).group(1)
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))["version"]
    assert pyproject == cff == zenodo == janus_kinetics.__version__


def test_author_and_license_agree():
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    cff = (ROOT / "CITATION.cff").read_text()
    assert zenodo["creators"][0]["orcid"] in cff
    assert zenodo["license"] == "MIT" and "license: MIT" in cff
    assert "Copyright (c) 2026 Odinzen LLC" in (ROOT / "LICENSE").read_text()
