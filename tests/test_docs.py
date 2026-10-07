"""Documentation that is generated from the code must match it."""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gen_api  # noqa: E402


def test_api_reference_is_current():
    assert (ROOT / "docs" / "api.md").read_text(encoding="utf-8") == gen_api.render(), "run python tools/gen_api.py"


def test_every_public_name_is_documented():
    import calphad_kinetics

    missing = [n for n in calphad_kinetics.__all__
               if callable(getattr(calphad_kinetics, n)) and not getattr(calphad_kinetics, n).__doc__]
    assert not missing, missing


def test_readme_example_runs():
    """The example in README.md is executed as written."""
    import re

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", readme, re.S).group(1)
    scope: dict = {}
    exec(compile(code, "README.md", "exec"), scope)
    assert scope["result"].markers[-1, 0] > 0
