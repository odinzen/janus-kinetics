"""Documentation that is generated from the code must match it."""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gen_api  # noqa: E402


def test_api_reference_is_current():
    assert (ROOT / "docs" / "api.md").read_text(encoding="utf-8") == gen_api.render(), "run python tools/gen_api.py"


def test_every_public_name_is_documented():
    import janus_kinetics
    from janus_kinetics import kawin_bridge

    missing = [n for module in (janus_kinetics, kawin_bridge) for n in module.__all__
               if callable(getattr(module, n)) and not getattr(module, n).__doc__]
    assert not missing, missing


def test_readme_example_runs():
    """The example in README.md is executed as written."""
    import re

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", readme, re.S).group(1)
    scope: dict = {}
    exec(compile(code, "README.md", "exec"), scope)
    assert scope["result"].markers[-1, 0] > 0


def test_validation_page_matches_the_suite():
    """docs/validation.md names only test files that exist, and lists one bullet per skipped test under its three
    headings of skipped tests."""
    import re

    page = (ROOT / "docs" / "validation.md").read_text(encoding="utf-8")
    named = set(re.findall(r"test_\w+\.py", page))
    assert named and all((ROOT / "tests" / n).is_file() for n in named), sorted(named)
    skipped = 0
    for heading in re.findall(r"### [^\n]*\(skipped tests\)\n\n(.*?)(?=\n###|\n## )", page, re.S):
        skipped += len(re.findall(r"^- ", heading, re.M))
    markers = sum(p.read_text(encoding="utf-8").count("@pytest.mark.skip(") for p in (ROOT / "tests").glob("test_*.py")
                  if p.name != "test_docs.py")
    assert skipped == markers == 10
