"""Write docs/api.md from the docstrings of the public API (``calphad_kinetics.__all__``).

    python tools/gen_api.py

tests/test_docs.py fails when the file is out of date, so the reference never
drifts from the code.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import calphad_kinetics

OUT = Path(__file__).parents[1] / "docs" / "api.md"


def render() -> str:
    parts = ["# API reference", "", "Generated from the docstrings by `tools/gen_api.py`; do not edit by hand.", ""]
    by_module: dict[str, list[str]] = {}
    for name in calphad_kinetics.__all__:
        obj = getattr(calphad_kinetics, name)
        module = getattr(obj, "__module__", "calphad_kinetics") or "calphad_kinetics"
        by_module.setdefault(module, []).append(name)
    for module in sorted(by_module):
        parts += [f"## `{module}`", ""]
        for name in sorted(by_module[module]):
            obj = getattr(calphad_kinetics, name)
            if inspect.isclass(obj) or inspect.isfunction(obj):
                try:
                    sig = str(inspect.signature(obj))
                except (TypeError, ValueError):
                    sig = ""
                kind = "class" if inspect.isclass(obj) else "def"
                parts += [f"### `{kind} {name}{sig}`", ""]
                parts += [inspect.cleandoc(obj.__doc__ or "(no docstring)"), ""]
                if inspect.isclass(obj):
                    for meth, member in inspect.getmembers(obj):
                        if meth.startswith("_") or not (inspect.isfunction(member) or inspect.ismethod(member)
                                                          or isinstance(member, property)):
                            continue
                        doc = inspect.getdoc(member)
                        if not doc:
                            continue
                        if inspect.isfunction(member) or inspect.ismethod(member):
                            parts += [f"- `{meth}{inspect.signature(member)}`: {doc.splitlines()[0]}"]
                        else:
                            parts += [f"- `{meth}` (property): {doc.splitlines()[0]}"]
                    parts += [""]
            else:
                parts += [f"### `{name}`", "", f"Constant, value `{obj!r}`.", ""]
    return "\n".join(parts).rstrip() + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT}")
