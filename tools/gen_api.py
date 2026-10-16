"""Write docs/api.md from the docstrings of the public API (``janus_kinetics.__all__``, then the modules
that need an optional dependency).

    python tools/gen_api.py

tests/test_docs.py fails when the file is out of date, so the reference never
drifts from the code.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import janus_kinetics
from janus_kinetics import kawin_bridge

OUT = Path(__file__).parents[1] / "docs" / "api.md"

# kept out of the top-level namespace because they need a package janus-kinetics does not require
OPTIONAL = {kawin_bridge: "needs kawin, `pip install kawin`"}


def render() -> str:
    parts = ["# API reference", "", "Generated from the docstrings by `tools/gen_api.py`; do not edit by hand.", ""]
    by_module: dict[str, list[str]] = {}
    for name in janus_kinetics.__all__:
        obj = getattr(janus_kinetics, name)
        module = getattr(obj, "__module__", "janus_kinetics") or "janus_kinetics"
        by_module.setdefault(module, []).append(name)
    for module in sorted(by_module):
        parts += [f"## `{module}`", ""]
        parts += _entries(janus_kinetics, by_module[module])
    for module, note in OPTIONAL.items():
        parts += [f"## `{module.__name__}` (optional, {note})", ""]
        parts += _entries(module, module.__all__)
    return "\n".join(parts).rstrip() + "\n"


def _entries(namespace, names) -> list[str]:
    parts = []
    for name in sorted(names):
        obj = getattr(namespace, name)
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
    return parts


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT}")
