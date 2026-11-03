"""Shared fixtures: a tiny plate problem and a small ladle data set."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"


@pytest.fixture(scope="session")
def tiny_problem():
    from ladle_twin.plate import build_problem
    return build_problem(8, dt=2e-4, n_steps=20)


@pytest.fixture(scope="session")
def small_ladle():
    """A small ladle data set: 40 normal heats + 6 abnormal, seed 0."""
    from ladle_twin.ladle import generate, volume_weights
    g, F, reg, lat = generate(n_single=40, n_double=6, seed=0)
    return dict(g=g, F=F, reg=reg, lat=lat, wg=volume_weights(g))


def notebook_cells(name):
    """Code cells of a notebook as a list of source strings (JSON parse only, no execution)."""
    nb = json.loads((NOTEBOOKS / name).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def notebook_functions(name, wanted, namespace):
    """Execute only the `def`/`class` statements named in `wanted` from a notebook's code cells.

    The statements are picked out of the cell source with `ast`, so no other
    code of the notebook runs (no solves, no plots, no file writes). The
    functions are defined inside `namespace`, which must supply the globals
    they close over (e.g. `np`).
    """
    found = {}
    for src in notebook_cells(name):
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted and node.name not in found:
                mod = ast.Module(body=[node], type_ignores=[])
                exec(compile(mod, f"<{name}:{node.name}>", "exec"), namespace)
                found[node.name] = namespace[node.name]
    missing = set(wanted) - set(found)
    assert not missing, f"not found in {name}: {missing}"
    return found
