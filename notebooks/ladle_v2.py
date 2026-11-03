"""Thin shim: the ladle generator now lives in ``ladle_twin.ladle`` (src/ladle_twin/).

Notebooks 02 and 03 import ``generate``, ``volume_weights``, ``temperature``,
``place_sensors`` and ``gappy_reconstruct`` from this module name; the functions
are unchanged. The shim finds the package in ``../src`` when it is not installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from ladle_twin.ladle import (draw_operating_point, gappy_reconstruct, generate,  # noqa: F401
                                  geom, place_sensors, temperature, volume_weights)
except ModuleNotFoundError:  # running from notebooks/ without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ladle_twin.ladle import (draw_operating_point, gappy_reconstruct, generate,  # noqa: F401
                                  geom, place_sensors, temperature, volume_weights)

__all__ = ["geom", "temperature", "draw_operating_point", "volume_weights",
           "generate", "place_sensors", "gappy_reconstruct"]

if __name__ == "__main__":
    # Self-test: reproduce the headline design numbers (python -m ladle_twin.cli).
    from ladle_twin.cli import main
    raise SystemExit(main())
