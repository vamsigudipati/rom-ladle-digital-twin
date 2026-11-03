"""ladle_twin: the pure functions of the three notebooks, extracted unchanged.

Modules
-------
plate    Stage 1: the parametric heat-conduction solver, POD-Galerkin and
         polynomial-ridge reduced-order models.
ladle    Stage 2/3: the analytic gas-stirred-ladle generator (formerly
         notebooks/ladle_v2.py), sensor placement and gappy-POD inverse.
pod      Volume-weighted POD basis, projection and lifting used by Stages 2 and 3.
metrics  Relative errors, the 5 % knee, RMS in kelvin, plant KPIs and their R².
twin     The pieces of the ladle twin: probe layout, sensor readings, physics
         core, residual basis, soft-sensor network, the two guards.
plotting Matplotlib helpers (heat maps, three-panel comparison).
cli      Argument parser and entry point of the generator self-test.
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("ladle-twin")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

__all__ = ["plate", "ladle", "pod", "metrics", "twin", "plotting", "cli", "__version__"]
