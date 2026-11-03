"""Command line of the ladle-generator self-test: ``python -m ladle_twin.cli``.

Reproduces the headline design numbers of the generator (the old
``python notebooks/ladle_v2.py``): cumulative POD energies, the number of modes
for 99 % energy, the condition number of the modes sampled at the QR/DEIM
sensors and the in-distribution gappy-POD error.
"""
from __future__ import annotations

import argparse

from .ladle import design_numbers


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ladle_twin.cli", description=__doc__.split("\n\n")[1])
    p.add_argument("--n-single", type=int, default=500, help="normal-regime heats (default 500)")
    p.add_argument("--n-double", type=int, default=120, help="abnormal double-plume heats (default 120)")
    p.add_argument("--seed", type=int, default=0, help="seed of the operating-point draws (default 0)")
    p.add_argument("--modes", type=int, default=8, help="POD modes kept, K (default 8)")
    p.add_argument("--sensors", type=int, default=10, help="QR/DEIM sensors placed, m (default 10)")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    d = design_numbers(n_single=a.n_single, n_double=a.n_double, seed=a.seed, K=a.modes, m=a.sensors)
    cum = d["cum_energy"]
    print(f"cumE K1={cum[0]:.3f} K6={cum[5]:.3f} K8={cum[7]:.4f} | 99% at K={d['modes_for_99']}")
    print(f"cond(Phi_P)={d['cond']:.1f} | in-dist relL2={d['in_dist_relL2']*100:.1f}%")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
