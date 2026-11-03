"""The scripts: the generator self-test CLI and the notebooks' ladle_v2 shim."""
import subprocess
import sys
from pathlib import Path

import numpy as np

from ladle_twin import cli
from ladle_twin import ladle

ROOT = Path(__file__).resolve().parent.parent


def test_parser_builds_with_defaults():
    p = cli.build_parser()
    a = p.parse_args([])
    assert (a.n_single, a.n_double, a.seed, a.modes, a.sensors) == (500, 120, 0, 8, 10)
    a = p.parse_args(["--n-single", "60", "--seed", "2", "--modes", "4", "--sensors", "6"])
    assert (a.n_single, a.seed, a.modes, a.sensors) == (60, 2, 4, 6)


def test_cli_main_runs_small(capsys):
    assert cli.main(["--n-single", "60", "--n-double", "8", "--modes", "6", "--sensors", "8"]) == 0
    out = capsys.readouterr().out
    assert "cumE" in out and "cond(Phi_P)" in out and "relL2" in out


def test_ladle_v2_shim_reexports_the_package_functions():
    sys.path.insert(0, str(ROOT / "notebooks"))
    try:
        import ladle_v2
    finally:
        sys.path.remove(str(ROOT / "notebooks"))
    for name in ("generate", "volume_weights", "temperature", "place_sensors", "gappy_reconstruct"):
        assert getattr(ladle_v2, name) is getattr(ladle, name)


def test_ladle_v2_shim_runs_as_a_script_without_install():
    """`python ladle_v2.py --help` from notebooks/ must find the package in ../src."""
    r = subprocess.run([sys.executable, "-I", "ladle_v2.py", "--help"], cwd=ROOT / "notebooks",
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--sensors" in r.stdout
