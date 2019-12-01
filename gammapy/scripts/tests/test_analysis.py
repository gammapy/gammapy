# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.scripts.main import cli
from gammapy.utils.testing import run_cli


def test_cli_analysis(tmp_path):
    path = tmp_path / "config.yaml"
    args = [
        "analysis",
        "config",
        f"--filename={path}"
    ]
    run_cli(cli, args)
    assert path.exists()

    args = [
        "analysis",
        "run",
        f"--filename={path}"
    ]
    result = run_cli(cli, args)
    assert "Data reduction process finished." in result.output

