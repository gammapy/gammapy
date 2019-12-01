# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.scripts.main import cli
from gammapy.utils.testing import run_cli


def test_cli_analysis(tmp_path):
    path_config = tmp_path / "config.yaml"
    args = [
        "analysis",
        "config",
        f"--filename={path_config}"
    ]
    run_cli(cli, args)
    assert path_config.exists()

    path_datasets = tmp_path / "datasets"
    args = [
        "analysis",
        "run",
        f"--filename={path_config}",
        f"--out={path_datasets}"
    ]
    run_cli(cli, args)
    assert path_datasets.exists()

