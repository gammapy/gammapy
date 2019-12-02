# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.scripts.main import cli
from gammapy.utils.testing import run_cli
from gammapy.analysis import AnalysisConfig


def test_cli_analysis_config(tmp_path):
    path_config = tmp_path / "config.yaml"
    args = ["analysis", "config", f"--filename={path_config}"]
    run_cli(cli, args)
    assert path_config.exists()


def test_cli_analysis_run(tmp_path):
    path_config = tmp_path / "config.yaml"
    config = AnalysisConfig.from_template("1d")
    config.write(path_config)
    path_datasets = tmp_path / "datasets"
    args = ["analysis", "run", f"--filename={path_config}", f"--out={path_datasets}"]
    run_cli(cli, args)
    assert path_datasets.exists()
