# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.scripts.main import cli
from gammapy.utils.testing import requires_data, run_cli
from ...analysis.tests.test_analysis import get_example_config


def test_cli_analysis_config(tmp_path):
    path_config = tmp_path / "config.yaml"
    args = ["analysis", "config", f"--filename={path_config}"]
    run_cli(cli, args)
    assert path_config.exists()


@requires_data()
def test_cli_analysis_run(tmp_path):
    
    path_config = tmp_path / "config.yaml"
    path_datasets = tmp_path / "datasets.yaml"
    config = get_example_config("1d")
    config.datasets.background.method = "reflected"
    config.general.datasets_file = str(path_datasets)
    config.general.steps = ["data-reduction"]

    config.write(path_config)
    args = [
        "analysis",
        "run",
        f"--filename={path_config}",
    ]
    run_cli(cli, args)
    assert path_datasets.exists()
