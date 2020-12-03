# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.scripts.main import cli
from gammapy.utils.testing import requires_dependency, run_cli


@pytest.fixture(scope="session")
def config():
    return {
        "release": "0.8",
        "notebook": "astro_dark_matter",
        "imagefile": "gammapy_datastore_butler.png",
        "envfilename": "gammapy-0.8-environment.yml",
    }


def test_cli_download_help():
    result = run_cli(cli, ["download", "--help"])
    assert "Usage" in result.output


@requires_dependency("parfive")
@pytest.mark.remote_data
def test_cli_download_datasets(tmp_path, config):
    args = [
        "download",
        "datasets",
        f"--out={tmp_path}",
    ]
    result = run_cli(cli, args)
    assert "GAMMAPY_DATA" in result.output


@requires_dependency("parfive")
@pytest.mark.remote_data
def test_cli_download_notebooks(tmp_path, config):
    args = [
        "download",
        "notebooks",
        f"--out={tmp_path}",
        f"--release={config['release']}",
    ]
    run_cli(cli, args)

    assert (tmp_path / config["envfilename"]).exists()
    path = tmp_path / f"notebooks-{config['release']}/{config['notebook']}.ipynb"
    assert path.exists()


@requires_dependency("parfive")
@pytest.mark.remote_data
def test_cli_download_tutorials(tmp_path, config):
    option_out = f"--out={tmp_path}"
    option_release = f"--release={config['release']}"

    args = ["download", "tutorials", option_out, option_release]
    result = run_cli(cli, args)
    assert (tmp_path / config["envfilename"]).exists()
    assert "GAMMAPY_DATA" in result.output
    assert "jupyter lab" in result.output
