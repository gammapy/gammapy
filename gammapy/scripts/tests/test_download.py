# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.scripts.main import cli
from gammapy.utils.testing import run_cli


@pytest.fixture(scope="session")
def config():
    return {
        "release": "0.18",
        "notebook": "astro_dark_matter",
        "envfilename": "gammapy-0.18-environment.yml",
    }


def test_cli_download_help():
    result = run_cli(cli, ["download", "--help"])
    assert "Usage" in result.output


@pytest.mark.remote_data
def test_cli_download_notebooks(tmp_path, config):
    args = [
        "download",
        "notebooks",
        f"--out={tmp_path}",
        f"--release={config['release']}",
    ]
    run_cli(cli, args)
    assert (tmp_path / config["release"] / config["envfilename"]).exists()
    assert (tmp_path / config["release"] / f"{config['notebook']}.ipynb").exists()


@pytest.mark.remote_data
def test_cli_download_datasets(tmp_path, config):
    option_out = f"--out={tmp_path}"

    args = ["download", "datasets", option_out]
    result = run_cli(cli, args)
    assert tmp_path.exists()
    assert "GAMMAPY_DATA" in result.output
