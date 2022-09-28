# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.scripts.main import cli
from gammapy.utils.testing import requires_dependency, run_cli


@pytest.fixture(scope="session")
def config():
    return {
        "release": "0.20",
        "notebook": "overview",
        "envfilename": "gammapy-0.20-environment.yml",
    }


def test_cli_download_help():
    result = run_cli(cli, ["download", "--help"])
    assert "Usage" in result.output


@requires_dependency("requests")
@requires_dependency("tqdm")
@pytest.mark.remote_data
def test_cli_download_notebooks_stable(tmp_path, config):
    args = [
        "download",
        "notebooks",
        f"--out={tmp_path}",
        f"--release={config['release']}",
    ]
    run_cli(cli, args)
    assert (tmp_path / config["release"] / config["envfilename"]).exists()
    assert (
        tmp_path
        / config["release"]
        / "tutorials"
        / "starting"
        / f"{config['notebook']}.ipynb"
    ).exists()


@requires_dependency("requests")
@requires_dependency("tqdm")
@pytest.mark.remote_data
def test_cli_download_notebooks_dev(tmp_path):
    args = [
        "download",
        "notebooks",
        f"--out={tmp_path}",
        "--release=dev",
    ]
    run_cli(cli, args)
    assert (tmp_path / "dev" / "gammapy-dev-environment.yml").exists()
    assert (tmp_path / "dev" / "starting" / "analysis_1.ipynb").exists()


@requires_dependency("requests")
@requires_dependency("tqdm")
@pytest.mark.remote_data
def test_cli_download_datasets(tmp_path):
    # TODO: this test downloads all datasets which is really slow...
    option_out = f"--out={tmp_path}"

    args = ["download", "datasets", option_out]
    result = run_cli(cli, args)
    assert (tmp_path / "dev").exists()
    assert "GAMMAPY_DATA" in result.output
