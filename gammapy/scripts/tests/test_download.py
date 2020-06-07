# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.scripts.main import cli
from gammapy.utils.testing import requires_dependency, run_cli


@pytest.fixture(scope="session")
def config():
    return {
        "release": "0.8",
        "dataset": "dark_matter_spectra",
        "notebook": "astro_dark_matter",
        "imagefile": "gammapy_datastore_butler.png",
        "script": "example_2_gauss",
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
        f"--src={config['dataset']}",
        f"--out={tmp_path}",
        f"--release={config['release']}",
    ]
    result = run_cli(cli, args)

    assert (tmp_path / config["dataset"]).exists()
    assert "GAMMAPY_DATA" in result.output


@requires_dependency("parfive")
@pytest.mark.remote_data
def test_cli_download_notebooks(tmp_path, config):
    args = [
        "download",
        "notebooks",
        f"--src={config['notebook']}",
        f"--out={tmp_path}",
        f"--release={config['release']}",
    ]
    run_cli(cli, args)

    assert (tmp_path / config["envfilename"]).exists()
    path = tmp_path / f"notebooks-{config['release']}/{config['notebook']}.ipynb"
    assert path.exists()


@requires_dependency("parfive")
@pytest.mark.remote_data
def test_cli_download_scripts(tmp_path, config):
    args = [
        "download",
        "scripts",
        f"--src={config['script']}",
        f"--out={tmp_path}",
        f"--release={config['release']}",
    ]
    run_cli(cli, args)
    assert (tmp_path / config["envfilename"]).exists()
    assert (tmp_path / f"scripts-{config['release']}/{config['script']}.py").exists()


@requires_dependency("parfive")
@pytest.mark.remote_data
def test_cli_download_tutorials(tmp_path, config):
    option_out = f"--out={tmp_path}"
    nboption_src = f"--src={config['notebook']}"
    scoption_src = f"--src={config['script']}"
    option_release = f"--release={config['release']}"
    dsdirname = "datasets"
    nbdirname = f"notebooks-{config['release']}"
    scdirname = f"scripts-{config['release']}"
    nbfilename = f"{config['notebook']}.ipynb"
    scfilename = f"{config['script']}.py"

    args = ["download", "tutorials", nboption_src, option_out, option_release]
    result = run_cli(cli, args)
    assert (tmp_path / config["envfilename"]).exists()
    assert (tmp_path / nbdirname / nbfilename).exists()
    assert (tmp_path / dsdirname / config["dataset"]).exists()
    assert "GAMMAPY_DATA" in result.output
    assert "jupyter lab" in result.output

    args = ["download", "tutorials", scoption_src, option_out, option_release]
    result = run_cli(cli, args)
    assert (tmp_path / config["envfilename"]).exists()
    assert (tmp_path / scdirname / scfilename).exists()
    assert "GAMMAPY_DATA" in result.output
    assert "jupyter lab" in result.output
