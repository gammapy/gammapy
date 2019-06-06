# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import pytest
from ...utils.testing import run_cli
from ..main import cli


@pytest.fixture(scope="session")
def files_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("tmpdwn"))


@pytest.fixture(scope="session")
def config():
    return dict(
        release="0.8",
        dataset="dark_matter_spectra",
        notebook="astro_dark_matter",
        imagefile="gammapy_datastore_butler.png",
        script="example_2_gauss",
        envfilename="gammapy-0.8-environment.yml",
    )


def test_cli_download_help():
    result = run_cli(cli, ["download", "--help"])
    assert "Usage" in result.output


@pytest.mark.remote_data
def test_cli_download_datasets(files_dir, config):
    option_out = "--out={}".format(files_dir)
    option_src = "--src={}".format(config["dataset"])
    option_release = "--release={}".format(config["release"])

    args = ["download", "datasets", option_src, option_out, option_release]
    result = run_cli(cli, args)
    assert (Path(files_dir) / config["dataset"]).exists()
    assert "GAMMAPY_DATA" in result.output


@pytest.mark.remote_data
def test_cli_download_notebooks(files_dir, config):
    option_out = "--out={}".format(files_dir)
    option_src = "--src={}".format(config["notebook"])
    option_release = "--release={}".format(config["release"])
    filename = "{}.ipynb".format(config["notebook"])
    dirname = "notebooks-{}".format(config["release"])

    args = ["download", "notebooks", option_src, option_out, option_release]
    run_cli(cli, args)
    assert (Path(files_dir) / config["envfilename"]).exists()
    # assert (Path(files_dir) / dirname / "images" / config["imagefile"]).exists()
    assert (Path(files_dir) / dirname / filename).exists()


@pytest.mark.remote_data
def test_cli_download_scripts(files_dir, config):
    option_out = "--out={}".format(files_dir)
    option_src = "--src={}".format(config["script"])
    option_release = "--release={}".format(config["release"])
    filename = "{}.py".format(config["script"])
    dirname = "scripts-{}".format(config["release"])

    args = ["download", "scripts", option_src, option_out, option_release]
    run_cli(cli, args)
    assert (Path(files_dir) / config["envfilename"]).exists()
    assert (Path(files_dir) / dirname / filename).exists()


@pytest.mark.remote_data
def test_cli_download_tutorials(files_dir, config):
    option_out = "--out={}".format(files_dir)
    nboption_src = "--src={}".format(config["notebook"])
    scoption_src = "--src={}".format(config["script"])
    option_release = "--release={}".format(config["release"])
    dsdirname = "datasets"
    nbdirname = "notebooks-{}".format(config["release"])
    scdirname = "scripts-{}".format(config["release"])
    nbfilename = "{}.ipynb".format(config["notebook"])
    scfilename = "{}.py".format(config["script"])

    args = ["download", "tutorials", nboption_src, option_out, option_release]
    result = run_cli(cli, args)
    assert (Path(files_dir) / config["envfilename"]).exists()
    assert (Path(files_dir) / nbdirname / nbfilename).exists()
    # assert (Path(files_dir) / nbdirname / "images" / config["imagefile"]).exists()
    assert (Path(files_dir) / dsdirname / config["dataset"]).exists()
    assert "GAMMAPY_DATA" in result.output
    assert "jupyter lab" in result.output

    args = ["download", "tutorials", scoption_src, option_out, option_release]
    result = run_cli(cli, args)
    assert (Path(files_dir) / config["envfilename"]).exists()
    assert (Path(files_dir) / scdirname / scfilename).exists()
    assert "GAMMAPY_DATA" in result.output
    assert "jupyter lab" in result.output
