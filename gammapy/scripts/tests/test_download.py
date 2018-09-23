# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import run_cli
from ..main import cli
from ...extern.pathlib import Path
from ...utils.testing import requires_dependency
import pytest


@pytest.fixture(scope="session")
def files_dir(tmpdir_factory):
    filesdir = tmpdir_factory.mktemp("tmpdwn")
    return filesdir


def test_cli_download_help():
    result = run_cli(cli, ["download", "--help"])
    assert "Usage" in result.output


@pytest.mark.xfail
@requires_dependency("yaml")
def test_cli_download_datasets(files_dir):
    dataset = "ebl"
    option_out = "--out=" + str(files_dir)
    option_src = "--src=" + dataset

    args = ["download", "datasets", option_src, option_out]
    run_cli(cli, args)

    filepath = Path(str(files_dir)) / dataset
    assert filepath.exists()


@pytest.mark.xfail
@requires_dependency("yaml")
def test_cli_download_notebooks(files_dir):
    release = "0.8"
    notebook = "first_steps"
    nbfilename = notebook + ".ipynb"
    envfilename = "gammapy-" + release + "-environment.yml"
    dirnbsname = "notebooks-" + release
    option_out = "--out=" + str(files_dir)
    option_src = "--src=" + notebook
    option_release = "--release=" + release

    args = ["download", "notebooks", option_src, option_out, option_release]
    run_cli(cli, args)

    envfilepath = Path(str(files_dir)) / envfilename
    nbfilepath = Path(str(files_dir)) / dirnbsname / nbfilename
    assert envfilepath.exists()
    assert nbfilepath.exists()
