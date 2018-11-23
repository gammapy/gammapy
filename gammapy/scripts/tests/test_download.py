# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from ...extern.pathlib import Path
from ...utils.testing import requires_dependency, run_cli
from ..main import cli


@pytest.fixture(scope="session")
def files_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("tmpdwn")


def test_cli_download_help():
    result = run_cli(cli, ["download", "--help"])
    assert "Usage" in result.output


@requires_dependency("yaml")
def test_cli_download_datasets(files_dir):
    dataset = "ebl"
    option_out = "--out=" + str(files_dir)
    option_src = "--src=" + dataset

    args = ["download", "datasets", option_src, option_out]
    run_cli(cli, args)

    path = Path(str(files_dir)) / dataset
    assert path.exists()


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

    assert (files_dir / envfilename).exists()
    assert (files_dir / dirnbsname / nbfilename).exists()
