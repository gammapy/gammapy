# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import run_cli
from ..main import cli
from ...extern.pathlib import Path
import pytest


@pytest.fixture(scope='session')
def files_dir(tmpdir_factory):
    filesdir = tmpdir_factory.mktemp('tmpdwn')
    return filesdir


def test_cli_download_help():
    result = run_cli(cli, ["download", "--help"])
    assert "Usage" in result.output


@pytest.mark.xfail
def test_cli_download_datasets(files_dir):
    filename = "data-register.yaml"
    option_dest = "--dest=" + str(files_dir)
    option_file = "--file=" + filename

    args = ["download", option_file, option_dest, "datasets"]
    run_cli(cli, args)

    filepath = Path(str(files_dir)) / 'datasets' / filename
    assert filepath.exists()


@pytest.mark.xfail
def test_cli_download_notebooks(files_dir):
    release = 'master'
    filename = "first_steps.ipynb"
    envfilename = 'environment-' + release + '.yml'
    dirnbsname = 'notebooks-' + release
    option_dest = "--dest=" + str(files_dir)
    option_file = "--file=" + filename
    option_release = "--release=" + release

    args = ["download", option_file, option_dest, option_release, "notebooks"]
    run_cli(cli, args)

    envfilepath = Path(str(files_dir)) / envfilename
    nbfilepath = Path(str(files_dir)) / dirnbsname / filename
    assert envfilepath.exists()
    assert nbfilepath.exists()
