# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ... import __version__
from ...utils.testing import run_cli
from ..main import cli


def test_cli_no_args():
    # No arguments should print help
    result = run_cli(cli, [])
    assert "Usage" in result.output


def test_cli_help():
    result = run_cli(cli, ["--help"])
    assert "Usage" in result.output


def test_cli_version():
    result = run_cli(cli, ["--version"])
    assert "gammapy version {}".format(__version__) in result.output
