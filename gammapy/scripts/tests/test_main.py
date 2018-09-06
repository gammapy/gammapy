# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import run_cli
from ... import version
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
    assert "gammapy version {}".format(version.version) in result.output
