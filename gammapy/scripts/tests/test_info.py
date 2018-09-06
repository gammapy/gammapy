# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import run_cli
from ..main import cli


def test_cli_info_help():
    result = run_cli(cli, ["info", "--help"])
    assert "Usage" in result.output


def test_cli_info_no_args():
    # No arguments should print all infos
    result = run_cli(cli, ["info"])
    assert "System" in result.output
    assert "Gammapy package" in result.output
    assert "Other packages" in result.output
    assert "Gammapy environment variables" in result.output
