# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from click.testing import CliRunner
from ..main import cli


def test_cli_info_help():
    result = CliRunner().invoke(cli, ['info', '--help'])
    assert result.exit_code == 0
    assert 'Usage' in result.output


def test_cli_info_no_args():
    # No arguments should print all infos
    result = CliRunner().invoke(cli, ['info'])
    assert result.exit_code == 0
    assert "Gammapy current install" in result.output
    assert "Gammapy dependencies" in result.output
    assert "Gammapy environment variables" in result.output
