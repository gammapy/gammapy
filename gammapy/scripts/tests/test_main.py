# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from click.testing import CliRunner
from ... import version
from ..main import cli


def test_cli_no_args():
    # No arguments should print help
    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 0
    assert 'Usage' in result.output


def test_cli_help():
    result = CliRunner().invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'Usage' in result.output


def test_cli_version():
    result = CliRunner().invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert 'gammapy version {}'.format(version.version) in result.output
