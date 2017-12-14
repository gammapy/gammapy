# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from ..python_api import run_command


def test_cmd_main(capsys):
    run_command('')
    captured = capsys.readouterr()
    assert "Gammapy is a toolbox for high level analysis" in captured.out


def test_cmd_main_version(capsys):
    run_command('', '--version')
    captured = capsys.readouterr()
    assert "gammapy" in captured.out
