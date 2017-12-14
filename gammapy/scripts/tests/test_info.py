# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from ..python_api import run_command


def test_cmd_info(capsys):
    run_command('info', '--all')
    captured = capsys.readouterr()
    assert "Gammapy current install" in captured.out
    assert "Gammapy dependencies" in captured.out
    assert "Gammapy environment variables" in captured.out
