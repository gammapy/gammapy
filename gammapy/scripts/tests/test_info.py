# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from ..python_api import run_command


def test_cmd_info(capsys):
    run_command('info', '--all')
    out, err = capsys.readouterr()
    assert "Gammapy current install" in out
    assert "Gammapy dependencies" in out
    assert "Gammapy environment variables" in out
