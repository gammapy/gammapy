# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import sys
from ..cli import main


def test_cmd_main(capsys):
    with pytest.raises(SystemExit):
        main(['--help'])
    out, err = capsys.readouterr()
    assert "Gammapy is a toolbox for high level analysis" in out


def test_cmd_main_version(capsys):
    with pytest.raises(SystemExit):
        main(['--version'])
    out, err = capsys.readouterr()
    # Looks like on python 2.7 it seems the version info is printed to stderr
    # whereas on Python 3 it's printed to stdout
    # TODO: understand this!
    txt = out if sys.version_info.major >= 3 else err
    assert 'gammapy' in txt
