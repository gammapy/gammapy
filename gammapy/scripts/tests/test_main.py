# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import sys
from ..python_api import run_command


def test_cmd_main(capsys):
    with pytest.raises(SystemExit):
        run_command('', '--help')
    out, err = capsys.readouterr()
    assert "Gammapy is a toolbox for high level analysis" in out


# on python 2.7 it seems the version info is printed to stderr so we skip the test
@pytest.mark.skipif(sys.version_info < (3,0),
                    reason="requires python 3")
def test_cmd_main_version(capsys):
	with pytest.raises(SystemExit):
	    run_command('', '--version')
	out, err = capsys.readouterr()
	assert "gammapy" in out
