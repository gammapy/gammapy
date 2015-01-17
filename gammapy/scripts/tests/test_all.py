# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
from ...utils.scripts import get_all_main_functions

SCRIPTS = get_all_main_functions()
NAMES = sorted(SCRIPTS.keys())


@pytest.mark.parametrize("name", NAMES)
def test_help(name):
    """Test that --help works for all scripts."""
    # main = SCRIPTS[name].resolve()
    main = SCRIPTS[name]
    with pytest.raises(SystemExit) as exc:
        main(['--help'])

    # TODO: how to assert that it ran OK?
    # Assert exit code or what was printed to sys.stdout?
    # print(exc.value)
    # assert exc.value == SystemExit(0)
