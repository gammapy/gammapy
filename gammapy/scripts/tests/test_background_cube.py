# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
from ..background_cube import main as background_cube_main


@pytest.mark.xfail
def test_background_cube_main(tmpdir):
    # TODO: implement
    run_list = 'TODO'
    exclusion_list = 'TODO'
    reference_file = 'TODO'
    filename = str(tmpdir.join('bkg_cube_test.fits'))
    background_cube_main([run_list, exclusion_list, reference_file, filename])
