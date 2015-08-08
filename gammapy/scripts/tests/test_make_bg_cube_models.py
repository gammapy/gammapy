# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
from ...datasets import get_path
from ..make_bg_cube_models import main as make_bg_cube_models_main


@pytest.mark.xfail
def test_make_bg_cube_models_main(tmpdir):
    # TODO: implement
    run_list = 'TODO'
    exclusion_list = 'TODO'
    reference_file = 'TODO'
    out_file = str(tmpdir.join('bkg_cube_test.fits'))
    make_bg_cube_models_main([run_list, exclusion_list, reference_file, out_file])
