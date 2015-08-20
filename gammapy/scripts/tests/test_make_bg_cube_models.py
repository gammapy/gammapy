# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.tests.helper import pytest, remote_data
from ...datasets import get_path
from ..make_bg_cube_models import main as make_bg_cube_models_main
from ...datasets import make_test_dataset

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.parametrize("extra_options,something_to_test", [
    (["--test"], 0),
    ])
@remote_data
def test_make_bg_cube_models_main(extra_options, something_to_test, tmpdir):
    # create a dataset
    fitspath = str(tmpdir.join('test_dataset'))
    outdir = str(tmpdir.join('bg_cube_models'))
    observatory_name = 'HESS'
    scheme = 'HESS'
    n_obs = 2
    random_state = np.random.RandomState(seed=0)

    make_test_dataset(fits_path=fitspath,
                      observatory_name=observatory_name,
                      n_obs=n_obs,
                      random_state=random_state)

    make_bg_cube_models_main([fitspath, scheme, outdir] + extra_options)
