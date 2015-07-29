# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.coordinates import Angle
from ...astro.population import make_catalog_random_positions_sphere
from ...catalog import (
    catalog_xmatch_circle,
    catalog_xmatch_combine,
    table_xmatch_circle_criterion,
    table_xmatch,
)
from ...utils.random import check_random_state


def test_catalog_xmatch_circle():
    # initialise random number generator
    np.random.seed(0)
    #rng = check_random_state(0)

    catalog = make_catalog_random_positions_sphere(size=100, center='Milky Way')
    catalog['Source_Name'] = ['source_{:04d}'.format(_) for _ in range(len(catalog))]
    catalog['Association_Radius'] = Angle(10 * np.random.random(len(catalog)), unit='deg')
    #catalog['Association_Radius'] = Angle(rng.uniform(0, 10, len(catalog)), unit='deg')
    # TODO: not working with rng!!! (but why did it work before??!!!)
    other_catalog = make_catalog_random_positions_sphere(size=100, center='Milky Way')
    other_catalog['Source_Name'] = ['source_{:04d}'.format(_) for _ in range(len(other_catalog))]
    result = catalog_xmatch_circle(catalog, other_catalog)
    assert len(result) == 23


def test_catalog_xmatch_combine():
    # TODO: implement tests
    assert True


def test_table_xmatch():
    # TODO: implement tests
    assert True
