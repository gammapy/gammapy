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
    rng = check_random_state(0)

    catalog = make_catalog_random_positions_sphere(size=100, center='Milky Way',
                                                   random_state=rng)
    catalog['Source_Name'] = ['source_{:04d}'.format(_) for _ in range(len(catalog))]
    catalog['Association_Radius'] = Angle(rng.uniform(0, 10, len(catalog)), unit='deg')
    other_catalog = make_catalog_random_positions_sphere(size=100, center='Milky Way',
                                                         random_state=rng)
    other_catalog['Source_Name'] = ['source_{:04d}'.format(_) for _ in range(len(other_catalog))]
    result = catalog_xmatch_circle(catalog, other_catalog)
    assert len(result) == 23


def test_catalog_xmatch_combine():
    # TODO: implement tests
    assert True


def test_table_xmatch():
    # TODO: implement tests
    assert True
