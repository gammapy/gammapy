# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from ...astro.population import make_catalog_random_positions_sphere
from ...catalog import catalog_xmatch_circle


def test_catalog_xmatch_circle():
    random_state = np.random.RandomState(seed=0)

    catalog = make_catalog_random_positions_sphere(size=100, center='Milky Way',
                                                   random_state=random_state)
    catalog['Source_Name'] = ['source_{:04d}'.format(_) for _ in range(len(catalog))]
    catalog['Association_Radius'] = Angle(random_state.uniform(0, 10, len(catalog)), unit='deg')
    other_catalog = make_catalog_random_positions_sphere(size=100, center='Milky Way',
                                                         random_state=random_state)
    other_catalog['Source_Name'] = ['source_{:04d}'.format(_) for _ in range(len(other_catalog))]
    result = catalog_xmatch_circle(catalog, other_catalog)
    assert len(result) == 23
