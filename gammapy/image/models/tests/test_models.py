# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from ..new import SkyGaussian2D
from astropy.tests.helper import assert_quantity_allclose
import pytest
import astropy.units as u

TEST_MODELS = [
    dict(
        name='skygaussian2d',
        model=SkyGaussian2D(
            amplitude=1,
            lon_mean=0 * u.deg,
            lat_mean=0 * u.deg,
            sigma=1 * u.deg,
        ),
        test_val=0.36787944117144233,
    )
]


@pytest.mark.parametrize(
    "spatial", TEST_MODELS, ids=[_['name'] for _ in TEST_MODELS]
)
def test_models(spatial):
    model = spatial['model']
    lon = 1 * u.deg
    lat = 1 * u.deg
    value = model(lon, lat)
    assert_quantity_allclose(value, spatial['test_val'])
