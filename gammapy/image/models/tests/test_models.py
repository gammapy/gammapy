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
            lon_mean=359 * u.deg,
            lat_mean=88 * u.deg,
            sigma=1 * u.deg,
        ),
        test_val=0.0964148382898712 / u.deg ** 2,
    )
]


@pytest.mark.parametrize(
    "spatial", TEST_MODELS, ids=[_['name'] for _ in TEST_MODELS]
)
def test_models(spatial):
    model = spatial['model']
    lon = 1 * u.deg
    lat = 89 * u.deg
    value = model(lon, lat)
    assert_quantity_allclose(value, spatial['test_val'])
