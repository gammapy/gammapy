# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.tests.helper import pytest
from ...image import healpix_to_image

try:
    import healpy
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False


@pytest.mark.skipif('not HAS_HEALPY')
def test_healpix_to_image():
    # TODO: implement me
    pass
