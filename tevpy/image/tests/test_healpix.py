# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from .. import healpix

try:
    import healpy
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

@pytest.mark.skipif('not HAS_HEALPY')
def test_healpix_to_image():
    # TODO: implement me
    pass
