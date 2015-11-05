# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.tests.helper import pytest
from .. import ExclusionMask, make_empty_image
from ...datasets import gammapy_extra
from ...utils.testing import requires_data
import numpy as np

def test_random_creation():
    hdu = make_empty_image(nxpix=300, nypix=100)
    mask = ExclusionMask.create_random(hdu, n=6, max_rad=10)
    assert mask.mask.shape[0] == 100
    
    excluded = np.where(mask.mask == 0)
    assert excluded[0].size != 0
    
@pytest.mark.xfail 
def test_distance_image():
    pass
    #TODO

