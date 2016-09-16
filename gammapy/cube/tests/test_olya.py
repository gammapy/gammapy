# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from numpy.testing import assert_allclose
from astropy import units as u

from gammapy.cube import SkyCube
from ...datasets import FermiGalacticCenter as F

from ...datasets.core import gammapy_extra


# def test_olya():
#     # skycube = F.diffuse_model()
#     print ('sdfsdfgdfgsfdg    ', type(skycube))
#     lon1, lat1 = skycube.spatial_coordinate_images
#     # lon2, lat2 = skycube.olya_spatial_coordinate_images()
#
#     assert_allclose(lon1, lon2)
#     assert_allclose(lat1, lat2)


def test_olya():
    filename = gammapy_extra.dir / 'datasets/vela_region/gll_iem_v05_rev1_cutout.fits'
    print("fn",filename, os.path.exists(str(filename)))
    skycube = SkyCube.read(filename=str(filename))

    lon = Angle(-90, unit=u.deg)
    lat = Angle(0, unit=u.deg)
    energy = 42.0 * u.MeV

    for ind in xrange(-10, 20):
        print ('Lon = ', lon)
        print ('Lat = ', lat)
        print ('Result')
        print (skycube.spectral_index(lon, lat, energy * (2 ** ind)))