# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.units import Unit
from astropy.coordinates import SkyCoord
from ...irf import EnergyDispersion2D
from ...maps import MapAxis, WcsGeom
from ...cube import EDispMap





def test_edisp_map(tmpdir):
    migra = np.linspace(0.,3.0,100.)
    edisp2d = EnergyDispersion2D.from_gauss(e_true, migra, 0.0, 0.1, offset)

