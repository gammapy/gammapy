# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_equal
from astropy.coordinates import Angle
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_dependency
from .. import SkyImage
from .. import radial_profile


@requires_dependency('scipy')
def test_radial_profile():
    center = SkyCoord(0, 0, unit='deg')
    image = SkyImage.empty(nxpix=250, nypix=250, binsz=0.02, xref=center.galactic.l.deg,
                           yref=center.galactic.b.deg, proj='TAN', coordsys='GAL')
    image.data[:] = 1
    #test with the default theta_bin
    table = radial_profile(image, center=center, theta_bin=None)
    offbincenter = table["RADIUS"]
    offbinsize = offbincenter[1] - offbincenter[0]
    excess_profile = table["BIN_VALUE"]
    assert_quantity_allclose(offbinsize, Angle(np.fabs(image.meta["CDELT1"]), image.meta["CUNIT1"]))
    assert_equal(excess_profile, np.ones(len(excess_profile)))

    #Test with a defined theta_bin
    table2 = radial_profile(image, center=center, theta_bin=Angle(0.04, "deg"))
    offbincenter2 = table2["RADIUS"]
    offbinsize2 = offbincenter2[1] - offbincenter2[0]
    assert_quantity_allclose(offbinsize2, Angle(0.04, "deg"))
