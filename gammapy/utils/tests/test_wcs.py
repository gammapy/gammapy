# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.coordinates import Angle
from ...utils.testing import assert_quantity_allclose
from ...utils.wcs import linear_wcs_to_arrays, linear_arrays_to_wcs


def test_wcs_object():
    bins_x = Angle([1., 2., 3., 4.], "deg")
    bins_y = Angle([-1.5, 0., 1.5], "deg")
    wcs = linear_arrays_to_wcs("X", "Y", bins_x, bins_y)
    nbins_x = len(bins_x) - 1
    nbins_y = len(bins_y) - 1
    reco_bins_x, reco_bins_y = linear_wcs_to_arrays(wcs, nbins_x, nbins_y)

    # test: reconstructed bins should match original bins
    assert_quantity_allclose(reco_bins_x, bins_x)
    assert_quantity_allclose(reco_bins_y, bins_y)
