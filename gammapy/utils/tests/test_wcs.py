# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from ...utils.wcs import (make_linear_bin_edges_arrays_from_wcs,
                          make_linear_wcs_from_bin_edges_arrays)


def test_wcs_object():
    bins_x = Angle([1., 2., 3., 4.], 'degree')
    bins_y = Angle([-1.5, 0., 1.5], 'degree')
    wcs = make_linear_wcs_from_bin_edges_arrays("X", "Y", bins_x, bins_y)
    nbins_x = len(bins_x) - 1
    nbins_y = len(bins_y) - 1
    reco_bins_x, reco_bins_y = make_linear_bin_edges_arrays_from_wcs(wcs,
                                                                     nbins_x,
                                                                     nbins_y)

    # test: reconstructed bins should match original bins
    assert_allclose(reco_bins_x, bins_x)
    assert_allclose(reco_bins_y, bins_y)
