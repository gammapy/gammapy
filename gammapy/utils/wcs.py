# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""WCS related utility functions."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['make_linear_bin_edges_arrays_from_wcs',
           'make_linear_wcs_from_bin_edges_arrays',
           ]

import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import Angle


def make_linear_bin_edges_arrays_from_wcs(w, nbins_x, nbins_y):
    """Make a 2D linear binning from a WCS object.

    This method gives the correct answer only for linear X, Y binning.
    The method expects angular quantities in the WCS object.
    X is identified with WCS axis 1, Y is identified with WCS axis 2.
    The method needs the number of bins as input, since it is not in
    the WCS object.

    Parameters
    ----------
    w : `~astropy.wcs.WCS`
        WCS object describing the bin coordinates
    nbins_x : `~int`
        number of bins in X coordinate
    nbins_y : `~int`
        number of bins in Y coordinate

    Returns
    -------
    bins_x : `~astropy.coordinates.Angle`
        array with the bin edges for the X coordinate
    bins_y : `~astropy.coordinates.Angle`
        array with the bin edges for the Y coordinate
    """
    # check number of dimensions
    if w.wcs.naxis != 2:
        raise ValueError("Expected exactly 2 dimensions, got {}"
                         .format(w.wcs.naxis))

    unit_x, unit_y = w.wcs.cunit
    delta_x, delta_y = w.wcs.cdelt
    delta_x = Angle(delta_x, unit_x)
    delta_y = Angle(delta_y, unit_y)
    bins_x = np.arange(nbins_x + 1)*delta_x
    bins_y = np.arange(nbins_y + 1)*delta_y
    # translate bins to correct values according to WCS reference
    # In FITS, the edge of the image is at pixel coordinate +0.5.
    refpix_x, refpix_y = w.wcs.crpix
    refval_x, refval_y = w.wcs.crval
    refval_x = Angle(refval_x, unit_x)
    refval_y = Angle(refval_y, unit_y)
    bins_x += refval_x - (refpix_x - 0.5)*delta_x
    bins_y += refval_y - (refpix_y - 0.5)*delta_y

    return bins_x, bins_y


def make_linear_wcs_from_bin_edges_arrays(name_x, name_y, bins_x, bins_y):
    """Make a 2D linear WCS object from arrays of bin edges.

    This method gives the correct answer only for linear X, Y binning.
    X is identified with WCS axis 1, Y is identified with WCS axis 2.

    Parameters
    ----------
    name_x : `~string`
        name of X coordinate, to be used as 'CTYPE' value
    name_y : `~string`
        name of Y coordinate, to be used as 'CTYPE' value
    bins_x : `~astropy.coordinates.Angle`
        array with the bin edges for the X coordinate
    bins_y : `~astropy.coordinates.Angle`
        array with the bin edges for the Y coordinate

    Returns
    -------
    w : `~astropy.wcs.WCS`
        WCS object describing the bin coordinates
    """
    # check units
    unit_x = bins_x.unit
    unit_y = bins_y.unit
    if unit_x != unit_y:
        ss_error = "Units of X ({0}) and Y ({1}) bins do not match!".format(
            unit_x, unit_y)
        ss_error += " Is this expected?"
        raise ValueError(ss_error)

    # Create a new WCS object. The number of axes must be set from the start
    w = WCS(naxis=2)

    # Set up DET coordinates in degrees
    nbins_x = len(bins_x) - 1
    nbins_y = len(bins_y) - 1
    range_x = Angle([bins_x[0], bins_x[-1]])
    range_y = Angle([bins_y[0], bins_y[-1]])
    delta_x = (range_x[1] - range_x[0])/nbins_x
    delta_y = (range_y[1] - range_y[0])/nbins_y
    w.wcs.ctype = [name_x, name_y]
    w.wcs.cunit = [unit_x, unit_y]
    w.wcs.cdelt = [delta_x.to(unit_x).value, delta_y.to(unit_y).value]
    # ref as lower left corner (start of (X, Y) bin coordinates)
    # coordinate start empiricaly determined at pix = 0.5: why 0.5?
    w.wcs.crpix = [0.5, 0.5]
    w.wcs.crval = [(bins_x[0] + (w.wcs.crpix[0] - 0.5)*delta_x).to(unit_x).value,
                   (bins_y[0] + (w.wcs.crpix[1] - 0.5)*delta_y).to(unit_y).value]

    return w
