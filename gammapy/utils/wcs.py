# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""WCS related utility functions."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['linear_arrays_from_wcs',
           'linear_wcs_from_arrays',
           ]

import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import Angle


def linear_arrays_from_wcs(w, nbins_x, nbins_y):
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
    bin_edges_x : `~astropy.coordinates.Angle`
        array with the bin edges for the X coordinate
    bin_edges_y : `~astropy.coordinates.Angle`
        array with the bin edges for the Y coordinate
    """
    # check number of dimensions
    if w.wcs.naxis != 2:
        raise ValueError("Expected exactly 2 dimensions, got {}"
                         .format(w.wcs.naxis))

    # set bins
    unit_x, unit_y = w.wcs.cunit
    delta_x, delta_y = w.wcs.cdelt
    delta_x = Angle(delta_x, unit_x)
    delta_y = Angle(delta_y, unit_y)
    bin_edges_x = np.arange(nbins_x + 1)*delta_x
    bin_edges_y = np.arange(nbins_y + 1)*delta_y
    # translate bins to correct values according to WCS reference
    # In FITS, the edge of the image is at pixel coordinate +0.5.
    refpix_x, refpix_y = w.wcs.crpix
    refval_x, refval_y = w.wcs.crval
    refval_x = Angle(refval_x, unit_x)
    refval_y = Angle(refval_y, unit_y)
    bin_edges_x += refval_x - (refpix_x - 0.5)*delta_x
    bin_edges_y += refval_y - (refpix_y - 0.5)*delta_y

    # set small values (compared to delta (i.e. step)) to 0
    for i in np.arange(len(bin_edges_x)):
        if np.abs(bin_edges_x[i]/delta_x) < 1.e-10:
            bin_edges_x[i] = Angle(0., unit_x)
    for i in np.arange(len(bin_edges_y)):
        if np.abs(bin_edges_y[i]/delta_y) < 1.e-10:
            bin_edges_y[i] = Angle(0., unit_y)

    return bin_edges_x, bin_edges_y


def linear_wcs_from_arrays(name_x, name_y, bin_edges_x, bin_edges_y):
    """Make a 2D linear WCS object from arrays of bin edges.

    This method gives the correct answer only for linear X, Y binning.
    X is identified with WCS axis 1, Y is identified with WCS axis 2.

    Parameters
    ----------
    name_x : `~string`
        name of X coordinate, to be used as 'CTYPE' value
    name_y : `~string`
        name of Y coordinate, to be used as 'CTYPE' value
    bin_edges_x : `~astropy.coordinates.Angle`
        array with the bin edges for the X coordinate
    bin_edges_y : `~astropy.coordinates.Angle`
        array with the bin edges for the Y coordinate

    Returns
    -------
    w : `~astropy.wcs.WCS`
        WCS object describing the bin coordinates
    """
    # check units
    unit_x = bin_edges_x.unit
    unit_y = bin_edges_y.unit
    if unit_x != unit_y:
        ss_error = "Units of X ({0}) and Y ({1}) bins do not match!".format(
            unit_x, unit_y)
        ss_error += " Is this expected?"
        raise ValueError(ss_error)

    # Create a new WCS object. The number of axes must be set from the start
    w = WCS(naxis=2)

    # Set up DET coordinates in degrees
    nbins_x = len(bin_edges_x) - 1
    nbins_y = len(bin_edges_y) - 1
    range_x = Angle([bin_edges_x[0], bin_edges_x[-1]])
    range_y = Angle([bin_edges_y[0], bin_edges_y[-1]])
    delta_x = (range_x[1] - range_x[0])/nbins_x
    delta_y = (range_y[1] - range_y[0])/nbins_y
    w.wcs.ctype = [name_x, name_y]
    w.wcs.cunit = [unit_x, unit_y]
    w.wcs.cdelt = [delta_x.to(unit_x).value, delta_y.to(unit_y).value]
    # ref as lower left corner (start of (X, Y) bin coordinates)
    # coordinate start empiricaly determined at pix = 0.5: why 0.5?
    w.wcs.crpix = [0.5, 0.5]
    w.wcs.crval = [(bin_edges_x[0] + (w.wcs.crpix[0] - 0.5)*delta_x).to(unit_x).value,
                   (bin_edges_y[0] + (w.wcs.crpix[1] - 0.5)*delta_y).to(unit_y).value]

    return w
