# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Field-of-view (FOV) background estimation
"""
from __future__ import print_function, division
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import numpy as np
from scipy.interpolate import interp1d

__all__ = ['fill_acceptance_image',
           ]


def fill_acceptance_image(image, center, offset, acceptance):
    """Generate a 2D image of a radial acceptance curve.

    The radial acceptance curve is given as an array of values
    defined at the specified offsets.

    Parameters
    ----------
    image : `~astropy.io.fits.ImageHDU`
        Empty image to fill.
    center : `~astropy.coordinates.SkyCoord`
        Coordinate of the center of the image.
    offset : `~numpy.ndarray` of `~astropy.coordinates.Angle`
        Array of offset values in degrees where acceptance is defined.
    acceptance : `~numpy.ndarray`
        Array of acceptance values.

    Returns
    -------
    image : `~astropy.io.fits.ImageHDU`
        Image filled with radial acceptance.
    """
    #initialize WCS to the header of the image
    w = WCS(image.header)

    #define grids of pixel coorinates
    nx, ny = image.shape
    xpix_coord_grid = np.zeros(image.shape)
    for y in range(0, ny):
        xpix_coord_grid[0:nx, y] = np.arange(0, nx)
    print(xpix_coord_grid) #debug
    ypix_coord_grid = np.zeros(image.shape)
    for x in range(0, nx):
        ypix_coord_grid[x, 0:nx] = np.arange(0, ny)
    print(ypix_coord_grid) #debug

    #calculate pixel offset from center (in world coordinates)
    coord = pixel_to_skycoord(xpix_coord_grid, ypix_coord_grid, w, 1)
    pix_off = coord.separation(center)

    #interpolate
    f = interp1d(offset, acceptance, kind='cubic')
    pix_acc = f(pix_off)

    #fill value in image
    image.data = pix_acc

    return image
