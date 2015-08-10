# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Field-of-view (FOV) background estimation
"""
from __future__ import print_function, division

from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord

from ..image.utils import coordinates

__all__ = [
    'fill_acceptance_image',
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
    offset : `~astropy.coordinates.Angle`
        1D array of offset values where acceptance is defined.
    acceptance : `~numpy.ndarray`
        Array of acceptance values.

    Returns
    -------
    image : `~astropy.io.fits.ImageHDU`
        Image filled with radial acceptance.
    """
    from scipy.interpolate import interp1d

    # initialize WCS to the header of the image
    wcs = WCS(image.header)

    # define grids of pixel coorinates
    xpix_coord_grid, ypix_coord_grid = coordinates(image, world=False)

    # calculate pixel offset from center (in world coordinates)
    coord = pixel_to_skycoord(xpix_coord_grid, ypix_coord_grid, wcs, origin=0)
    pix_off = coord.separation(center)

    # interpolate
    model = interp1d(offset, acceptance, kind='cubic')
    pix_acc = model(pix_off)

    # fill value in image
    image.data = pix_acc

    return image
