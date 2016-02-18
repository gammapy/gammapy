# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Field-of-view (FOV) background estimation
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.io import fits
from ..image.utils import coordinates

__all__ = [
    'fill_acceptance_image',
]


def fill_acceptance_image(header, center, offset, acceptance, offset_max=None, interp_kwargs=None):
    """Generate a 2D image of a radial acceptance curve.

    The radial acceptance curve is given as an array of values
    defined at the specified offsets.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Fits header of the reference image
    center : `~astropy.coordinates.SkyCoord`
        Coordinate of the center of the image.
    offset : `~astropy.coordinates.Angle`
        1D array of offset values where acceptance is defined.
    acceptance : `~numpy.ndarray`
        Array of acceptance values.
    interp_kwargs : dict
        option for interpolation for `~scipy.interpolate.interp1d`

    Returns
    -------
    image : `~astropy.io.fits.ImageHDU`
        New image filled with radial acceptance.
    """
    from scipy.interpolate import interp1d
    if not offset_max:
        offset_max = offset[-1]
    if not interp_kwargs:
        interp_kwargs = dict(bounds_error=None, fill_value=acceptance[0])

    # initialize WCS to the header of the image
    wcs = WCS(header)
    data = np.zeros((header["NAXIS2"], header["NAXIS1"]))
    image = fits.ImageHDU(data=data, header=header)

    # define grids of pixel coorinates
    xpix_coord_grid, ypix_coord_grid = coordinates(image, world=False)

    # calculate pixel offset from center (in world coordinates)
    coord = pixel_to_skycoord(xpix_coord_grid, ypix_coord_grid, wcs, origin=0)
    pix_off = coord.separation(center)

    model = interp1d(offset, acceptance, kind='cubic', **interp_kwargs)
    image.data += model(pix_off)
    image.data[pix_off >= offset_max] = 0

    return image
