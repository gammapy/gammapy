# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Field-of-view (FOV) background estimation."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import Angle
from ..image import SkyImage

__all__ = [
    'fill_acceptance_image',
]


def fill_acceptance_image(header, center, offset, acceptance,
                          offset_max=Angle(2.5, "deg"), offset_min=Angle(0.0, "deg"), interp_kwargs=None):
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
    if offset_max > Angle(offset)[-1]:
        raise ValueError('Offset max used for the acceptance curve ({} deg) is '
                         'inferior to the one you asked to fill the map ({} deg)'
                         ''.format(offset[-1], offset_max.value))
    if not interp_kwargs:
        interp_kwargs = dict(bounds_error=False, fill_value=acceptance[0])

    # initialize WCS to the header of the image
    wcs = WCS(header)
    data = np.zeros((header["NAXIS2"], header["NAXIS1"]))
    image = SkyImage(data=data, wcs=wcs)

    # calculate pixel offset from center (in world coordinates)
    coord = image.coordinates()
    pix_off = coord.separation(center)

    model = interp1d(offset, acceptance, kind='cubic', **interp_kwargs)
    image.data += model(pix_off)
    image.data[pix_off >= offset_max] = 0
    image.data[pix_off <= offset_min] = 0

    # TODO: return SkyImage here and adapt callers.
    return image.to_image_hdu()
