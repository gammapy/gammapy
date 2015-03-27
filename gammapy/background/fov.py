# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Field-of-view (FOV) background estimation
"""
from __future__ import print_function, division
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from scipy.interpolate import interp1d

__all__ = ['fill_acceptance_image',
           ]


def fill_acceptance_image(image, center, offset, acceptance):
    """Generate a 2D image of a radial acceptance curve given as
    an array of values defined at the specified offsets.

    Parameters
    ----------
    image : ImageHDU
        Empty image to fill.
    center : SkyCoord
        Coordinate of the center of the image.
    offset : `~numpy.ndarray`
        Array of offset values in degrees where acceptance is defined.
    acceptance : `~numpy.ndarray`
        Array of acceptance values.

    Returns
    -------
    image : ImageHDU
        Image filled with radial acceptance.
    """
    #initialize WCS to the header of the image
    w = WCS(image.header)
    #loop over pixels and fill acceptance according to offset angle
    nx, ny = image.shape
    for ix in range(0, nx):
        print(ix) #debug
        for iy in range(0, ny):
            print(" ", iy) #debug
            #calculate pixel offset from center (in world coordinates)
            coord = pixel_to_skycoord(ix, iy, w, 1)
            pix_off = coord.separation(center)
            #interpolate
            f = interp1d(offset, acceptance, kind='cubic')
            pix_acc = f(pix_off)
            #fill value in image
            image.data[ix, iy] = pix_acc

    #TODO: method very slow (pixel_to_skycoord + separation in loop)
    #      interpolate makes it even slower
    #      try to avoid explicit loop: I think it can work with arrays directly without the need to loop (maybe it's faster).

    return image
