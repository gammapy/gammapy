# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Image utility functions"""
import numpy as np
from astropy.coordinates import Angle
from astropy.wcs.utils import skycoord_to_pixel
from astropy.table import QTable
from gammapy.image import SkyImage

__all__ = [
    'image_label',
    'radial_profile',
]


def image_label(image, theta_bin=None, center=None):
    """
   Calculate the labeled image to know which pixel is group together for the radial profile.

   Parameters
   ----------
   image : `~gammapy.image.SkyImage`
         SkyImage of the image we cant to compute the radial profile
   theta_bin : `~astropy.coordinates.Angle`
         Bining for the radial profile
   center : `~astropy.coordinates.SkyCoord`
         The `SkyCoord` of the pixel used as the center. The default is
            None, which then uses the coordinates of the center of the image.

    Returns
    -------
    index_map : `~numpy.arrays`
        label image
   """

    if not center:
        center = image.center()
    image_bin = Angle(np.fabs(image.meta["CDELT1"]), image.meta["CUNIT1"])
    # Calculate the indices from the image
    pix = image.coordinates_pix()
    x = pix.x
    y = pix.y
    center_pix = skycoord_to_pixel(center, image.wcs, origin=0)
    r = np.hypot(x - center_pix[0], y - center_pix[1])
    if not theta_bin:
        theta_bin = image_bin
    elif theta_bin < image_bin:
        raise ValueError("The binning for the radial profile is lower that the binning of the map")
    theta_pix = theta_bin / image_bin
    label_array = (r / theta_pix).astype(np.int)
    label_image = SkyImage.empty_like(image)
    label_image.data=label_array
    return label_image


def radial_profile(image, theta_bin=None, center=None):
    """
   Calculate the labeled image to know which pixel is group together for the radial profile.

   Parameters
   ----------
   image : `~gammapy.image.SkyImage`
         SkyImage of the image we cant to compute the radial profile
   theta_bin : `~astropy.coordinates.Angle`
         Bining for the radial profile
   center : `~astropy.coordinates.SkyCoord`
         The `SkyCoord` of the pixel used as the center. The default is
            None, which then uses the coordinates of the center of the image.

    Returns
    -------
    table : `~astropy.table.Qtable`
        Three columns: value in each bin, offset bin and err on the radial profil values
   """
    from scipy.ndimage.measurements import labeled_comprehension

    label_image = image_label(image, theta_bin, center).data
    index_tab = np.arange(0, label_image.max() + 1)
    image_bin = Angle(np.fabs(image.meta["CDELT1"]), image.meta["CUNIT1"])
    if not theta_bin:
        theta_bin = image_bin
    theta_bin_tab = index_tab * theta_bin
    values = labeled_comprehension(image.data, label_image, index_tab, np.mean, np.float, -1.)
    myerr = lambda x: np.sqrt(np.fabs(x.astype(float).sum())) / len(x)
    err = labeled_comprehension(image.data, label_image, index_tab, myerr, np.float, -1.)
    table = QTable([theta_bin_tab,
                    values,
                    err],
                   names=('RADIUS', 'BIN_VALUE', 'BIN_ERR'))
    return table
