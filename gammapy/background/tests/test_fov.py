# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.tests.helper import pytest
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.modeling import models
from ...background import fill_acceptance_image
from ...image import make_empty_image
from ...image.utils import coordinates

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_fill_acceptance_image():
    # create empty image
    # odd number of pixels needed for having the center in its own pixel
    n_pix_x = 101
    n_pix_y = 101
    bin_size = Angle(0.1, 'degree')
    image = make_empty_image(n_pix_x, n_pix_y, bin_size.degree,
                             xref=0, yref=0, fill=0,
                             proj='CAR', coordsys='GAL',
                             xrefpix=None, yrefpix=None, dtype='float32')

    # define center coordinate of the image in wolrd and pixel coordinates
    lon = image.header['CRVAL1']
    lat = image.header['CRVAL2']

    center = SkyCoord(lon, lat, unit='deg', frame='galactic')

    # initialize WCS to the header of the image
    w = WCS(image.header)

    x_center_pix, y_center_pix = skycoord_to_pixel(center, w, origin=0)

    # define pixel sizes
    # x_pix_size = Angle(abs(image.header['CDELT1']), 'deg')
    # y_pix_size = Angle(abs(image.header['CDELT2']), 'deg')

    # define radial acceptance and offset angles
    # using bin_size for the offset step makes the test comparison easier
    offset = Angle(np.arange(0., 30., bin_size.degree), 'deg')
    # acceptance = np.zeros_like(offset)
    sigma = Angle(1.0, 'degree')  # gaussian width
    amplitude = 1.
    mean = 0.
    stddev = sigma.radian
    gaus_model = models.Gaussian1D(amplitude, mean, stddev)
    acceptance = gaus_model(offset.radian)

    # fill acceptance in the image
    image = fill_acceptance_image(image, center, offset, acceptance)

    # test: check points at the offsets where the acceptance is defined
    # along the x axis

    # define grids of pixel coorinates
    xpix_coord_grid, ypix_coord_grid = coordinates(image, world=False)

    # calculate pixel offset from center (in world coordinates)
    coord = pixel_to_skycoord(xpix_coord_grid, ypix_coord_grid, w, origin=0)
    pix_off = coord.separation(center)

    # x axis defined in the array positions [y_center_pix,:]
    # only interested in semi axis, so [y_center_pix, x_center_pix:]
    ix_min = int(x_center_pix)
    iy = int(y_center_pix)
    pix_off_x_axis = pix_off[iy, ix_min:]
    image.data_x_axis = image.data[iy, ix_min:]

    # cut offset and acceptance arrays to match image size
    # this is only valid if the offset step matches the pixel size
    n = pix_off_x_axis.size
    # offset_cut = offset[0:n]
    acceptance_cut = acceptance[0:n]

    # check acceptance of the image:
    decimal = 4
    s_error = "image acceptance not compatible with defined radial acceptance"
    np.testing.assert_almost_equal(image.data_x_axis, acceptance_cut,
                                   decimal, s_error)
