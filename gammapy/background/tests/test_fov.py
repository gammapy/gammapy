# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.modeling import models
from ...utils.testing import requires_dependency
from ...background import fill_acceptance_image
from ...image import make_empty_image, SkyMap


@requires_dependency('scipy')
def test_fill_acceptance_image():
    # TODO: the code can be simplified, taking full advantage of the SkyMap class. 
    # create empty image
    # odd number of pixels needed for having the center in its own pixel
    bin_size = Angle(0.1, 'deg')
    image = SkyMap.empty(nxpix=101, nypix=101, binsz=bin_size.degree,
                         xref=0, yref=0, fill=0, proj='CAR', coordsys='GAL',
                         xrefpix=None, yrefpix=None, dtype='float32')
    image = image.to_image_hdu()
    
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
    sigma = Angle(1.0, 'deg')
    amplitude = 1.
    mean = 0.
    stddev = sigma.radian
    gaus_model = models.Gaussian1D(amplitude, mean, stddev)
    acceptance = gaus_model(offset.radian)

    # fill acceptance in the image
    image = fill_acceptance_image(image.header, center, offset, acceptance)

    # test: check points at the offsets where the acceptance is defined
    # along the x axis

    # define grids of pixel coorinates
    xpix_coord_grid, ypix_coord_grid = SkyMap.read(image).coordinates('pix')

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
    acceptance_cut = acceptance[0:n]

    # check acceptance of the image:
    np.testing.assert_almost_equal(image.data_x_axis, acceptance_cut, decimal=4)
