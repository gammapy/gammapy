# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

from gammapy.background import fill_acceptance_image
from gammapy.image import make_empty_image

from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import astropy.units as u
from astropy.units.quantity import Quantity

import numpy as np
from scipy.integrate import quad

def radial_gaussian_2D(r, sig):
    """Radially symmetric Gaussian function for emulating acceptance curve.

    Please give parameters in radians.

    TODO: I can't define parameters as `~astropy.coordinates.Angle` because it conflicts with the integral call afterwards!!!

    Parameters
    ----------
    r : `~numpy.ndarray`
    Radial coordinate.
    sig : float
    Gaussian width.

    Returns
    -------
    val: `~numpy.ndarray`
    Gaussian evaluated at the requested radii.
    """
    ##norm = 1./(sig * np.sqrt(2. * np.pi))
    ##norm *= 2. #due to definition only for r>=0
    ##norm /= (2.*np.pi) #due to radial symmetry
    norm = 1.
    val = norm * np.exp(-np.power(r, 2.) / (2. * np.power(sig, 2.)))
    return val


def test_fill_acceptance_image():

    #create empty image
    image = make_empty_image()

    #define center coordinate of the image
    #in FITS the reference pixel is given by CRPIXn, and its coordinate by CRVALn
    #the coordinate increment along the axis is given by CDELTn
    x = image.header['CRVAL1']
    y = image.header['CRVAL2']

    #sanity checks: am I using galactic coordinates in degrees?
    assert image.header['CTYPE1'] == 'GLON-CAR', "Error: x coordinate is not in Galactic coordinates!"
    assert image.header['CTYPE2'] == 'GLAT-CAR', "Error: y coordinate is not in Galactic coordinates!"
    assert image.header['CUNIT1'] == 'deg', "Error: x coordinate is not in degrees!"
    assert image.header['CUNIT2'] == 'deg', "Error: y coordinate is not in degrees!"

    center = SkyCoord(l=x*u.degree, b=y*u.degree, frame='galactic')

    #define pixel sizes
    x_pix_size = Angle(abs(image.header['CDELT1'])*u.degree)
    y_pix_size = Angle(abs(image.header['CDELT2'])*u.degree)

    #define radial acceptance and offset angles
    offset = Angle(np.arange(0., 30., 0.1), unit=u.degree)
    acceptance = np.zeros_like(offset)
    sigma = Angle(1.0, unit=u.degree) #gaussian width
    acceptance = radial_gaussian_2D(offset.to(u.radian).value, sigma.to(u.radian).value)

    #fill acceptance in the image
    image = fill_acceptance_image(image, center, offset, acceptance)

    #test: sum of the image (weigthed by the pixel areas) should be equal to the integral of the radial acceptance

    #initialize WCS to the header of the image
    w = WCS(image.header)

    #define grids of pixel coorinates
    nx, ny = image.data.shape
    xpix_coord_grid = np.zeros(image.shape)
    for y in range(0, ny):
        xpix_coord_grid[0:nx, y] = np.arange(0, nx)
    ypix_coord_grid = np.zeros(image.shape)
    for x in range(0, nx):
        ypix_coord_grid[x, 0:nx] = np.arange(0, ny)

    #calculate pixel coordinates (in world coordinates)
    coord = pixel_to_skycoord(xpix_coord_grid, ypix_coord_grid, w, 1)

    #pixel area = delta x * delta y * cos(zenith)
    pix_area_grid = np.cos(coord.l.to(u.radian))*x_pix_size.to(u.radian)*y_pix_size.to(u.radian)

    #sum image, weighted by pixel sizes (i.e. calculate integral of the image)
    image_int_grid = image.data*pix_area_grid
    image_int = image_int_grid.sum()

    #integrate acceptance (i.e. gaussian function)
    #1st integrate in r
    acc_int = Quantity(quad(radial_gaussian_2D, Angle(0.*u.degree).to(u.radian).value,  Angle(10.*u.degree).to(u.radian).value, args=(sigma.to(u.radian).value))*u.radian)
    acc_int_value = acc_int[0]
    #2nd integrate in phi: phi range [0, 2pi)
    acc_int_value *= Angle(2.*np.pi*u.radian)

    #check sum of the image:
    # int ~= sum (bin content * bin size)
    epsilon = 1.e-4
    assert abs(image_int.to(u.rad**2).value - acc_int_value.to(u.rad**2).value) < epsilon, "image integral not compatible with radial acceptance integral"
