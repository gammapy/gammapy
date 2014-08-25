"""HEALPIX (Hierarchical Equal-Area and Isolatitude Pixelization) utility functions.

This is a thin wrapper convenience functions around
`healpy` (http://code.google.com/p/healpy/) functionality.

Really these utility functions belong in `healpy` ... I've made a feature request here:
https://github.com/healpy/healpy/issues/129
"""
from __future__ import print_function, division
from astropy.coordinates import SkyCoord
from ..image import coordinates

__all__ = ['healpix_to_image', 'image_to_healpix']

__doctest_skip__ = ['healpix_to_image']


def healpix_to_image(healpix_data, reference_image, hpx_coord_system):
    """Convert image in HEALPIX format to a normal FITS projection image (e.g. CAR or AIT).

    Parameters
    ----------
    healpix_data : `numpy.ndarray`
        HEALPIX data array
    reference_image : `~astropy.io.fits.ImageHDU`
        A reference image to project to.  Must have a 'COORDSYS' keyword of
        either 'galactic' or 'icrs'
    hpx_coord_system : 'galactic' or 'icrs'
        The target coordinate system.  Should be derived from the HEALPIX
        COORDSYS keyword if it is a FITS file

    Returns
    -------
    reprojected_data : `numpy.ndarray`
        HEALPIX image resampled onto the reference image

    Examples
    --------
    >>> import healpy as hp
    >>> from astropy.io import fits
    >>> from gammapy.image.healpix import healpix_to_image
    >>> healpix_data = hp.read_map('healpix.fits')
    >>> healpix_system = fits.getheader('healpix.fits')['COORDSYS']
    >>> reference_image = fits.open('reference_image.fits')[0]
    >>> reprojected_data = healpix_to_image(healpix_data, reference_image, healpix_system)
    >>> fits.writeto('new_image.fits', reprojected_data, reference_image.header)
    """
    import healpy as hp
    lon, lat = coordinates(reference_image, lon_sym=False, radians=True)

    # If the reference image uses a different celestial coordinate system from
    # the HEALPIX image we need to transform the coordinates
    ref_coord_system = reference_image.header['COORDSYS']
    if ref_coord_system != hpx_coord_system:
        coordinate = SkyCoord(lon, lat, frame=ref_coord_system)
        coordinate = coordinate.transform_to(hpx_coord_system)
        lon, lat = coordinate.data.lon, coordinate.data.lat

    data = hp.get_interp_val(healpix_data, lon, lat)
    return data


def image_to_healpix(image, healpix_pars):
    """Convert image in a normal FITS projection (e.g. CAR or AIT) to HEALPIX format.

    Parameters
    ----------
    image : `~astropy.io.fits.ImageHDU`
        The input image
    healpix_pars : TODO
        TODO: what HEALPIX parameters do we need?
    Returns
    -------
    healpix_data : `numpy.array`
        HEALPIX array data
    """
    raise NotImplementedError
    # Can we use Kapteyn or Healpy to get e.g. bilinear interpolation?
