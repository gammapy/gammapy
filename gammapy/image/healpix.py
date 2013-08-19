"""HEALPIX (Hierarchical Equal-Area and Isolatitude Pixelization) utility functions.

This is a thin wrapper convenience functions around
`healpy` (http://code.google.com/p/healpy/) functionality.
"""
from __future__ import division
from .utils import coordinates


def healpix_to_image(healpix_data, other_image):
    """Convert image in HPX format to some other format (e.g. CAR or AIT).

    @param healpix_data: numpy.ndarray containing the HEALPIX data
    @param other_image: kapteyn.maputils.FITSimage containing the other image"""
    import healpy
    glon, glat = coordinates(other_image, glon_sym=False, radians=True)
    data = healpy.get_interp_val(healpix_data, glon, glat)
    other_image.dat = data


def other_to_healpix(other_image, healpix_data):
    """Convert image in some other format (e.g. CAR or AIT) to HPX

    @param other_image: kapteyn.maputils.FITSimage containing the other image
    @param healpix_data: numpy.ndarray containing the HEALPIX data"""
    raise NotImplementedError
    # Can we use Kapteyn or Healpy to get e.g. bilinear interpolation?
