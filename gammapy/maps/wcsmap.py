# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from .base import MapBase
from .wcs import WCSGeom

__all__ = [
    'WcsMap',
]


class WcsMap(MapBase):
    def __init__(self, geom, data=None):
        MapBase.__init__(self, geom, data)

    @classmethod
    def create(cls, map_type=None, npix=None, binsz=0.1, width=None,
               proj='CAR', coordsys='CEL', refpix=None,
               axes=None, skydir=None, dtype='float32'):
        """Factory method to create an empty WCS map.

        Parameters
        ----------
        map_type : str
            Internal map representation.  Valid types are `WcsMapND`/`wcs` and
            `WcsMapSparse`/`wcs-sparse`.
        npix : int or tuple or list
            Width of the map in pixels. A tuple will be interpreted as
            parameters for longitude and latitude axes.  For maps with
            non-spatial dimensions, list input can be used to define a
            different map width in each image plane.  This option
            supersedes width.
        width : float or tuple or list
            Width of the map in degrees.  A tuple will be interpreted
            as parameters for longitude and latitude axes.  For maps
            with non-spatial dimensions, list input can be used to
            define a different map width in each image plane.
        binsz : float or tuple or list
            Map pixel size in degrees.  A tuple will be interpreted
            as parameters for longitude and latitude axes.  For maps
            with non-spatial dimensions, list input can be used to
            define a different bin size in each image plane.
        skydir : tuple or `~astropy.coordinates.SkyCoord`
            Sky position of map center.  Can be either a SkyCoord
            object or a tuple of longitude and latitude in deg in the
            coordinate system of the map.
        coordsys : {'CEL', 'GAL'}, optional
            Coordinate system, either Galactic ('GAL') or Equatorial ('CEL').
        axes : list
            List of non-spatial axes.
        proj : string, optional
            Any valid WCS projection type. Default is 'CAR' (cartesian).
        refpix : tuple
            Reference pixel of the projection.  If None then this will
            be chosen to be center of the map.
        dtype : str, optional
            Data type, default is float32

        Returns
        -------
        map : `~WcsMap`
            A WCS map object.
        """
        from .wcsnd import WcsMapND
        # from .wcssparse import WcsMapSparse

        geom = WCSGeom.create(npix=npix, binsz=binsz, width=width,
                              proj=proj, skydir=skydir,
                              coordsys=coordsys, refpix=refpix, axes=axes)

        if map_type in [None, 'wcs', 'WcsMapND']:
            return WcsMapND(geom, dtype=dtype)
        elif map_type in ['wcs-sparse', 'WcsMapSparse']:
            raise NotImplementedError
        else:
            raise ValueError('Unregnized Map type: {}'.format(map_type))
