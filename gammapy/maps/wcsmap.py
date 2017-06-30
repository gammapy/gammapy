# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import SkyCoord
from .base import MapBase
from .geom import MapAxis, skydir_to_lonlat
from .wcs import WCSGeom


__all__ = [
    'WcsMap',
]


class WcsMap(MapBase):
    def __init__(self, wcs, data=None):
        MapBase.__init__(self, wcs, data)

    @classmethod
    def create(cls, map_type=None, nxpix=None, nypix=None, binsz=0.1,
               xref=None, yref=None, width=None,
               proj='CAR', coordsys='CEL', xrefpix=None, yrefpix=None,
               axes=None, skydir=None, dtype='float32'):
        """Factory method to create an empty map.

        Parameters
        ----------
        map_type : str
            Internal map representation.  Valid types are `WcsMapND`/`wcs` and
            `WcsMapSparse`/`wcs-sparse`.
        nxpix : int, optional
            Number of pixels in x axis. This option supersedes width.
        nypix : int, optional
            Number of pixels in y axis. This option supersedes width.
        width : float, optional
            Width of the map in degrees.  If None then an all-sky
            geometry will be created.

        """
        from .wcsnd import WcsMapND
        #from .wcssparse import WcsMapSparse

        if width is None:
            width = 360.

        if nxpix is None:
            nxpix = int(np.rint(max(width, 360.) / binsz))

        if nypix is None:
            nypix = int(np.rint(max(width, 180.) / binsz))

        if skydir is None:
            skydir = SkyCoord(0.0, 0.0, unit='deg')
            
        lonlat = skydir_to_lonlat(skydir, coordsys=coordsys)

        wcs = WCSGeom.create(nxpix=nxpix, nypix=nypix, binsz=binsz,
                             xref=lonlat[0], yref=lonlat[1], proj=proj,
                             coordsys=coordsys, xrefpix=xrefpix, yrefpix=yrefpix)

        if map_type in [None, 'wcs', 'WcsMapND']:
            return WcsMapND(wcs)
        elif map_type in ['wcs-sparse', 'WcsMapSparse']:
            raise NotImplementedError
        else:
            raise ValueError('Unregnized Map type: {}'.format(map_type))
