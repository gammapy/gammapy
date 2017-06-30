# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import SkyCoord
from .base import MapBase
from .wcs import WCSGeom
from .wcsmap import WcsMap

__all__ = [
    'WcsMapND',
]


class WcsMapND(WcsMap):
    def __init__(self, wcs, data=None):

        shape = tuple(list(wcs.npix) + [ax.nbin for ax in wcs.axes])
        if data is None:
            data = np.zeros(shape).T
        elif data.shape != shape[::-1]:
            raise ValueError('Wrong shape for input data array. Expected {} '
                             'but got {}'.format(shape, data.shape))

        WcsMap.__init__(self, wcs, data)

    @classmethod
    def from_skydir(cls, skydir, cdelt, npix, coordsys='CEL', projection='AIT', axes=None):
        wcs = WCSGeom.from_skydir(skydir, cdelt, npix, coordsys, projection,
                                  axes)
        return cls(wcs, np.zeros(npix).T)

    @classmethod
    def from_lonlat(cls, lon, lat, cdelt, npix, coordsys='CEL', projection='AIT', axes=None):
        skydir = SkyCoord(lon, lat, unit='deg')
        return cls.from_skydir(skydir, cdelt, npix, coordsys, projection, axes)

    def get_by_coords(self, coords, interp=None):
        raise NotImplementedError

    def get_by_pix(self, pix, interp=None):
        raise NotImplementedError

    def fill_by_coords(self, coords, weights=None):
        raise NotImplementedError

    def fill_by_pix(self, pix, weights=None):
        raise NotImplementedError

    def set_by_coords(self, coords, vals):
        raise NotImplementedError

    def set_by_pix(self, pix, vals):
        raise NotImplementedError

    def sum_over_axes(self):
        raise NotImplementedError

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection=self.geom.wcs)

        im = ax.imshow(self.data, interpolation='nearest', cmap='magma',
                       origin='lower')
        ax.coords.grid(color='w', linestyle=':', linewidth=0.5)
        return im
