# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import SkyCoord
from .geom import MapCoords, pix_tuple_to_idx, coord_to_idx
from .base import MapBase
from .wcs import WCSGeom
from .wcsmap import WcsMap

__all__ = [
    'WcsMapND',
]


class WcsMapND(WcsMap):
    """Representation of a N+2D map using WCS with two spatial
    dimensions and N non-spatial dimensions.

    Parameters
    ----------
    wcs : `~gammapy.maps.wcs.WCSGeom`
        WCS geometry object.
    data : `~numpy.ndarray`
        Data array. If none then an empty array will be allocated.

    """

    def __init__(self, wcs, data=None):

        # FIXME: Update logic for creating array shape once
        # sparse-support is added to WCSGeom

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
        return cls(wcs)

    @classmethod
    def from_lonlat(cls, lon, lat, cdelt, npix, coordsys='CEL', projection='AIT', axes=None):
        skydir = SkyCoord(lon, lat, unit='deg')
        return cls.from_skydir(skydir, cdelt, npix, coordsys, projection, axes)

    def get_by_pix(self, pix, interp=None):
        if interp is None:
            return self.get_by_idx(pix)
        else:
            raise NotImplementedError

    def get_by_idx(self, idx):
        idx = pix_tuple_to_idx(idx)
        return self._data[idx]

    def interp_by_coords(self, coords, interp=None):
        if interp == 'linear':
            raise NotImplementedError
        else:
            raise ValueError('Invalid interpolation method: {}'.format(interp))

    def fill_by_idx(self, idx, weights=None):
        idx = pix_tuple_to_idx(idx)
        if weights is None:
            weights = np.ones(idx[0].shape)
        self.data.T[idx] += weights

    def set_by_idx(self, idx, vals):
        idx = pix_tuple_to_idx(idx)
        self.data.T[idx] = vals

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
