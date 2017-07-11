# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from .geom import pix_tuple_to_idx
from .wcsmap import WcsMap

__all__ = [
    'WcsMapND',
]


class WcsMapND(WcsMap):
    """Representation of a N+2D map using WCS with two spatial dimensions
    and N non-spatial dimensions.  This class uses an ND numpy array
    to store map values.  For maps with non-spatial dimensions and
    variable pixel size it will allocate an array with dimensions
    commensurate with the largest image plane.

    Parameters
    ----------
    wcs : `~gammapy.maps.wcs.WCSGeom`
        WCS geometry object.
    data : `~numpy.ndarray`
        Data array. If none then an empty array will be allocated.
    dtype : str, optional
        Data type, default is float32
    """

    def __init__(self, wcs, data=None, dtype='float32'):

        # TODO: Figure out how to mask pixels for integer data types

        shape = tuple([np.max(wcs.npix[0]), np.max(wcs.npix[1])] +
                      [ax.nbin for ax in wcs.axes])
        if data is None:
            data = np.nan * np.ones(shape, dtype=dtype).T
            pix = wcs.get_pixels()
            data[pix[::-1]] = 0.0
        elif data.shape != shape[::-1]:
            raise ValueError('Wrong shape for input data array. Expected {} '
                             'but got {}'.format(shape, data.shape))

        WcsMap.__init__(self, wcs, data)

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
