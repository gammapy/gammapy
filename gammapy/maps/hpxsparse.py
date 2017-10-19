# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from .sparse import SparseArray
from .geom import pix_tuple_to_idx
from .hpxmap import HpxMap
from .hpx import ravel_hpx_index

__all__ = [
    'HpxMapSparse',
]


class HpxMapSparse(HpxMap):
    """Representation of a N+2D map using HEALPIX with two spatial
    dimensions and N non-spatial dimensions.

    This class uses a sparse matrix for HEALPix pixel values.

    Parameters
    ----------
    hpx : `~gammapy.maps.hpx.HpxGeom`
        HEALPIX geometry object.
    data : `~numpy.ndarray`
        HEALPIX data array.
    """

    def __init__(self, hpx, data=None, dtype='float32'):

        if data is None:
            shape = tuple([np.max(hpx.npix)] + [ax.nbin for ax in hpx.axes])
            data = SparseArray(shape[::-1])
        elif isinstance(data, np.ndarray):
            data = SparseArray.from_array(data)

        HpxMap.__init__(self, hpx, data)

    def get_by_pix(self, pix, interp=None):

        if interp is None:
            return self.get_by_idx(pix)
        else:
            raise NotImplementedError

    def get_by_idx(self, idx):

        # Convert to local pixel indices
        idx = pix_tuple_to_idx(idx)
        idx = self.hpx.global_to_local(idx)
        return self.data[idx[::-1]]

    def interp_by_coords(self, coords, interp=None):
        raise NotImplementedError

    def fill_by_idx(self, idx, weights=None):

        idx = pix_tuple_to_idx(idx)
        if weights is None:
            weights = np.ones(idx[0].shape)
        idx = self.hpx.global_to_local(idx)
        idx_flat = np.ravel_multi_index(idx, self.data.shape[::-1])
        idx_flat, idx_inv = np.unique(idx_flat, return_inverse=True)
        idx = np.unravel_index(idx_flat, self.data.shape[::-1])
        weights = np.bincount(idx_inv, weights=weights)
        self.data.set(idx[::-1], weights, fill=True)

    def set_by_idx(self, idx, vals):

        idx = pix_tuple_to_idx(idx)
        idx = self.hpx.global_to_local(idx)
        self.data[idx[::-1]] = vals

    def make_cols(self, header, conv):

        from astropy.io import fits
        shape = self.data.shape
        cols = []
        if header['INDXSCHM'] == 'SPARSE':
            idx = np.unravel_index(self.data.idx, shape)
            pix = self.geom.local_to_global(idx[::-1])[0]
            if len(shape) == 1:
                cols.append(fits.Column('PIX', 'J', array=pix))
                cols.append(fits.Column('VALUE', 'E',
                                        array=self.data.data.astype(float)))

            else:
                channel = np.ravel_multi_index(idx[:-1], shape[:-1])
                cols.append(fits.Column('PIX', 'J', array=pix))
                cols.append(fits.Column('CHANNEL', 'I', array=channel))
                cols.append(fits.Column('VALUE', 'E',
                                        array=self.data.data.astype(float)))

        elif len(shape) == 1:
            cols.append(fits.Column(conv.colname(indx=conv.firstcol),
                                    'E', array=self.data.data.astype(float)))
        else:
            for i, idx in enumerate(np.ndindex(shape[:-1])):
                cols.append(fits.Column(conv.colname(indx=i + conv.firstcol), 'E',
                                        array=self.data[idx].astype(float)))
        return cols

    def iter_by_image(self):
        raise NotImplementedError

    def iter_by_pix(self):
        raise NotImplementedError

    def iter_by_coords(self):
        raise NotImplementedError

    def sum_over_axes(self):
        raise NotImplementedError

    def pad(self, pad_width):
        raise NotImplementedError

    def crop(self, crop_width):
        raise NotImplementedError

    def upsample(self, factor):
        raise NotImplementedError

    def downsample(self, factor):
        raise NotImplementedError

    def to_wcs(self, sum_bands=False, normalize=True, proj='AIT', oversample=2):
        raise NotImplementedError

    def to_swapped_scheme(self):
        raise NotImplementedError

    def to_ud_graded(self):
        raise NotImplementedError
