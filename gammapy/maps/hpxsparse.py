# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
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

        from scipy.sparse import csr_matrix

        shape = (1, np.sum(hpx.npix),)

        # TODO : accept sparse matrix for data argument
        if data is None:
            data = csr_matrix(shape, dtype=dtype)
        else:
            data = csr_matrix(np.ravel(data).reshape((1, -1)))

        HpxMap.__init__(self, hpx, data)

    def get_by_pix(self, pix, interp=None):

        if interp is None:
            return get_by_idx(pix)
        else:
            raise NotImplementedError

    def get_by_idx(self, idx):

        # Convert to local pixel indices
        idx = pix_tuple_to_idx(idx)
        idx = self.hpx.global_to_local(idx)
        idx = ravel_hpx_index(idx, self.hpx.npix)
        return np.array(self.data[0, idx])

    def interp_by_coords(self, coords, interp=None):
        raise NotImplementedError

    def fill_by_idx(self, idx, weights=None):

        idx = pix_tuple_to_idx(idx)
        if weights is None:
            weights = np.ones(idx[0].shape)
        idx = self.hpx.global_to_local(idx)
        idx = ravel_hpx_index(idx, self.hpx.npix)
        self.data[0, idx] += weights

    def set_by_idx(self, idx, vals):

        idx = pix_tuple_to_idx(idx)
        idx = self.hpx.global_to_local(idx)
        idx = ravel_hpx_index(idx, self.hpx.npix)
        self.data[0, idx] = vals

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
