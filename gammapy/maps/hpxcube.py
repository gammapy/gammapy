# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from .geom import MapCoords, pix_tuple_to_idx, coord_to_idx
from .hpxmap import HpxMap
from .hpx import HpxGeom, HpxToWcsMapping

__all__ = [
    'HpxMapND',
]


class HpxMapND(HpxMap):
    """Representation of a N+2D map using HEALPIX with two spatial
    dimensions and N non-spatial dimensions.

    This class uses a numpy
    array to represent the sequence of HEALPix image planes.  As such
    it can only be used for maps with the same geometry (NSIDE and
    HPX_REG) in every plane.  Following the convention of WCS-based
    maps this class uses a column-wise ordering for the data array
    with the spatial dimension being tied to the last index of the
    array.

    Parameters
    ----------
    hpx : `~gammapy.maps.hpx.HpxGeom`
        HEALPIX geometry object.
    data : `~numpy.ndarray`
        HEALPIX data array.
        If none then an empty array will be allocated.
    """

    def __init__(self, hpx, data=None, dtype='float32'):

        npix = np.unique(hpx.npix)
        if len(npix) > 1:
            raise Exception('HpxMapND can only be instantiated from a '
                            'HPX geometry with the same nside in '
                            'every plane.')

        shape = tuple(list(hpx._shape[::-1]) + [npix[0]])
        if data is None:
            data = np.zeros(shape, dtype=dtype)
        elif data.shape != shape:
            raise ValueError('Wrong shape for input data array. Expected {} '
                             'but got {}'.format(shape, data.shape))

        HpxMap.__init__(self, hpx, data)
        self._wcs2d = None
        self._hpx2wcs = None

    @classmethod
    def from_hdu(cls, hdu, axes):
        """Make a HpxMapND object from a FITS HDU.

        Parameters
        ----------
        hdu : `~astropy.fits.BinTableHDU`
            The FITS HDU
        axes : list
            List of axes for non-spatial dimensions.
        """
        hpx = HpxGeom.from_header(hdu.header, axes)
        shape = tuple([ax.nbin for ax in axes[::-1]])
        shape_data = shape + tuple(np.unique(hpx.npix))

        # FIXME: We need to assert here if the file is incompatible
        # with an ND-array
        # TODO: Should we support extracting slices?

        colnames = hdu.columns.names
        cnames = []
        if hdu.header['INDXSCHM'] == 'SPARSE':
            pix = hdu.data.field('PIX')
            vals = hdu.data.field('VALUE')
            if 'CHANNEL' in hdu.data.columns.names:
                chan = hdu.data.field('CHANNEL')
                chan = np.unravel_index(chan, shape)
                idx = chan + (pix,)
            else:
                idx = (pix,)

            data = np.zeros(shape_data)
            data[idx] = vals
        else:
            for c in colnames:
                if c.find(hpx.conv.colstring) == 0:
                    cnames.append(c)
            nbin = len(cnames)
            data = np.ndarray(shape_data)
            if len(cnames) == 1:
                data[:] = hdu.data.field(cnames[0])
            else:
                for i, cname in enumerate(cnames):
                    idx = np.unravel_index(i, shape)
                    data[idx] = hdu.data.field(cname)
        return cls(hpx, data)

    def make_wcs_mapping(self, sum_bands=False, proj='AIT', oversample=2):
        """Make a HEALPix to WCS mapping object.

        Parameters
        ----------
        sum_bands : bool
           sum over non-spatial dimensions before reprojecting
        proj  : str
           WCS-projection
        oversample : int
           Oversampling factor for WCS map
        normalize : bool
           True -> preserve integral by splitting HEALPIX values between bins

        Returns
        -------
        wcs : `~astropy.wcs.WCS`
            WCS object
        data : `~numpy.ndarray`
            Reprojected data
        """
        self._wcs_proj = proj
        self._wcs_oversample = oversample
        self._wcs2d = self.hpx.make_wcs(proj=proj, oversample=oversample,
                                        drop_axes=True)
        self._hpx2wcs = HpxToWcsMapping.create(self.hpx, self._wcs2d)

    def to_wcs(self, sum_bands=False, normalize=True, proj='AIT', oversample=2):

        from .wcsnd import WcsMapND

        # FIXME: Check whether the old mapping is still valid and reuse it
        self.make_wcs_mapping(oversample=oversample)
        hpx_data = self.data

        # FIXME: Need a function to extract a valid shape from npix property

        if sum_bands:
            hpx_data = np.apply_over_axes(np.sum, hpx_data,
                                          axes=np.arange(hpx_data.ndim - 1))
            wcs_shape = tuple([t.flat[0] for t in self._hpx2wcs.npix])
            wcs_data = np.zeros(wcs_shape).T
            wcs = self.hpx.make_wcs(proj=proj,
                                    oversample=oversample,
                                    drop_axes=True)
        else:
            wcs_shape = tuple([t.flat[0] for t in
                               self._hpx2wcs.npix]) + self.hpx._shape
            wcs_data = np.zeros(wcs_shape).T
            wcs = self.hpx.make_wcs(proj=proj,
                                    oversample=oversample,
                                    drop_axes=False)

        # FIXME: Should reimplement instantiating map first and fill data array

        self._hpx2wcs.fill_wcs_map_from_hpx_data(hpx_data, wcs_data, normalize)
        return WcsMapND(wcs, wcs_data)

    def get_pixel_skydirs(self):
        """Get a list of sky coordinates for the centers of every pixel. """
        return self._hpx.get_skydirs()

    def sum_over_axes(self):
        """Sum over all non-spatial dimensions."""
        # We sum over axis 0 in the array,
        # and drop the energy binning in the hpx object
        return self.__class__(self.hpx.copy_and_drop_axes(),
                              np.apply_over_axes(np.sum, self.data,
                                                 axes=np.arange(self.data.ndim - 1)))

    def interp_by_coords(self, coords, interp=None):

        if interp == 'linear':
            return self._interp_by_coords(coords, interp)
        else:
            raise ValueError('Invalid interpolation method: {}'.format(interp))

    def get_by_pix(self, pix, interp=None):
        # FIXME: Support local indexing here?
        # FIXME: Support slicing?
        # FIXME: What to return for pixels outside the map

        if interp is None:
            return self.get_by_idx(pix)
        else:
            raise NotImplementedError

    def get_by_idx(self, idx):
        idx = pix_tuple_to_idx(idx)
        idx = self.hpx.global_to_local(idx)
        return self.data.T[idx]

    def _interp_by_coords(self, coords, interp):
        """Linearly interpolate map values."""
        import healpy as hp
        c = MapCoords.create(coords)
        theta = np.array(np.pi / 2. - np.radians(c.lat), ndmin=1)
        phi = np.array(np.radians(c.lon), ndmin=1)

        pix_ctr = self.hpx.coord_to_pix(c)[0]
        pix, wts = hp.pixelfunc.get_interp_weights(self.hpx.nside, theta,
                                                   phi, nest=self.hpx.nest)

        # Convert to local pixel indices
        pix_local = [self.hpx[pix]]

        m = pix_local[0] == -1
        pix_local[0][m] = self.hpx[(
            pix_ctr * np.ones(pix.shape, dtype=int))[m]]

        if np.any(pix_local[0] == -1):
            raise ValueError('HPX pixel index out of map bounds.')

        if self.hpx.ndim == 2:
            return np.sum(self.data.T[pix_local] * wts, axis=0)

        val = np.zeros(theta.shape)
        # Loop over function values at corners
        for i, t in enumerate(range(2 ** len(self.hpx.axes))):

            pix = []
            wt = np.ones(theta.shape)[None, ...]
            for j, ax in enumerate(self.hpx.axes):

                idx = coord_to_idx(ax.center[:-1],
                                   c[2 + j], bounded=True)[None, ...]

                w = ax.center[idx + 1] - ax.center[idx]
                if (i & (1 << j)):
                    wt *= (c[2 + j] - ax.center[idx]) / w
                    pix += [1 + idx]
                else:
                    wt *= (1.0 - (c[2 + j] - ax.center[idx]) / w)
                    pix += [idx]
            val += np.sum(wts * wt * self.data.T[pix_local + pix], axis=0)

        return val

    def fill_by_idx(self, idx, weights=None):

        idx = pix_tuple_to_idx(idx)
        if weights is None:
            weights = np.ones(idx[0].shape)
        idx_local = (self.hpx[idx[0]],) + tuple(idx[1:])
        self.data.T[idx_local] += weights

    def set_by_idx(self, idx, vals):

        idx = pix_tuple_to_idx(idx)
        idx_local = (self.hpx[idx[0]],) + tuple(idx[1:])
        self.data.T[idx_local] = vals

    def to_swapped_scheme(self):

        import healpy as hp
        hpx_out = self.hpx.to_swapped()
        map_out = self.__class__(hpx_out)
        idx = list(self.hpx.get_pixels())
        msk = np.ravel(self.data > 0)
        idx = [t[msk] for t in idx]

        if self.hpx.nest:
            idx_new = tuple([hp.nest2ring(self.hpx.nside, idx[0])] + idx[1:])
        else:
            idx_new = tuple([hp.ring2nest(self.hpx.nside, idx[0])] + idx[1:])

        map_out.set_by_pix(idx_new, np.ravel(self.data)[msk])
        return map_out

    def to_ud_graded(self, order, preserve_counts=False):

        import healpy as hp
        new_hpx = self.hpx.ud_graded_hpx(order)
        map_out = self.__class__(new_hpx)

        idx = list(self.hpx.get_pixels())
        msk = np.ravel(self.data != 0)
        idx = [t[msk] for t in idx]

        if self.hpx.nest:
            idx_new = tuple([hp.nest2ring(self.hpx.nside, idx[0])] + idx[1:])
        else:
            idx_new = tuple([hp.ring2nest(self.hpx.nside, idx[0])] + idx[1:])

        map_out.fill_by_pix(idx_new, np.ravel(self.data)[msk])

        if not preserve_counts:
            map_out.data *= (2 ** order) ** 2 / (2 ** self.hpx.order) ** 2

        return map_out

    def plot(self, ax=None, normalize=False, proj='AIT', oversample=10):
        """Quickplot method.

        This will generate a basic visualization by
        converting to a rasterized WCS image.
        """
        m = self.to_wcs(sum_bands=True,
                        normalize=normalize,
                        proj=proj, oversample=oversample)
        return m.plot(ax)
