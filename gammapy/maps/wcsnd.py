# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import numpy as np
from astropy.io import fits
from collections import OrderedDict
from .utils import unpack_seq
from .geom import pix_tuple_to_idx, axes_pix_to_coord
from .utils import interp_to_order
from .wcsmap import WcsGeom
from .wcsmap import WcsMap
from .reproject import reproject_car_to_hpx, reproject_car_to_wcs

__all__ = [
    'WcsNDMap',
]


class WcsNDMap(WcsMap):
    """Representation of a N+2D map using WCS with two spatial dimensions
    and N non-spatial dimensions.

    This class uses an ND numpy array to store map values. For maps with
    non-spatial dimensions and variable pixel size it will allocate an
    array with dimensions commensurate with the largest image plane.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        WCS geometry object.
    data : `~numpy.ndarray`
        Data array. If none then an empty array will be allocated.
    dtype : str, optional
        Data type, default is float32
    meta : `~collections.OrderedDict`
        Dictionary to store meta data.
    """

    def __init__(self, geom, data=None, dtype='float32', meta=None):
        # TODO: Figure out how to mask pixels for integer data types

        shape = tuple([np.max(geom.npix[0]), np.max(geom.npix[1])] +
                      [ax.nbin for ax in geom.axes])
        if data is None:
            data = self._init_data(geom, shape, dtype)
        elif data.shape != shape[::-1]:
            raise ValueError('Wrong shape for input data array. Expected {} '
                             'but got {}'.format(shape, data.shape))

        super(WcsNDMap, self).__init__(geom, data, meta)

    def _init_data(self, geom, shape, dtype):
        # Check whether corners of each image plane are valid
        coords = []
        if not geom.is_regular:

            for idx in np.ndindex(geom.shape):
                pix = (np.array([0.0, float(geom.npix[0][idx] - 1)]),
                       np.array([0.0, float(geom.npix[1][idx] - 1)]))
                pix += tuple([np.array(2 * [t]) for t in idx])
                coords += geom.pix_to_coord(pix)

        else:

            pix = (np.array([0.0, float(geom.npix[0] - 1)]),
                   np.array([0.0, float(geom.npix[1] - 1)]))
            pix += tuple([np.array(2 * [0.0]) for i in range(geom.ndim - 2)])
            coords += geom.pix_to_coord(pix)

        if np.all(np.isfinite(np.vstack(coords))):

            if geom.is_regular:
                data = np.zeros(shape, dtype=dtype).T
            else:
                data = np.nan * np.ones(shape, dtype=dtype).T
                for idx in np.ndindex(geom.shape):
                    data[idx,
                         slice(geom.npix[0][idx]),
                         slice(geom.npix[1][idx])] = 0.0
        else:
            data = np.full(shape, np.nan, dtype=dtype).T
            idx = geom.get_idx()
            m = np.all(np.stack([t != -1 for t in idx]), axis=0)
            data[m] = 0.0

        return data

    @classmethod
    def from_hdu(cls, hdu, hdu_bands=None):
        """Make a WcsNDMap object from a FITS HDU.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU` or `~astropy.io.fits.ImageHDU`
            The map FITS HDU.
        hdu_bands : `~astropy.io.fits.BinTableHDU`
            The BANDS table HDU.
        """
        geom = WcsGeom.from_header(hdu.header, hdu_bands)
        shape = tuple([ax.nbin for ax in geom.axes])
        shape_wcs = tuple([np.max(geom.npix[0]),
                           np.max(geom.npix[1])])
        meta = cls._get_meta_from_header(hdu.header)
        map_out = cls(geom, meta=meta)

        # TODO: Should we support extracting slices?
        if isinstance(hdu, fits.BinTableHDU):
            pix = hdu.data.field('PIX')
            pix = np.unravel_index(pix, shape_wcs[::-1])
            vals = hdu.data.field('VALUE')
            if 'CHANNEL' in hdu.data.columns.names and shape:
                chan = hdu.data.field('CHANNEL')
                chan = np.unravel_index(chan, shape[::-1])
                idx = chan + pix
            else:
                idx = pix

            map_out.set_by_idx(idx[::-1], vals)
        else:
            map_out.data = hdu.data

        return map_out

    def get_by_idx(self, idx):
        idx = pix_tuple_to_idx(idx)
        return self._data.T[idx]

    def interp_by_coord(self, coords, interp=None):

        if self.geom.is_regular:
            pix = self.geom.coord_to_pix(coords)
            return self.interp_by_pix(pix, interp=interp)
        else:
            return self._interp_by_coord_griddata(coords, interp=interp)

    def interp_by_pix(self, pix, interp=None):
        """Interpolate map values at the given pixel coordinates.
        """
        if not self.geom.is_regular:
            raise ValueError('Pixel-based interpolation not supported for '
                             'non-regular geometries.')

        order = interp_to_order(interp)
        if order == 0 or order == 1:
            return self._interp_by_pix_linear_grid(pix, order=order)
        elif order == 2 or order == 3:
            return self._interp_by_pix_map_coordinates(pix, order=order)
        else:
            raise ValueError('Invalid interpolation order: {}'.format(order))

    def _interp_by_pix_linear_grid(self, pix, order=1):
        # TODO: Cache interpolator
        method_lookup = {0: 'nearest', 1: 'linear'}
        method = method_lookup.get(order, None)
        if method is None:
            raise ValueError('Invalid interpolation method: {}'.format(interp))

        from scipy.interpolate import RegularGridInterpolator
        grid_pix = [np.arange(n, dtype=float) for n in self.data.shape[::-1]]

        if np.any(np.isfinite(self.data)):
            data = self.data.copy().T
            data[~np.isfinite(data)] = 0.0
        else:
            data = self.data.T

        fn = RegularGridInterpolator(grid_pix, data, fill_value=None,
                                     bounds_error=False, method=method)
        return fn(tuple(pix))

    def _interp_by_pix_map_coordinates(self, pix, order=1):
        from scipy.ndimage import map_coordinates
        pix = tuple([np.array(x, ndmin=1)
                     if not isinstance(x, np.ndarray) or x.ndim == 0 else x for x in pix])
        return map_coordinates(self.data.T, pix, order=order, mode='nearest')

    def _interp_by_coord_griddata(self, coords, interp=None):
        order = interp_to_order(interp)
        method_lookup = {0: 'nearest', 1: 'linear', 3: 'cubic'}
        method = method_lookup.get(order, None)
        if method is None:
            raise ValueError('Invalid interpolation method: {}'.format(interp))

        from scipy.interpolate import griddata
        grid_coords = self.geom.get_coord(flat=True)
        data = self.data[np.isfinite(self.data)]
        vals = griddata(grid_coords, data, coords, method=method)
        m = ~np.isfinite(vals)
        if np.any(m):
            vals_fill = griddata(grid_coords, data, tuple([c[m] for c in coords]),
                                 method='nearest')
            vals[m] = vals_fill
        return vals

    def interp_image(self, coords, order=1):
        if self.geom.ndim == 2:
            raise ValueError('Operation only supported for maps with one or more '
                             'non-spatial dimensions.')
        elif self.geom.ndim == 3:
            return self._interp_image_cube(coords, order)
        else:
            raise NotImplementedError

    def _interp_image_cube(self, coords, order=1):
        """Interpolate an image plane of a cube."""
        # TODO: consider re-writing to support maps with > 3 dimensions

        from scipy.interpolate import interp1d

        axis = self.geom.axes[0]
        idx = axis.coord_to_idx_interp(coords[0])
        map_slice = slice(int(idx[0]), int(idx[-1]) + 1)
        pix_vals = [float(t) for t in idx]
        pix = axis.coord_to_pix(coords[0])
        data = self.data[map_slice]

        if coords[0] < axis.center[0] or coords[0] > axis.center[-1]:
            kind = 'linear' if order >= 1 else 'nearest'
            fill_value = 'extrapolate'
        else:
            kind = order
            fill_value = None

        # TODO: Cache interpolating function?

        fn = interp1d(pix_vals, data, copy=False, axis=0,
                      kind=kind, fill_value=fill_value)
        data_interp = fn(float(pix))
        geom = self.geom.to_image()
        return self.__class__(geom, data_interp)

    def fill_by_idx(self, idx, weights=None):
        idx = pix_tuple_to_idx(idx)
        msk = np.all(np.stack([t != -1 for t in idx]), axis=0)
        idx = [t[msk] for t in idx]
        if weights is not None:
            weights = np.asarray(weights, dtype=self.data.dtype)
            weights = weights[msk]
        idx = np.ravel_multi_index(idx, self.data.T.shape)
        idx, idx_inv = np.unique(idx, return_inverse=True)
        weights = np.bincount(idx_inv, weights=weights).astype(self.data.dtype)
        self.data.T.flat[idx] += weights

    def set_by_idx(self, idx, vals):
        idx = pix_tuple_to_idx(idx)
        self.data.T[idx] = vals

    def iter_by_image(self):
        for idx in np.ndindex(self.geom.shape):
            yield self.data[idx[::-1]], idx

    def iter_by_pix(self, buffersize=1):
        pix = list(self.geom.get_idx(flat=True))
        vals = self.data[np.isfinite(self.data)]
        return unpack_seq(np.nditer([vals] + pix,
                                    flags=['external_loop', 'buffered'],
                                    buffersize=buffersize))

    def iter_by_coord(self, buffersize=1):
        coords = list(self.geom.get_coord(flat=True))
        vals = self.data[np.isfinite(self.data)]
        return unpack_seq(np.nditer([vals] + coords,
                                    flags=['external_loop', 'buffered'],
                                    buffersize=buffersize))

    def sum_over_axes(self):
        if self.geom.ndim == 2:
            return copy.deepcopy(self)

        map_out = self.__class__(self.geom.to_image())
        if not self.geom.is_regular:
            vals = self.get_by_idx(self.geom.get_idx())
            map_out.fill_by_coord(self.geom.get_coord()[:2], vals)
        else:
            axis = tuple(np.arange(self.data.ndim - 2).tolist())
            map_out.data = np.sum(self.data, axis=axis)

        return map_out

    def _reproject_wcs(self, geom, mode='interp', order=1):
        from reproject import reproject_interp, reproject_exact

        map_out = WcsNDMap(geom)
        axes_eq = np.all([ax0 == ax1 for ax0, ax1 in
                          zip(geom.axes, self.geom.axes)])

        for vals, idx in map_out.iter_by_image():

            if self.geom.ndim == 2 or axes_eq:
                img = self.data[idx[::-1]]
            else:
                coords = axes_pix_to_coord(geom.axes, idx)
                img = self.interp_image(coords, order=order).data

            # FIXME: This is a temporary solution for handling maps
            # with undefined pixels
            if np.any(~np.isfinite(img)):
                img = img.copy()
                img[~np.isfinite(img)] = 0.0

            # TODO: Create WCS object for image plane if
            # multi-resolution geom
            shape_out = geom.get_image_shape(idx)[::-1]

            if self.geom.projection == 'CAR' and self.geom.is_allsky:
                data, footprint = reproject_car_to_wcs((img, self.geom.wcs),
                                                       geom.wcs,
                                                       shape_out=shape_out)
            elif mode == 'interp':
                data, footprint = reproject_interp((img, self.geom.wcs),
                                                   geom.wcs,
                                                   shape_out=shape_out)
            elif mode == 'exact':
                data, footprint = reproject_exact((img, self.geom.wcs),
                                                  geom.wcs,
                                                  shape_out=shape_out)
            else:
                raise TypeError(
                    "Invalid reprojection mode, either choose 'interp' or 'exact'")

            vals[...] = data

        return map_out

    def _reproject_hpx(self, geom, mode='interp', order=1):
        from reproject import reproject_to_healpix
        from .hpxnd import HpxNDMap

        map_out = HpxNDMap(geom)
        coordsys = 'galactic' if geom.coordsys == 'GAL' else 'icrs'
        axes_eq = np.all([ax0 == ax1 for ax0, ax1 in
                          zip(geom.axes, self.geom.axes)])

        for vals, idx in map_out.iter_by_image():

            if self.geom.ndim == 2 or axes_eq:
                img = self.data[idx[::-1]]
            else:
                coords = axes_pix_to_coord(geom.axes, idx)
                img = self.interp_image(coords, order=order).data

            # TODO: For partial-sky HPX we need to map from full- to
            # partial-sky indices
            if self.geom.projection == 'CAR' and self.geom.is_allsky:
                data, footprint = reproject_car_to_hpx((img, self.geom.wcs),
                                                       coordsys,
                                                       nside=geom.nside,
                                                       nested=geom.nest,
                                                       order=order)
            else:
                data, footprint = reproject_to_healpix((img, self.geom.wcs),
                                                       coordsys,
                                                       nside=geom.nside,
                                                       nested=geom.nest,
                                                       order=order)
            vals[...] = data

        return map_out

    def pad(self, pad_width, mode='constant', cval=0, order=1):

        if np.isscalar(pad_width):
            pad_width = (pad_width, pad_width)
            pad_width += (0,) * (self.geom.ndim - 2)

        geom = self.geom.pad(pad_width[:2])
        if self.geom.is_regular and mode != 'interp':
            return self._pad_np(geom, pad_width, mode, cval)
        else:
            return self._pad_coadd(geom, pad_width, mode, cval, order)

    def _pad_np(self, geom, pad_width, mode, cval):
        """Pad a map with `~np.pad`.  This method only works for regular
        geometries but should be more efficient when working with
        large maps.
        """
        kw = {}
        if mode == 'constant':
            kw['constant_values'] = cval

        pad_width = [(t, t) for t in pad_width]
        data = np.pad(self.data, pad_width[::-1], mode, **kw)
        map_out = self.__class__(geom, data, meta=copy.deepcopy(self.meta))
        return map_out

    def _pad_coadd(self, geom, pad_width, mode, cval, order):
        """Pad a map manually by coadding the original map with the new
        map."""
        idx_in = self.geom.get_idx(flat=True)
        idx_in = tuple([t + w for t, w in zip(idx_in, pad_width)])[::-1]
        idx_out = geom.get_idx(flat=True)[::-1]
        map_out = self.__class__(geom, meta=copy.deepcopy(self.meta))
        map_out.coadd(self)
        if mode == 'constant':
            pad_msk = np.zeros_like(map_out.data, dtype=bool)
            pad_msk[idx_out] = True
            pad_msk[idx_in] = False
            map_out.data[pad_msk] = cval
        elif mode in ['edge', 'interp']:
            coords = geom.pix_to_coord(idx_out[::-1])
            m = self.geom.contains(coords)
            coords = tuple([c[~m] for c in coords])
            vals = self.interp_by_coord(coords, interp=0 if mode == 'edge'
                                        else order)
            map_out.set_by_coord(coords, vals)
        else:
            raise ValueError('Unrecognized pad mode: {}'.format(mode))

        return map_out

    def crop(self, crop_width):

        if np.isscalar(crop_width):
            crop_width = (crop_width, crop_width)
        geom = self.geom.crop(crop_width)
        if self.geom.is_regular:
            slices = [slice(crop_width[0], int(self.geom.npix[0] - crop_width[0])),
                      slice(crop_width[1], int(self.geom.npix[1] - crop_width[1]))]
            for ax in self.geom.axes:
                slices += [slice(None)]
            data = self.data[slices[::-1]]
            map_out = self.__class__(geom, data, meta=copy.deepcopy(self.meta))
        else:
            # FIXME: This could be done more efficiently by
            # constructing the appropriate slices for each image plane
            map_out = self.__class__(geom, meta=copy.deepcopy(self.meta))
            map_out.coadd(self)

        return map_out

    def upsample(self, factor, order=0, preserve_counts=True):
        from scipy.ndimage import map_coordinates
        geom = self.geom.upsample(factor)
        idx = geom.get_idx()
        pix = ((idx[0] - 0.5 * (factor - 1)) / factor,
               (idx[1] - 0.5 * (factor - 1)) / factor,) + idx[2:]
        data = map_coordinates(self.data.T, pix, order=order, mode='nearest')
        if preserve_counts:
            data /= factor**2
        return self.__class__(geom, data, meta=copy.deepcopy(self.meta))

    def downsample(self, factor, preserve_counts=True):
        from skimage.measure import block_reduce
        geom = self.geom.downsample(factor)
        block_size = tuple([factor, factor] + [1] * (self.geom.ndim - 2))
        data = block_reduce(self.data, block_size[::-1], np.nansum)
        if not preserve_counts:
            data /= factor**2
        return self.__class__(geom, data, meta=copy.deepcopy(self.meta))

    def plot(self, ax=None, idx=None, **kwargs):
        """Quickplot method.

        Parameters
        ----------
        norm : str
            Set the normalization scheme of the color map.
        idx : tuple
            Set the image slice to plot if this map has non-spatial
            dimensions.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.imshow`.   

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object.
        ax : `~astropy.visualization.wcsaxes.WCSAxes`
            WCS axis object
        im : `~matplotlib.image.AxesImage`
            Image object.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection=self.geom.wcs)

        if idx is not None:
            slices = (slice(None), slice(None)) + idx
            data = self.data[slices[::-1]]
        else:
            data = self.data

        kwargs.setdefault('interpolation', 'nearest')
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('norm', None)

        if kwargs['norm'] == 'log':
            kwargs['norm'] = colors.LogNorm()
        elif kwargs['norm'] == 'pow2':
            kwargs['norm'] = colors.PowerNorm(gamma=0.5)

        im = ax.imshow(data, **kwargs)
        ax.coords.grid(color='w', linestyle=':', linewidth=0.5)
        return fig, ax, im
