# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import numpy as np
from astropy.io import fits
from .utils import unpack_seq
from .geom import pix_tuple_to_idx
from .wcsmap import WcsGeom
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
    wcs : `~gammapy.maps.wcs.WcsGeom`
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

    @classmethod
    def from_hdu(cls, hdu, hdu_bands=None):
        """Make a WcsMapND object from a FITS HDU.

        Parameters
        ----------
        hdu : `~astropy.fits.BinTableHDU` or `~astropy.fits.ImageHDU`
            The map FITS HDU.
        hdu_bands : `~astropy.fits.BinTableHDU`
            The BANDS table HDU.
        """
        geom = WcsGeom.from_header(hdu.header, hdu_bands)
        shape = tuple([ax.nbin for ax in geom.axes])
        shape_wcs = tuple([np.max(geom.npix[0]),
                           np.max(geom.npix[1])])
        shape_data = shape_wcs + shape
        map_out = cls(geom)

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

    def get_by_pix(self, pix, interp=None):
        if interp is None:
            return self.get_by_idx(pix)
        else:
            raise NotImplementedError

    def get_by_idx(self, idx):
        idx = pix_tuple_to_idx(idx)
        return self._data.T[idx]

    def interp_by_coords(self, coords, interp=None):
        if interp == 'linear':
            raise NotImplementedError
        else:
            raise ValueError('Invalid interpolation method: {}'.format(interp))

    def interp_image(self, coords, method='linear'):

        if self.geom.ndim == 2:
            raise ValueError('Operation only supported for maps with one or more '
                             'non-spatial dimensions.')
        elif self.geom.ndim == 3:
            return self._interp_image_cube(coords, method)
        else:
            raise NotImplementedError

    def _interp_image_cube(self, coords, method):
        """Interpolate an image plane of a cube."""

        # TODO: consider re-writing to support maps with > 3 dimensions

        from scipy.interpolate import interp1d

        axis = self.geom.axes[0]
        idx = axis.coord_to_idx_interp(coords[0])
        map_slice = slice(int(idx[0]), int(idx[-1]) + 1)
        pix_vals = [float(t) for t in idx]
        pix = axis.coord_to_pix(coords[0])
        data = self.data[map_slice]
        fn = interp1d(pix_vals, data, copy=False, axis=0, fill_value='extrapolate',
                      kind=method)
        data_interp = fn(float(pix))
        geom = self.geom.to_image()
        return self.__class__(geom, data_interp)

    def fill_by_idx(self, idx, weights=None):
        idx = pix_tuple_to_idx(idx)
        msk = np.all(np.stack([t != -1 for t in idx]), axis=0)
        idx = [t[msk] for t in idx]
        if weights is not None:
            weights = weights[msk]
        idx = np.ravel_multi_index(idx, self.data.T.shape)
        idx, idx_inv = np.unique(idx, return_inverse=True)
        weights = np.bincount(idx_inv, weights=weights)
        self.data.T.flat[idx] += weights

    def set_by_idx(self, idx, vals):
        idx = pix_tuple_to_idx(idx)
        self.data.T[idx] = vals

    def iter_by_image(self):
        for idx in np.ndindex(self.geom.shape):
            yield self.data[idx[::-1]], idx

    def iter_by_pix(self, buffersize=1):
        pix = list(self.geom.get_pixels())
        vals = self.data[np.isfinite(self.data)]
        return unpack_seq(np.nditer([vals] + pix,
                                    flags=['external_loop', 'buffered'],
                                    buffersize=buffersize))

    def iter_by_coords(self, buffersize=1):
        coords = list(self.geom.get_coords())
        vals = self.data[np.isfinite(self.data)]
        return unpack_seq(np.nditer([vals] + coords,
                                    flags=['external_loop', 'buffered'],
                                    buffersize=buffersize))

    def sum_over_axes(self):

        if self.geom.ndim == 2:
            return copy.deepcopy(self)

        map_out = self.__class__(self.geom.to_image())
        if self.geom.npix[0].size > 1:
            vals = self.get_by_idx(self.geom.get_pixels())
            map_out.fill_by_coords(self.geom.get_coords()[:2], vals)
        else:
            data = np.apply_over_axes(np.sum, self.data,
                                      axes=np.arange(self.data.ndim - 2))
            map_out.data = np.squeeze(data)

        return map_out

    def reproject(self, geom, mode='interp'):

        from reproject import reproject_interp, reproject_exact
        from reproject import reproject_from_healpix, reproject_to_healpix
        from .hpx import HpxGeom
        from .hpxcube import HpxMapND

        coordsys = 'galactic' if geom.coordsys == 'GAL' else 'icrs'

        if isinstance(geom, HpxGeom):

            map_out = HpxMapND(geom)
            for vals, idx in map_out.iter_by_image():

                # TODO: Check if non-spatial geometries are compatible
                # otherwise interpolate
                coords = self.geom.pix_to_coord((0,) + idx)
                img = self.interp_image(coords).data

                # TODO: Create WCS object for image plane if
                # multi-resolution geom
                data, footprint = reproject_to_healpix((img, self.geom.wcs),
                                                       coordsys,
                                                       nside=geom.nside, nested=geom.nest,
                                                       order='bilinear')
                vals[...] = data
            return map_out
        elif isinstance(geom, WcsGeom):

            map_out = WcsMapND(geom)
            for vals, idx in map_out.iter_by_image():

                # TODO: Check if geometries are compatible otherwise
                # interpolate
                coords = self.geom.pix_to_coord((0, 0) + idx)
                img = self.interp_image(coords[2:]).data
                shape_out = geom.npix[::-1]

                if mode == 'interp':
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

        else:
            raise TypeError('Invalid type for map geometry.')

    def pad(self, pad_width):
        raise NotImplementedError

    def crop(self, crop_width):
        raise NotImplementedError

    def upsample(self, factor):
        raise NotImplementedError

    def downsample(self, factor):
        raise NotImplementedError

    def plot(self, ax=None, pix_slice=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection=self.geom.wcs)

        if pix_slice is not None:
            slices = (slice(None), slice(None)) + pix_slice
            data = self.data[slices[::-1]]
        else:
            data = self.data

        kwargs.setdefault('interpolation', 'nearest')
        kwargs.setdefault('origin', 'lower')
        im = ax.imshow(data, **kwargs)
        ax.coords.grid(color='w', linestyle=':', linewidth=0.5)
        return im
