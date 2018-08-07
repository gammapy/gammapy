# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from astropy.nddata import Cutout2D
from astropy.convolution import Tophat2DKernel
from .utils import unpack_seq
from .geom import pix_tuple_to_idx, axes_pix_to_coord
from .utils import interp_to_order
from .wcsmap import WcsGeom
from .wcsmap import WcsMap
from .reproject import reproject_car_to_hpx, reproject_car_to_wcs

__all__ = [
    'WcsNDMap',
]

log = logging.getLogger(__name__)


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
    unit : str or `~astropy.units.Unit`
        The map unit
    """

    def __init__(self, geom, data=None, dtype='float32', meta=None, unit=''):
        # TODO: Figure out how to mask pixels for integer data types

        data_shape = geom.data_shape

        if data is None:
            data = self._make_default_data(geom, data_shape, dtype)

        super(WcsNDMap, self).__init__(geom, data, meta, unit)

    @staticmethod
    def _make_default_data(geom, shape_np, dtype):
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
                data = np.zeros(shape_np, dtype=dtype)
            else:
                data = np.full(shape_np, np.nan, dtype=dtype)
                for idx in np.ndindex(geom.shape):
                    data[idx,
                         slice(geom.npix[0][idx]),
                         slice(geom.npix[1][idx])] = 0.0
        else:
            data = np.full(shape_np, np.nan, dtype=dtype)
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

        unit = hdu.header.get('UNIT', '')

        map_out = cls(geom, meta=meta, unit=unit)

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
        return self.data.T[idx]

    def interp_by_coord(self, coords, interp=None, fill_value=None):

        if self.geom.is_regular:
            pix = self.geom.coord_to_pix(coords)
            return self.interp_by_pix(pix, interp=interp, fill_value=fill_value)
        else:
            return self._interp_by_coord_griddata(coords, interp=interp)

    def interp_by_pix(self, pix, interp=None, fill_value=None):
        """Interpolate map values at the given pixel coordinates.
        """
        if not self.geom.is_regular:
            raise ValueError('Pixel-based interpolation not supported for '
                             'non-regular geometries.')

        order = interp_to_order(interp)
        if order == 0 or order == 1:
            return self._interp_by_pix_linear_grid(pix, order=order, fill_value=fill_value)
        elif order == 2 or order == 3:
            return self._interp_by_pix_map_coordinates(pix, order=order)
        else:
            raise ValueError('Invalid interpolation order: {}'.format(order))

    def _interp_by_pix_linear_grid(self, pix, order=1, fill_value=None):
        # TODO: Cache interpolator
        method_lookup = {0: 'nearest', 1: 'linear'}
        try:
            method = method_lookup[order]
        except KeyError:
            raise ValueError('Invalid interpolation order: {}'.format(order))

        from scipy.interpolate import RegularGridInterpolator
        grid_pix = [np.arange(n, dtype=float) for n in self.data.shape[::-1]]

        if np.any(np.isfinite(self.data)):
            data = self.data.copy().T
            data[~np.isfinite(data)] = 0.0
        else:
            data = self.data.T

        fn = RegularGridInterpolator(grid_pix, data, fill_value=fill_value,
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
        grid_coords = tuple(self.geom.get_coord(flat=True))
        data = self.data[np.isfinite(self.data)]
        vals = griddata(grid_coords, data, tuple(coords), method=method)

        m = ~np.isfinite(vals)
        if np.any(m):
            vals_fill = griddata(grid_coords, data, tuple([c[m] for c in coords]),
                                 method='nearest')
            vals[m] = vals_fill

        return vals

    # Currently used by reproject.
    # TODO: Consider replacing with `interp_by_coord`.
    def _interp_image(self, coords, order=1):
        from scipy.interpolate import interp1d

        if self.geom.ndim != 3:
            raise ValueError('Only support geometry with ndim=3 at the moment')

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

        fn = interp1d(pix_vals, data, copy=False, axis=0,
                      kind=kind, fill_value=fill_value)
        data_interp = fn(float(pix))
        geom = self.geom.to_image()
        return self._init_copy(data=data_interp, geom=geom)

    def fill_by_idx(self, idx, weights=None):
        idx = pix_tuple_to_idx(idx)
        msk = np.all(np.stack([t != -1 for t in idx]), axis=0)
        idx = [t[msk] for t in idx]

        if weights is not None:
            if isinstance(weights, Quantity):
                weights = weights.to(self.unit).value
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
        axis = tuple(range(self.data.ndim - 2))
        data = np.nansum(self.data, axis=axis)
        geom = self.geom.to_image()
        # TODO: summing over the axis can change the unit, handle this correctly
        return self._init_copy(geom=geom, data=data)

    def _reproject_wcs(self, geom, mode='interp', order=1):
        from reproject import reproject_interp, reproject_exact

        map_out = WcsNDMap(geom, unit=self.unit)
        axes_eq = np.all([ax0 == ax1 for ax0, ax1 in
                          zip(geom.axes, self.geom.axes)])

        for vals, idx in map_out.iter_by_image():

            if self.geom.ndim == 2 or axes_eq:
                img = self.data[idx[::-1]]
            else:
                coords = axes_pix_to_coord(geom.axes, idx)
                img = self._interp_image(coords, order=order).data

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
                img = self._interp_image(coords, order=order).data

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
        kwargs = {}
        if mode == 'constant':
            kwargs['constant_values'] = cval

        pad_width = [(t, t) for t in pad_width]
        data = np.pad(self.data, pad_width[::-1], mode)
        return self._init_copy(geom=geom, data=data)

    def _pad_coadd(self, geom, pad_width, mode, cval, order):
        """Pad a map manually by coadding the original map with the new
        map."""
        idx_in = self.geom.get_idx(flat=True)
        idx_in = tuple([t + w for t, w in zip(idx_in, pad_width)])[::-1]
        idx_out = geom.get_idx(flat=True)[::-1]
        map_out = self._init_copy(geom=geom, data=None)
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
            map_out = self._init_copy(geom=geom, data=data)
        else:
            # FIXME: This could be done more efficiently by
            # constructing the appropriate slices for each image plane
            map_out = self._init_copy(geom=geom, data=None)
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
            data /= factor ** 2
        return self._init_copy(geom=geom, data=data)

    def downsample(self, factor, preserve_counts=True):
        from skimage.measure import block_reduce
        geom = self.geom.downsample(factor)
        block_size = tuple([factor, factor] + [1] * (self.geom.ndim - 2))
        data = block_reduce(self.data, block_size[::-1], np.nansum)
        if not preserve_counts:
            data /= factor ** 2
        return self._init_copy(geom=geom, data=data)

    def plot(self, ax=None, fig=None, add_cbar=False, stretch='linear', **kwargs):
        """
        Plot image on matplotlib WCS axes.

        Parameters
        ----------
        ax : `~astropy.visualization.wcsaxes.WCSAxes`, optional
            WCS axis object to plot on.
        fig : `~matplotlib.figure.Figure`
            Figure object.
        add_cbar : bool
            Add color bar?
        stretch : str
            Passed to `astropy.visualization.simple_norm`.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.imshow`.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object.
        ax : `~astropy.visualization.wcsaxes.WCSAxes`
            WCS axis object
        cbar : `~matplotlib.colorbar.Colorbar` or None
            Colorbar object.
        """
        import matplotlib.pyplot as plt
        from astropy.visualization import simple_norm

        if not self.geom.is_image:
            raise TypeError('Use .plot_interactive() for Map dimension > 2')

        if fig is None:
            fig = plt.gcf()

        if ax is None:
            ax = fig.add_subplot(1, 1, 1, projection=self.geom.wcs)

        data = self.data.astype(float)

        kwargs.setdefault('interpolation', 'nearest')
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('cmap', 'afmhot')
        norm = simple_norm(data[np.isfinite(data)], stretch)
        kwargs.setdefault('norm', norm)

        caxes = ax.imshow(data, **kwargs)

        cbar = fig.colorbar(caxes, ax=ax) if add_cbar else None
        try:
            ax.coords['glon'].set_axislabel('Galactic Longitude')
            ax.coords['glat'].set_axislabel('Galactic Latitude')
        except KeyError:
            ax.coords['ra'].set_axislabel('Right Ascension')
            ax.coords['dec'].set_axislabel('Declination')
        except AttributeError:
            log.info("Can't set coordinate axes. No WCS information available.")

        # without this the axis limits are changed when calling scatter
        ax.autoscale(enable=False)
        return fig, ax, cbar

    def plot_interactive(self, ax=None, fig=None, **kwargs):
        """
        Plot ND array on matplotlib WCS axes with interactive widgets
        to explore the non spatial axes.

        Parameters
        ----------
        ax : `~astropy.visualization.wcsaxes.WCSAxes`, optional
            WCS axis object to plot on.
        fig : `~matplotlib.figure.Figure`
            Figure object.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.imshow`.

        Examples
        --------

        You can try this out e.g. using a Fermi-LAT diffuse model cube with an energy axis::

            %matplotlib inline
            from gammapy.maps import Map

            m = Map.read("$GAMMAPY_EXTRA/datasets/vela_region/gll_iem_v05_rev1_cutout.fits")
            m.plot_interactive(cmap='gnuplot2')
        """
        import matplotlib.pyplot as plt
        from astropy.visualization import simple_norm
        from ipywidgets.widgets.interaction import interact, fixed
        import ipywidgets as widgets

        kwargs.setdefault('interpolation', 'nearest')
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('cmap', 'afmhot')

        @interact(
            index=widgets.IntSlider(min=0, max=self.data.shape[0] - 1, step=1, value=1,
                                    description=self.geom.axes[0].name + ' slice'),
            stretch=widgets.RadioButtons(options=['linear', 'sqrt', 'log'], value='sqrt',
                                         description='Plot stretch'),
            ax=fixed(ax),
            fig=fixed(fig),
        )
        def _plot_interactive(index, stretch, ax=None, fig=None):
            if self.geom.is_image:
                raise TypeError('Use .plot() for 2D Maps')

            if fig is None:
                fig = plt.gcf()

            if ax is None:
                ax = fig.add_subplot(1, 1, 1, projection=self.geom.wcs)

            axes = self.geom.axes[0]

            data = self.get_image_by_idx([index]).data
            norm = simple_norm(data[np.isfinite(data)], stretch)

            caxes = ax.imshow(data, norm=norm, **kwargs)
            fig.colorbar(caxes, ax=ax)
            ax.set_title(
                '{:.2f}-{:.2f} {} '.format(
                    axes.edges[index], axes.edges[index + 1],
                    self.geom.axes[0].unit.name,
                )
            )

            try:
                ax.coords['glon'].set_axislabel('Galactic Longitude')
                ax.coords['glat'].set_axislabel('Galactic Latitude')
            except KeyError:
                ax.coords['ra'].set_axislabel('Right Ascension')
                ax.coords['dec'].set_axislabel('Declination')
            except AttributeError:
                log.info("Can't set coordinate axes. No WCS information available.")

            plt.show()

    def smooth(self, radius, kernel='gauss', **kwargs):
        """
        Smooth the image (works on a 2D image and returns a copy).

        The definition of the smoothing parameter radius is equivalent to the
        one that is used in ds9 (see `ds9 smoothing <http://ds9.si.edu/doc/ref/how.html#Smoothing>`_).

        Parameters
        ----------
        radius : `~astropy.units.Quantity` or float
            Smoothing width given as quantity or float. If a float is given it
            interpreted as smoothing width in pixels. If an (angular) quantity
            is given it converted to pixels using `geom.wcs.wcs.cdelt`.
        kernel : {'gauss', 'disk', 'box'}
            Kernel shape
        kwargs : dict
            Keyword arguments passed to `~scipy.ndimage.uniform_filter`
            ('box'), `~scipy.ndimage.gaussian_filter` ('gauss') or
            `~scipy.ndimage.convolve` ('disk').

        Returns
        -------
        image : `WcsNDMap`
            Smoothed image (a copy, the original object is unchanged).
        """
        from scipy.ndimage import gaussian_filter, uniform_filter, convolve

        if not self.geom.is_image:
            raise ValueError('Only supported on 2D maps')

        if isinstance(radius, Quantity):
            radius = (radius.to('deg') / self.geom.pixel_scales.mean()).value

        if kernel == 'gauss':
            width = radius / 2.
            data = gaussian_filter(self.data, width, **kwargs)
        elif kernel == 'disk':
            width = 2 * radius + 1
            disk = Tophat2DKernel(width)
            disk.normalize('integral')
            data = convolve(self.data, disk.array, **kwargs)
        elif kernel == 'box':
            width = 2 * radius + 1
            data = uniform_filter(self.data, width, **kwargs)
        else:
            raise ValueError('Invalid option kernel = {}'.format(kernel))

        return self._init_copy(data=data)

    def make_cutout(self, position, width, mode="strict", copy=True):
        """
        Create a cutout of a WcsNDMap around a given position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Center position of the cutout region.
        width : tuple of `~astropy.coordinates.Angle`
            Angular sizes of the region in (lon, lat). If only one value is passed,
            a square region is extracted. For more options see also `~astropy.nddata.utils.Cutout2D`.
        mode : {'trim', 'partial', 'strict'}
            Mode option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.
        copy : bool, optional
               If False (default), then the cutout data will be a view into the original data  array.
               If True, then the cutout data will hold a copy of the original data array.

        Returns
        -------
        cutout : `~gammapy.maps.WcsNDMap`
            The cutout map itself
        cutout_slices : Tuple of slice objects (with dimension 1 less than that of the non-spatial axes of the map)
        """
        idx = (0,) * len(self.geom.axes)

        cutout2d = Cutout2D(data=self.data[idx], wcs=self.geom.wcs,
                            position=position, size=width, mode=mode)

        # Create the slices with the non-spatial axis
        cutout_slices = Ellipsis, cutout2d.slices_original[0], cutout2d.slices_original[1]

        geom = WcsGeom(cutout2d.wcs, cutout2d.shape[::-1], axes=self.geom.axes)
        data = self.data[cutout_slices]

        if copy:
            data = data.copy()

        return self._init_copy(geom=geom, data=data), cutout_slices
