# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.convolution import Tophat2DKernel
from ..extern.skimage import block_reduce
from ..utils.units import unit_from_fits_image_hdu
from .geom import pix_tuple_to_idx
from .wcs import _check_width
from .utils import interp_to_order
from .wcsmap import WcsGeom, WcsMap
from .reproject import reproject_car_to_hpx, reproject_car_to_wcs

__all__ = ["WcsNDMap"]

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

    def __init__(self, geom, data=None, dtype="float32", meta=None, unit=""):
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
                pix = (
                    np.array([0.0, float(geom.npix[0][idx] - 1)]),
                    np.array([0.0, float(geom.npix[1][idx] - 1)]),
                )
                pix += tuple([np.array(2 * [t]) for t in idx])
                coords += geom.pix_to_coord(pix)

        else:
            pix = (
                np.array([0.0, float(geom.npix[0] - 1)]),
                np.array([0.0, float(geom.npix[1] - 1)]),
            )
            pix += tuple([np.array(2 * [0.0]) for i in range(geom.ndim - 2)])
            coords += geom.pix_to_coord(pix)

        if np.all(np.isfinite(np.vstack(coords))):
            if geom.is_regular:
                data = np.zeros(shape_np, dtype=dtype)
            else:
                data = np.full(shape_np, np.nan, dtype=dtype)
                for idx in np.ndindex(geom.shape):
                    data[idx, slice(geom.npix[0][idx]), slice(geom.npix[1][idx])] = 0.0
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
        shape_wcs = tuple([np.max(geom.npix[0]), np.max(geom.npix[1])])

        meta = cls._get_meta_from_header(hdu.header)
        unit = unit_from_fits_image_hdu(hdu.header)
        map_out = cls(geom, meta=meta, unit=unit)

        # TODO: Should we support extracting slices?
        if isinstance(hdu, fits.BinTableHDU):
            pix = hdu.data.field("PIX")
            pix = np.unravel_index(pix, shape_wcs[::-1])
            vals = hdu.data.field("VALUE")
            if "CHANNEL" in hdu.data.columns.names and shape:
                chan = hdu.data.field("CHANNEL")
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
            raise ValueError("interp_by_pix only supported for regular geom.")

        order = interp_to_order(interp)
        if order == 0 or order == 1:
            return self._interp_by_pix_linear_grid(
                pix, order=order, fill_value=fill_value
            )
        elif order == 2 or order == 3:
            return self._interp_by_pix_map_coordinates(pix, order=order)
        else:
            raise ValueError("Invalid interpolation order: {!r}".format(order))

    def _interp_by_pix_linear_grid(self, pix, order=1, fill_value=None):
        # TODO: Cache interpolator
        method_lookup = {0: "nearest", 1: "linear"}
        try:
            method = method_lookup[order]
        except KeyError:
            raise ValueError("Invalid interpolation order: {!r}".format(order))

        from scipy.interpolate import RegularGridInterpolator

        grid_pix = [np.arange(n, dtype=float) for n in self.data.shape[::-1]]

        if np.any(np.isfinite(self.data)):
            data = self.data.copy().T
            data[~np.isfinite(data)] = 0.0
        else:
            data = self.data.T

        fn = RegularGridInterpolator(
            grid_pix, data, fill_value=fill_value, bounds_error=False, method=method
        )
        return fn(tuple(pix))

    def _interp_by_pix_map_coordinates(self, pix, order=1):
        from scipy.ndimage import map_coordinates

        pix = tuple(
            [
                np.array(x, ndmin=1)
                if not isinstance(x, np.ndarray) or x.ndim == 0
                else x
                for x in pix
            ]
        )
        return map_coordinates(self.data.T, pix, order=order, mode="nearest")

    def _interp_by_coord_griddata(self, coords, interp=None):
        order = interp_to_order(interp)
        method_lookup = {0: "nearest", 1: "linear", 3: "cubic"}
        method = method_lookup.get(order, None)
        if method is None:
            raise ValueError("Invalid interp: {!r}".format(interp))

        from scipy.interpolate import griddata

        grid_coords = tuple(self.geom.get_coord(flat=True))
        data = self.data[np.isfinite(self.data)]
        vals = griddata(grid_coords, data, tuple(coords), method=method)

        m = ~np.isfinite(vals)
        if np.any(m):
            vals_fill = griddata(
                grid_coords, data, tuple([c[m] for c in coords]), method="nearest"
            )
            vals[m] = vals_fill

        return vals

    def fill_by_idx(self, idx, weights=None):
        idx = pix_tuple_to_idx(idx)
        msk = np.all(np.stack([t != -1 for t in idx]), axis=0)
        idx = [t[msk] for t in idx]

        if weights is not None:
            if isinstance(weights, u.Quantity):
                weights = weights.to(self.unit).value
            weights = weights[msk]

        idx = np.ravel_multi_index(idx, self.data.T.shape)
        idx, idx_inv = np.unique(idx, return_inverse=True)
        weights = np.bincount(idx_inv, weights=weights).astype(self.data.dtype)
        self.data.T.flat[idx] += weights

    def set_by_idx(self, idx, vals):
        idx = pix_tuple_to_idx(idx)
        self.data.T[idx] = vals

    def sum_over_axes(self):
        axis = tuple(range(self.data.ndim - 2))
        data = np.nansum(self.data, axis=axis)
        geom = self.geom.to_image()
        # TODO: summing over the axis can change the unit, handle this correctly
        return self._init_copy(geom=geom, data=data)

    def _reproject_to_wcs(self, geom, mode="interp", order=1):
        from reproject import reproject_interp, reproject_exact

        data = np.empty(geom.data_shape)

        for img, idx in self.iter_by_image():
            # TODO: Create WCS object for image plane if
            # multi-resolution geom
            shape_out = geom.get_image_shape(idx)[::-1]

            if self.geom.projection == "CAR" and self.geom.is_allsky:
                vals, footprint = reproject_car_to_wcs(
                    (img, self.geom.wcs), geom.wcs, shape_out=shape_out
                )
            elif mode == "interp":
                vals, footprint = reproject_interp(
                    (img, self.geom.wcs), geom.wcs, shape_out=shape_out
                )
            elif mode == "exact":
                vals, footprint = reproject_exact(
                    (img, self.geom.wcs), geom.wcs, shape_out=shape_out
                )
            else:
                raise TypeError(
                    "mode must be 'interp' or 'exact'. Got: {!r}".format(mode)
                )

            data[idx] = vals

        return self._init_copy(geom=geom, data=data)

    def _reproject_to_hpx(self, geom, mode="interp", order=1):
        from reproject import reproject_to_healpix

        data = np.empty(geom.data_shape)
        coordsys = "galactic" if geom.coordsys == "GAL" else "icrs"

        for img, idx in self.iter_by_image():
            # TODO: For partial-sky HPX we need to map from full- to
            # partial-sky indices
            if self.geom.projection == "CAR" and self.geom.is_allsky:
                vals, footprint = reproject_car_to_hpx(
                    (img, self.geom.wcs),
                    coordsys,
                    nside=geom.nside,
                    nested=geom.nest,
                    order=order,
                )
            else:
                vals, footprint = reproject_to_healpix(
                    (img, self.geom.wcs),
                    coordsys,
                    nside=geom.nside,
                    nested=geom.nest,
                    order=order,
                )
            data[idx] = vals

        return self._init_copy(geom=geom, data=data)

    def pad(self, pad_width, mode="constant", cval=0, order=1):
        if np.isscalar(pad_width):
            pad_width = (pad_width, pad_width)
            pad_width += (0,) * (self.geom.ndim - 2)

        geom = self.geom.pad(pad_width[:2])
        if self.geom.is_regular and mode != "interp":
            return self._pad_np(geom, pad_width, mode, cval)
        else:
            return self._pad_coadd(geom, pad_width, mode, cval, order)

    def _pad_np(self, geom, pad_width, mode, cval):
        """Pad a map with `~np.pad`.  This method only works for regular
        geometries but should be more efficient when working with
        large maps.
        """
        kwargs = {}
        if mode == "constant":
            kwargs["constant_values"] = cval

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

        if mode == "constant":
            pad_msk = np.zeros_like(map_out.data, dtype=bool)
            pad_msk[idx_out] = True
            pad_msk[idx_in] = False
            map_out.data[pad_msk] = cval
        elif mode == "interp":
            coords = geom.pix_to_coord(idx_out[::-1])
            m = self.geom.contains(coords)
            coords = tuple([c[~m] for c in coords])
            vals = self.interp_by_coord(coords, interp=order)
            map_out.set_by_coord(coords, vals)
        else:
            raise ValueError("Invalid mode: {!r}".format(mode))

        return map_out

    def crop(self, crop_width):
        if np.isscalar(crop_width):
            crop_width = (crop_width, crop_width)

        geom = self.geom.crop(crop_width)
        if self.geom.is_regular:
            slices = [slice(None)] * len(self.geom.axes)
            slices += [
                slice(crop_width[1], int(self.geom.npix[1] - crop_width[1])),
                slice(crop_width[0], int(self.geom.npix[0] - crop_width[0])),
            ]
            data = self.data[tuple(slices)]
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
        pix = (
            (idx[0] - 0.5 * (factor - 1)) / factor,
            (idx[1] - 0.5 * (factor - 1)) / factor,
        ) + idx[2:]
        data = map_coordinates(self.data.T, pix, order=order, mode="nearest")
        if preserve_counts:
            data /= factor ** 2

        return self._init_copy(geom=geom, data=data)

    def downsample(self, factor, preserve_counts=True):
        geom = self.geom.downsample(factor)
        block_size = (factor, factor) + (1,) * len(self.geom.axes)
        data = block_reduce(self.data, block_size[::-1], np.nansum)
        if not preserve_counts:
            data /= factor ** 2

        return self._init_copy(geom=geom, data=data)

    def plot(self, ax=None, fig=None, add_cbar=False, stretch="linear", **kwargs):
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
        from astropy.visualization.wcsaxes.frame import EllipticalFrame

        if not self.geom.is_image:
            raise TypeError("Use .plot_interactive() for Map dimension > 2")

        if fig is None:
            fig = plt.gcf()

        if ax is None:
            if self.geom.is_allsky:
                ax = fig.add_subplot(
                    1, 1, 1, projection=self.geom.wcs, frame_class=EllipticalFrame
                )
            else:
                ax = fig.add_subplot(1, 1, 1, projection=self.geom.wcs)

        data = self.data.astype(float)

        kwargs.setdefault("interpolation", "nearest")
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("cmap", "afmhot")
        norm = simple_norm(data[np.isfinite(data)], stretch)
        kwargs.setdefault("norm", norm)

        caxes = ax.imshow(data, **kwargs)
        cbar = fig.colorbar(caxes, ax=ax, label=str(self.unit)) if add_cbar else None

        if self.geom.is_allsky:
            ax = self._plot_format_allsky(ax)
        else:
            ax = self._plot_format(ax)

        # without this the axis limits are changed when calling scatter
        ax.autoscale(enable=False)
        return fig, ax, cbar

    def _plot_format(self, ax):
        try:
            ax.coords["glon"].set_axislabel("Galactic Longitude")
            ax.coords["glat"].set_axislabel("Galactic Latitude")
        except KeyError:
            ax.coords["ra"].set_axislabel("Right Ascension")
            ax.coords["dec"].set_axislabel("Declination")
        except AttributeError:
            log.info("Can't set coordinate axes. No WCS information available.")
        return ax

    def _plot_format_allsky(self, ax):
        # Remove frame
        ax.coords.frame.set_linewidth(0)

        # Set plot axis limits
        ymax, xmax = self.data.shape
        xmargin, _ = self.geom.coord_to_pix({"lon": 180, "lat": 0})
        _, ymargin = self.geom.coord_to_pix({"lon": 0, "lat": -90})

        ax.set_xlim(xmargin, xmax - xmargin)
        ax.set_ylim(ymargin, ymax - ymargin)

        ax.text(0, ymax, self.geom.coordsys + " coords")

        # Grid and ticks
        glon_spacing, glat_spacing = 45, 15
        lon, lat = ax.coords
        lon.set_ticks(spacing=glon_spacing * u.deg, color="w", alpha=0.8)
        lat.set_ticks(spacing=glat_spacing * u.deg)
        lon.set_ticks_visible(False)

        lon.set_ticklabel(color="w", alpha=0.8)
        lon.grid(alpha=0.2, linestyle="solid", color="w")
        lat.grid(alpha=0.2, linestyle="solid", color="w")
        return ax

    def smooth(self, width, kernel="gauss", **kwargs):
        """
        Smooth the image (works on a 2D image and returns a copy).


        Parameters
        ----------
        width : `~astropy.units.Quantity` or float
            Smoothing width given as quantity or float. If a float is given it
            interpreted as smoothing width in pixels. If an (angular) quantity
            is given it converted to pixels using ``geom.wcs.wcs.cdelt``.
            It corresponds to the standard deviation in case of a Gaussian kernel,
            the radius in case of a disk kernel, and the side length in case
            of a box kernel.
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

        if isinstance(width, u.Quantity):
            width = (width.to("deg") / self.geom.pixel_scales.mean()).value

        smoothed_data = np.empty_like(self.data)

        for img, idx in self.iter_by_image():
            if kernel == "gauss":
                data = gaussian_filter(img, width, **kwargs)
            elif kernel == "disk":
                disk = Tophat2DKernel(width)
                disk.normalize("integral")
                data = convolve(img, disk.array, **kwargs)
            elif kernel == "box":
                data = uniform_filter(img, width, **kwargs)
            else:
                raise ValueError("Invalid kernel: {!r}".format(kernel))
            smoothed_data[idx] = data

        return self._init_copy(data=smoothed_data)

    def convolve(self, kernel, use_fft=True, **kwargs):
        """
        Convolve map with a kernel.

        If the kernel is two dimensional, it is applied to all image planes likewise.
        If the kernel is higher dimensional it must match the map in the number of
        dimensions and the corresponding kernel is selected for every image plane.

        Parameters
        ----------
        kernel : `~gammapy.cube.PSFKernel` or `numpy.ndarray`
            Convolution kernel.
        use_fft : bool
            Use `scipy.signal.fftconvolve` or `scipy.ndimage.convolve`.
        kwargs : dict
            Keyword arguments passed to `scipy.signal.fftconvolve` or
            `scipy.ndimage.convolve`.

        Returns
        -------
        map : `WcsNDMap`
            Convolved map.
        """
        from scipy.signal import fftconvolve
        from scipy.ndimage import convolve
        from ..cube.psf_kernel import PSFKernel

        conv_function = fftconvolve if use_fft else convolve
        convolved_data = np.empty(self.data.shape, dtype=np.float32)
        if use_fft:
            kwargs.setdefault("mode", "same")

        if isinstance(kernel, PSFKernel):
            kmap = kernel.psf_kernel_map
            if not np.allclose(
                self.geom.pixel_scales.deg, kmap.geom.pixel_scales.deg, rtol=1e-5
            ):
                raise ValueError("Pixel size of kernel and map not compatible.")
            kernel = kmap.data

        for img, idx in self.iter_by_image():
            idx = Ellipsis if kernel.ndim == 2 else idx
            convolved_data[idx] = conv_function(img, kernel[idx], **kwargs)

        return self._init_copy(data=convolved_data)

    def cutout(self, position, width, mode="trim"):
        """
        Create a cutout around a given position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Center position of the cutout region.
        width : tuple of `~astropy.coordinates.Angle`
            Angular sizes of the region in (lon, lat) in that specific order.
            If only one value is passed, a square region is extracted.
        mode : {'trim', 'partial', 'strict'}
            Mode option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.

        Returns
        -------
        cutout : `~gammapy.maps.WcsNDMap`
            Cutout map
        """
        width = _check_width(width)
        idx = (0,) * len(self.geom.axes)
        c2d = Cutout2D(
            data=self.data[idx],
            wcs=self.geom.wcs,
            position=position,
            # Cutout2D takes size with order (lat, lon)
            size=width[::-1] * u.deg,
            mode=mode,
        )

        # Create the slices with the non-spatial axis
        cutout_slices = Ellipsis, c2d.slices_original[0], c2d.slices_original[1]

        geom = WcsGeom(c2d.wcs, c2d.shape[::-1], axes=self.geom.axes)
        data = self.data[cutout_slices]

        return self._init_copy(geom=geom, data=data)
