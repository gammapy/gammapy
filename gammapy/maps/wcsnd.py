# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from collections import OrderedDict
import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import astropy.units as u
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import SkyCoord
from astropy.io import fits
from regions import PointSkyRegion, RectangleSkyRegion
from gammapy.extern.skimage import block_reduce
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from gammapy.utils.random import InverseCDFSampler, get_random_state
from gammapy.utils.units import unit_from_fits_image_hdu
from .geom import MapCoord, pix_tuple_to_idx
from .regionnd import RegionGeom, RegionNDMap
from .utils import INVALID_INDEX, interp_to_order
from .wcsmap import WcsGeom, WcsMap

__all__ = ["WcsNDMap"]

log = logging.getLogger(__name__)


class WcsNDMap(WcsMap):
    """WCS map with any number of non-spatial dimensions.

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
    meta : `dict`
        Dictionary to store meta data.
    unit : str or `~astropy.units.Unit`
        The map unit
    """

    def __init__(self, geom, data=None, dtype="float32", meta=None, unit=""):
        # TODO: Figure out how to mask pixels for integer data types

        data_shape = geom.data_shape

        if data is None:
            data = self._make_default_data(geom, data_shape, dtype)

        super().__init__(geom, data, meta, unit)

    @staticmethod
    def _make_default_data(geom, shape_np, dtype):
        # Check whether corners of each image plane are valid

        data = np.zeros(shape_np, dtype=dtype)

        if not geom.is_regular or geom.is_allsky:
            coords = geom.get_coord()
            is_nan = np.isnan(coords.lon)
            data[is_nan] = np.nan

        return data

    @classmethod
    def from_hdu(cls, hdu, hdu_bands=None, format=None):
        """Make a WcsNDMap object from a FITS HDU.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU` or `~astropy.io.fits.ImageHDU`
            The map FITS HDU.
        hdu_bands : `~astropy.io.fits.BinTableHDU`
            The BANDS table HDU.
        format : {'gadf', 'fgst-ccube','fgst-template'}
            FITS format convention.

        Returns
        -------
        map : `WcsNDMap`
            Wcs map
        """
        geom = WcsGeom.from_header(hdu.header, hdu_bands, format=format)
        shape = geom.axes.shape
        shape_wcs = tuple([np.max(geom.npix[0]), np.max(geom.npix[1])])

        meta = cls._get_meta_from_header(hdu.header)
        unit = unit_from_fits_image_hdu(hdu.header)

        # TODO: Should we support extracting slices?
        if isinstance(hdu, fits.BinTableHDU):
            map_out = cls(geom, meta=meta, unit=unit)
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
            if "mask" in hdu.name.lower():
                data = hdu.data.astype(bool)
            else:
                data = hdu.data

            map_out = cls(geom=geom, meta=meta, data=data, unit=unit)

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
            raise ValueError(f"Invalid interpolation order: {order!r}")

    def _interp_by_pix_linear_grid(self, pix, order=1, fill_value=None):
        # TODO: Cache interpolator
        method_lookup = {0: "nearest", 1: "linear"}
        try:
            method = method_lookup[order]
        except KeyError:
            raise ValueError(f"Invalid interpolation order: {order!r}")

        grid_pix = [np.arange(n, dtype=float) for n in self.data.shape[::-1]]

        if np.any(np.isfinite(self.data)):
            data = self.data.copy().T
            data[~np.isfinite(data)] = 0.0
        else:
            data = self.data.T

        fn = ScaledRegularGridInterpolator(
            grid_pix, data, fill_value=fill_value, bounds_error=False, method=method
        )
        return fn(tuple(pix), clip=False)

    def _interp_by_pix_map_coordinates(self, pix, order=1):
        pix = tuple(
            [
                np.array(x, ndmin=1)
                if not isinstance(x, np.ndarray) or x.ndim == 0
                else x
                for x in pix
            ]
        )
        return scipy.ndimage.map_coordinates(
            self.data.T, pix, order=order, mode="nearest"
        )

    def _interp_by_coord_griddata(self, coords, interp=None):
        order = interp_to_order(interp)
        method_lookup = {0: "nearest", 1: "linear", 3: "cubic"}
        method = method_lookup.get(order, None)
        if method is None:
            raise ValueError(f"Invalid interp: {interp!r}")

        grid_coords = tuple(self.geom.get_coord(flat=True))
        data = self.data[np.isfinite(self.data)]
        vals = scipy.interpolate.griddata(
            grid_coords, data, tuple(coords), method=method
        )

        m = ~np.isfinite(vals)
        if np.any(m):
            vals_fill = scipy.interpolate.griddata(
                grid_coords, data, tuple([c[m] for c in coords]), method="nearest"
            )
            vals[m] = vals_fill

        return vals

    def fill_by_idx(self, idx, weights=None):
        idx = pix_tuple_to_idx(idx)
        msk = np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)
        idx = [t[msk] for t in idx]

        if weights is not None:
            if isinstance(weights, u.Quantity):
                weights = weights.to_value(self.unit)
            weights = weights[msk]

        idx = np.ravel_multi_index(idx, self.data.T.shape)
        idx, idx_inv = np.unique(idx, return_inverse=True)
        weights = np.bincount(idx_inv, weights=weights).astype(self.data.dtype)
        self.data.T.flat[idx] += weights

    def set_by_idx(self, idx, vals):
        idx = pix_tuple_to_idx(idx)
        self.data.T[idx] = vals

    def pad(self, pad_width, mode="constant", cval=0, order=1):
        if np.isscalar(pad_width):
            pad_width = (pad_width, pad_width)

        if len(pad_width) == 2:
            pad_width += (0,) * (self.geom.ndim - 2)

        geom = self.geom.pad(pad_width[:2])
        if self.geom.is_regular and mode != "interp":
            return self._pad_np(geom, pad_width, mode, cval)
        else:
            return self._pad_coadd(geom, pad_width, mode, cval, order)

    def _pad_np(self, geom, pad_width, mode, cval):
        """Pad a map using ``numpy.pad``.

        This method only works for regular geometries but should be more
        efficient when working with large maps.
        """
        kwargs = {}
        if mode == "constant":
            kwargs["constant_values"] = cval

        pad_width = [(t, t) for t in pad_width]
        data = np.pad(self.data, pad_width[::-1], mode, **kwargs)
        return self._init_copy(geom=geom, data=data)

    def _pad_coadd(self, geom, pad_width, mode, cval, order):
        """Pad a map manually by coadding the original map with the new map."""
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
            raise ValueError(f"Invalid mode: {mode!r}")

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

    def upsample(self, factor, order=0, preserve_counts=True, axis_name=None):
        geom = self.geom.upsample(factor, axis_name=axis_name)
        idx = geom.get_idx()

        if axis_name is None:
            pix = (
                (idx[0] - 0.5 * (factor - 1)) / factor,
                (idx[1] - 0.5 * (factor - 1)) / factor,
            ) + idx[2:]
        else:
            pix = list(idx)
            idx_ax = self.geom.axes.index(axis_name)
            pix[idx_ax] = (pix[idx_ax] - 0.5 * (factor - 1)) / factor

        if preserve_counts:
            data = self.data / self.geom.bin_volume().value
        else:
            data = self.data

        data = scipy.ndimage.map_coordinates(
            data.T, tuple(pix), order=order, mode="nearest"
        )

        if preserve_counts:
            data *= geom.bin_volume().value

        return self._init_copy(geom=geom, data=data.astype(self.data.dtype))

    def downsample(self, factor, preserve_counts=True, axis_name=None, weights=None):
        geom = self.geom.downsample(factor, axis_name=axis_name)

        if axis_name is None:
            block_size = (1,) * len(self.geom.axes) + (factor, factor)
        else:
            block_size = [1] * self.data.ndim
            idx = self.geom.axes.index_data(axis_name)
            block_size[idx] = factor

        func = np.nansum if preserve_counts else np.nanmean

        if weights is None:
            weights = 1
        else:
            weights = weights.data

        data = block_reduce(self.data * weights, tuple(block_size), func=func)
        return self._init_copy(geom=geom, data=data.astype(self.data.dtype))

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

        if not self.geom.is_flat:
            raise TypeError("Use .plot_interactive() for Map dimension > 2")

        if fig is None:
            fig = plt.gcf()

        if ax is None:
            if self.geom.projection in ["AIT"]:
                ax = fig.add_subplot(
                    1, 1, 1, projection=self.geom.wcs, frame_class=EllipticalFrame
                )
            else:
                ax = fig.add_subplot(1, 1, 1, projection=self.geom.wcs)

        if self.geom.is_image:
            data = self.data.astype(float)
        else:
            axis = tuple(np.arange(len(self.geom.axes)))
            data = np.squeeze(self.data, axis=axis).astype(float)

        kwargs.setdefault("interpolation", "nearest")
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("cmap", "afmhot")

        norm = simple_norm(data[np.isfinite(data)], stretch)
        kwargs.setdefault("norm", norm)

        im = ax.imshow(data, **kwargs)

        cbar = fig.colorbar(im, ax=ax, label=str(self.unit)) if add_cbar else None

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
        xmin, _ = self.geom.coord_to_pix({"lon": 180, "lat": 0})
        xmax, _ = self.geom.coord_to_pix({"lon": -180, "lat": 0})

        _, ymin = self.geom.coord_to_pix({"lon": 0, "lat": -90})
        _, ymax = self.geom.coord_to_pix({"lon": 0, "lat": 90})

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.text(0, ymax, self.geom.frame + " coords")

        # Grid and ticks
        glon_spacing, glat_spacing = 45, 15
        lon, lat = ax.coords
        lon.set_ticks(spacing=glon_spacing * u.deg, color="w", alpha=0.8)
        lat.set_ticks(spacing=glat_spacing * u.deg)
        lon.set_ticks_visible(False)

        lon.set_major_formatter("d")
        lat.set_major_formatter("d")

        lon.set_ticklabel(color="w", alpha=0.8)
        lon.grid(alpha=0.2, linestyle="solid", color="w")
        lat.grid(alpha=0.2, linestyle="solid", color="w")
        return ax

    def smooth(self, width, kernel="gauss", **kwargs):
        """Smooth the map.

        Iterates over 2D image planes, processing one at a time.

        Parameters
        ----------
        width : `~astropy.units.Quantity`, str or float
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
        if isinstance(width, (u.Quantity, str)):
            width = u.Quantity(width) / self.geom.pixel_scales.mean()
            width = width.to_value("")

        smoothed_data = np.empty(self.data.shape, dtype=float)

        for img, idx in self.iter_by_image():
            img = img.astype(float)
            if kernel == "gauss":
                data = scipy.ndimage.gaussian_filter(img, width, **kwargs)
            elif kernel == "disk":
                disk = Tophat2DKernel(width)
                disk.normalize("integral")
                data = scipy.ndimage.convolve(img, disk.array, **kwargs)
            elif kernel == "box":
                data = scipy.ndimage.uniform_filter(img, width, **kwargs)
            else:
                raise ValueError(f"Invalid kernel: {kernel!r}")
            smoothed_data[idx] = data

        return self._init_copy(data=smoothed_data)

    def to_region_nd_map(self, region=None, func=np.nansum, weights=None):
        """Get region ND map in a given region.

        By default the whole map region is considered.

        Parameters
        ----------
        region: `~regions.Region` or `~astropy.coordinates.SkyCoord`
             Region.
        func : numpy.func
            Function to reduce the data. Default is np.nansum.
            For boolean Map, use np.any or np.all.
        weights : `WcsNDMap`
            Array to be used as weights. The geometry must be equivalent.

        Returns
        -------
        spectrum : `~gammapy.maps.RegionNDMap`
            Spectrum in the given region.
        """
        if isinstance(region, SkyCoord):
            region = PointSkyRegion(region)
        elif region is None:
            width, height = self.geom.width
            region = RectangleSkyRegion(
                center=self.geom.center_skydir, width=width[0], height=height[0]
            )

        if weights is not None:
            if not self.geom == weights.geom:
                raise ValueError("Incompatible spatial geoms between map and weights")

        geom = RegionGeom(region=region, axes=self.geom.axes, wcs=self.geom.wcs)

        if isinstance(region, PointSkyRegion):
            coords = geom.get_coord()
            data = self.get_by_coord(coords=coords)
            if weights is not None:
                data *= weights.get_by_coord(coords=coords)
        else:
            cutout = self.cutout(position=geom.center_skydir, width=geom.width)

            if weights is not None:
                weights_cutout = weights.cutout(
                    position=geom.center_skydir, width=geom.width
                )
                cutout.data *= weights_cutout.data

            mask = cutout.geom.to_image().region_mask([region])
            idx_y, idx_x = np.where(mask)
            data = func(cutout.data[..., idx_y, idx_x], axis=-1)

        return RegionNDMap(geom=geom, data=data, unit=self.unit)

    def get_spectrum(self, region=None, func=np.nansum, weights=None):
        """Extract spectrum in a given region.

        The spectrum can be computed by summing (or, more generally, applying ``func``)
        along the spatial axes in each energy bin. This occurs only inside the ``region``,
        which by default is assumed to be the whole spatial extension of the map.

        Parameters
        ----------
        region: `~regions.Region`
             Region (pixel or sky regions accepted).
        func : numpy.func
            Function to reduce the data. Default is np.nansum.
            For a boolean Map, use np.any or np.all.
        weights : `WcsNDMap`
            Array to be used as weights. The geometry must be equivalent.

        Returns
        -------
        spectrum : `~gammapy.maps.RegionNDMap`
            Spectrum in the given region.
        """
        has_energy_axis = ("energy" in self.geom.axes.names) ^ (
            "energy_true" in self.geom.axes.names
        )

        if not has_energy_axis:
            raise ValueError("Energy axis required")

        return self.to_region_nd_map(region=region, func=func, weights=weights)

    def convolve(self, kernel, use_fft=True, **kwargs):
        """
        Convolve map with a kernel.

        If the kernel is two dimensional, it is applied to all image planes likewise.
        If the kernel is higher dimensional it must match the map in the number of
        dimensions and the corresponding kernel is selected for every image plane.

        Parameters
        ----------
        kernel : `~gammapy.irf.PSFKernel` or `numpy.ndarray`
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
        from gammapy.irf import PSFKernel

        conv_function = scipy.signal.fftconvolve if use_fft else scipy.ndimage.convolve

        if use_fft:
            kwargs.setdefault("mode", "same")

        if self.geom.is_image and not isinstance(kernel, PSFKernel):
            if kernel.ndim > 2:
                raise ValueError(
                    "Image convolution with 3D kernel requires a PSFKernel object"
                )

        geom = self.geom.copy()

        if isinstance(kernel, PSFKernel):
            kmap = kernel.psf_kernel_map
            if not np.allclose(
                self.geom.pixel_scales.deg, kmap.geom.pixel_scales.deg, rtol=1e-5
            ):
                raise ValueError("Pixel size of kernel and map not compatible.")
            kernel = kmap.data.astype(np.float32)
            if self.geom.is_image:
                geom = geom.to_cube([kmap.geom.axes[0]])

        convolved_data = np.empty(geom.data_shape, dtype=np.float32)

        shape_axes_kernel = kernel.shape[slice(0, -2)]

        if len(shape_axes_kernel) > 0:
            if not geom.shape_axes == shape_axes_kernel:
                raise ValueError(
                    f"Incompatible shape between data {geom.shape_axes} and kernel {shape_axes_kernel}"
                )

        if self.geom.is_image and kernel.ndim == 3:
            for idx in range(kernel.shape[0]):
                convolved_data[idx] = conv_function(
                    self.data.astype(np.float32), kernel[idx], **kwargs
                )
        else:
            for img, idx in self.iter_by_image():
                ikern = Ellipsis if kernel.ndim == 2 else idx
                convolved_data[idx] = conv_function(
                    img.astype(np.float32), kernel[ikern], **kwargs
                )
        return self._init_copy(data=convolved_data, geom=geom)

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
        geom_cutout = self.geom.cutout(position=position, width=width, mode=mode)

        slices = geom_cutout.cutout_info["parent-slices"]
        parent_slices = Ellipsis, slices[0], slices[1]

        slices = geom_cutout.cutout_info["cutout-slices"]
        cutout_slices = Ellipsis, slices[0], slices[1]

        data = np.zeros(shape=geom_cutout.data_shape, dtype=self.data.dtype)
        data[cutout_slices] = self.data[parent_slices]

        return self._init_copy(geom=geom_cutout, data=data)

    def stack(self, other, weights=None):
        """Stack cutout into map.

        Parameters
        ----------
        other : `WcsNDMap`
            Other map to stack
        weights : `WcsNDMap`
            Array to be used as weights. The spatial geometry must be equivalent
            to `other` and additional axes must be broadcastable.
        """
        if self.geom == other.geom:
            parent_slices, cutout_slices = None, None
        elif self.geom.is_aligned(other.geom):
            slices = other.geom.cutout_info["parent-slices"]
            parent_slices = Ellipsis, slices[0], slices[1]

            slices = other.geom.cutout_info["cutout-slices"]
            cutout_slices = Ellipsis, slices[0], slices[1]
        else:
            raise ValueError(
                "Can only stack equivalent maps or cutout of the same map."
            )

        data = other.quantity[cutout_slices].to_value(self.unit)

        if weights is not None:
            if not other.geom.to_image() == weights.geom.to_image():
                raise ValueError("Incompatible spatial geoms between map and weights")
            data = data * weights.data[cutout_slices]
        self.data[parent_slices] += data

    def sample_coord(self, n_events, random_state=0):
        """Sample position and energy of events.

        Parameters
        ----------
        n_events : int
            Number of events to sample.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        coords : `~gammapy.maps.MapCoord` object.
            Sequence of coordinates and energies of the sampled events.
        """

        random_state = get_random_state(random_state)
        sampler = InverseCDFSampler(pdf=self.data, random_state=random_state)

        coords_pix = sampler.sample(n_events)
        coords = self.geom.pix_to_coord(coords_pix[::-1])

        # TODO: pix_to_coord should return a MapCoord object
        axes_names = ["lon", "lat"] + self.geom.axes.names
        cdict = OrderedDict(zip(axes_names, coords))

        return MapCoord.create(cdict, frame=self.geom.frame)
