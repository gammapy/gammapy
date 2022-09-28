# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import scipy.interpolate
import scipy.ndimage as ndi
import scipy.signal
import astropy.units as u
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import block_reduce
from regions import (
    PixCoord,
    PointPixelRegion,
    PointSkyRegion,
    RectangleSkyRegion,
    SkyRegion,
)
import matplotlib.pyplot as plt
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from gammapy.utils.units import unit_from_fits_image_hdu
from ..geom import pix_tuple_to_idx
from ..utils import INVALID_INDEX
from .core import WcsMap
from .geom import WcsGeom

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
            if any(x in hdu.name.lower() for x in ["mask", "is_ul", "success"]):
                data = hdu.data.astype(bool)
            else:
                data = hdu.data

            map_out = cls(geom=geom, meta=meta, data=data, unit=unit)

        return map_out

    def get_by_idx(self, idx):
        idx = pix_tuple_to_idx(idx)
        return self.data.T[idx]

    def interp_by_coord(self, coords, method="linear", fill_value=None):

        if self.geom.is_regular:
            pix = self.geom.coord_to_pix(coords)
            return self.interp_by_pix(pix, method=method, fill_value=fill_value)
        else:
            return self._interp_by_coord_griddata(coords, method=method)

    def interp_by_pix(self, pix, method="linear", fill_value=None):
        if not self.geom.is_regular:
            raise ValueError("interp_by_pix only supported for regular geom.")

        grid_pix = [np.arange(n, dtype=float) for n in self.data.shape[::-1]]

        if np.any(np.isfinite(self.data)):
            data = self.data.copy().T
            data[~np.isfinite(data)] = 0.0
        else:
            data = self.data.T

        fn = ScaledRegularGridInterpolator(
            grid_pix, data, fill_value=None, bounds_error=False, method=method
        )
        interp_data = fn(tuple(pix), clip=False)

        if fill_value is not None:
            idxs = self.geom.pix_to_idx(pix, clip=False)
            invalid = np.broadcast_arrays(*[idx == -1 for idx in idxs])
            mask = np.any(invalid, axis=0)
            if not interp_data.shape:
                mask = mask.squeeze()
            interp_data[mask] = fill_value
            interp_data[~np.isfinite(interp_data)] = fill_value

        return interp_data

    def _interp_by_coord_griddata(self, coords, method="linear"):
        grid_coords = self.geom.get_coord()

        data = self.data[np.isfinite(self.data)]
        vals = scipy.interpolate.griddata(
            tuple(grid_coords.flat), data, tuple(coords), method=method
        )

        m = ~np.isfinite(vals)
        if np.any(m):
            vals_fill = scipy.interpolate.griddata(
                tuple(grid_coords.flat),
                data,
                tuple([c[m] for c in coords]),
                method="nearest",
            )
            vals[m] = vals_fill

        return vals

    def _resample_by_idx(self, idx, weights=None, preserve_counts=False):
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

        if not preserve_counts:
            weights /= np.bincount(idx_inv).astype(self.data.dtype)

        self.data.T.flat[idx] += weights

    def fill_by_idx(self, idx, weights=None):
        return self._resample_by_idx(idx, weights=weights, preserve_counts=True)

    def set_by_idx(self, idx, vals):
        idx = pix_tuple_to_idx(idx)
        self.data.T[idx] = vals

    def _pad_spatial(
        self, pad_width, axis_name=None, mode="constant", cval=0, method="linear"
    ):
        if axis_name is None:
            if np.isscalar(pad_width):
                pad_width = (pad_width, pad_width)

            if len(pad_width) == 2:
                pad_width += (0,) * (self.geom.ndim - 2)

            geom = self.geom._pad_spatial(pad_width[:2])
            if self.geom.is_regular and mode != "interp":
                return self._pad_np(geom, pad_width, mode, cval)
            else:
                return self._pad_coadd(geom, pad_width, mode, cval, method)

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

    def _pad_coadd(self, geom, pad_width, mode, cval, method):
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
            vals = self.interp_by_coord(coords, method=method)
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
        if factor == 1 or factor is None:
            return self

        geom = self.geom.upsample(factor, axis_name=axis_name)
        idx = geom.get_idx()

        if axis_name is None:
            pix = (
                (idx[0] - 0.5 * (factor - 1)) / factor,
                (idx[1] - 0.5 * (factor - 1)) / factor,
            ) + idx[2:]
        else:
            pix = list(idx)
            idx_ax = self.geom.axes_names.index(axis_name)
            pix[idx_ax] = (pix[idx_ax] - 0.5 * (factor - 1)) / factor

        if preserve_counts:
            data = self.data / self.geom.bin_volume().value
        else:
            data = self.data

        data = ndi.map_coordinates(data.T, tuple(pix), order=order, mode="nearest")

        if preserve_counts:
            data *= geom.bin_volume().value

        return self._init_copy(geom=geom, data=data.astype(self.data.dtype))

    def downsample(self, factor, preserve_counts=True, axis_name=None, weights=None):
        if factor == 1 or factor is None:
            return self

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
        ax : `~astropy.visualization.wcsaxes.WCSAxes`
            WCS axis object
        """
        from astropy.visualization import simple_norm

        if not self.geom.is_flat:
            raise TypeError("Use .plot_interactive() for Map dimension > 2")

        ax = self._plot_default_axes(ax=ax)

        if fig is None:
            fig = plt.gcf()

        if self.geom.is_image:
            data = self.data.astype(float)
        else:
            axis = tuple(np.arange(len(self.geom.axes)))
            data = np.squeeze(self.data, axis=axis).astype(float)

        kwargs.setdefault("interpolation", "nearest")
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("cmap", "afmhot")

        mask = np.isfinite(data)

        if mask.any():
            min_cut, max_cut = kwargs.pop("vmin", None), kwargs.pop("vmax", None)
            norm = simple_norm(data[mask], stretch, min_cut=min_cut, max_cut=max_cut)
            kwargs.setdefault("norm", norm)

        im = ax.imshow(data, **kwargs)

        if add_cbar:
            fig.colorbar(im, ax=ax, label=str(self.unit))

        if self.geom.is_allsky:
            ax = self._plot_format_allsky(ax)
        else:
            ax = self._plot_format(ax)

        # without this the axis limits are changed when calling scatter
        ax.autoscale(enable=False)
        return ax

    def plot_mask(self, ax=None, **kwargs):
        """Plot the mask as a shaded area

        Parameters
        ----------
        ax : `~astropy.visualization.wcsaxes.WCSAxes`, optional
            WCS axis object to plot on.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.contourf`

        Returns
        -------
        ax : `~astropy.visualization.wcsaxes.WCSAxes`, optional
            WCS axis object to plot on.
        """
        if not self.geom.is_flat:
            raise TypeError("Use .plot_interactive() for Map dimension > 2")

        if not self.is_mask:
            raise ValueError(
                "`.plot_mask()` only supports maps containing boolean values."
            )

        ax = self._plot_default_axes(ax=ax)

        kwargs.setdefault("alpha", 0.5)
        kwargs.setdefault("colors", "w")

        data = np.squeeze(self.data).astype(float)

        ax.contourf(data, levels=[0, 0.5], **kwargs)

        if self.geom.is_allsky:
            ax = self._plot_format_allsky(ax)
        else:
            ax = self._plot_format(ax)

        # without this the axis limits are changed when calling scatter
        ax.autoscale(enable=False)
        return ax

    def _plot_default_axes(self, ax):
        from astropy.visualization.wcsaxes.frame import EllipticalFrame

        if ax is None:
            fig = plt.gcf()
            if self.geom.projection in ["AIT"]:
                ax = fig.add_subplot(
                    1, 1, 1, projection=self.geom.wcs, frame_class=EllipticalFrame
                )
            else:
                ax = fig.add_subplot(1, 1, 1, projection=self.geom.wcs)

        return ax

    @staticmethod
    def _plot_format(ax):
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
        xmin, _ = self.geom.to_image().coord_to_pix({"lon": 180, "lat": 0})
        xmax, _ = self.geom.to_image().coord_to_pix({"lon": -180, "lat": 0})

        _, ymin = self.geom.to_image().coord_to_pix({"lon": 0, "lat": -90})
        _, ymax = self.geom.to_image().coord_to_pix({"lon": 0, "lat": 90})

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

    def to_region_nd_map(
        self, region=None, func=np.nansum, weights=None, method="nearest"
    ):
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
        method : {"nearest", "linear"}
            How to interpolate if a position is given.

        Returns
        -------
        spectrum : `~gammapy.maps.RegionNDMap`
            Spectrum in the given region.
        """
        from gammapy.maps import RegionGeom, RegionNDMap

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
            data = self.interp_by_coord(coords=coords, method=method)
            if weights is not None:
                data *= weights.interp_by_coord(coords=coords, method=method)
            # Casting needed as interp_by_coord transforms boolean
            data = data.astype(self.data.dtype)
        else:
            cutout = self.cutout(position=geom.center_skydir, width=geom.width)

            if weights is not None:
                weights_cutout = weights.cutout(
                    position=geom.center_skydir, width=geom.width
                )
                cutout.data *= weights_cutout.data

            mask = cutout.geom.to_image().region_mask([region]).data
            idx_y, idx_x = np.where(mask)
            data = func(cutout.data[..., idx_y, idx_x], axis=-1)

        return RegionNDMap(geom=geom, data=data, unit=self.unit, meta=self.meta.copy())

    def mask_contains_region(self, region):
        """Check if input region is contained in a boolean mask map.

        Parameters
        ----------
        region: `~regions.SkyRegion` or `~regions.PixRegion`
             Region or list of Regions (pixel or sky regions accepted).

        Returns
        -------
        contained : bool
            Whether region is contained in the mask
        """
        if not self.is_mask:
            raise ValueError("mask_contains_region is only supported for boolean masks")

        if not self.geom.is_image:
            raise ValueError("Method only supported for 2D images")

        if isinstance(region, SkyRegion):
            region = region.to_pixel(self.geom.wcs)

        if isinstance(region, PointPixelRegion):
            lon, lat = region.center.x, region.center.y
            contains = self.get_by_pix((lon, lat))
        else:
            idx = self.geom.get_idx()
            coords_pix = PixCoord(idx[0][self.data], idx[1][self.data])
            contains = region.contains(coords_pix)

        return np.any(contains)

    def binary_erode(self, width, kernel="disk", use_fft=True):
        """Binary erosion of boolean mask removing a given margin

        Parameters
        ----------
        width : `~astropy.units.Quantity`, str or float
            If a float is given it interpreted as width in pixels. If an (angular)
            quantity is given it converted to pixels using ``geom.wcs.wcs.cdelt``.
            The width corresponds to radius in case of a disk kernel, and
            the side length in case of a box kernel.
        kernel : {'disk', 'box'}
            Kernel shape
        use_fft : bool
            Use `scipy.signal.fftconvolve` if True (default)
            and `scipy.ndimage.binary_erosion` otherwise.


        Returns
        -------
        map : `WcsNDMap`
            Eroded mask map

        """
        if not self.is_mask:
            raise ValueError("Binary operations only supported for boolean masks")

        structure = self.geom.binary_structure(width=width, kernel=kernel)

        if use_fft:
            return self.convolve(structure.squeeze(), method="fft") > (
                structure.sum() - 1
            )

        data = ndi.binary_erosion(self.data, structure=structure)
        return self._init_copy(data=data)

    def binary_dilate(self, width, kernel="disk", use_fft=True):
        """Binary dilation of boolean mask adding a given margin

        Parameters
        ----------
        width : tuple of `~astropy.units.Quantity`
            Angular sizes of the margin in (lon, lat) in that specific order.
            If only one value is passed, the same margin is applied in (lon, lat).
        kernel : {'disk', 'box'}
            Kernel shape
        use_fft : bool
            Use `scipy.signal.fftconvolve` if True (default)
            and `scipy.ndimage.binary_dilation` otherwise.

        Returns
        -------
        map : `WcsNDMap`
            Dilated mask map
        """
        if not self.is_mask:
            raise ValueError("Binary operations only supported for boolean masks")

        structure = self.geom.binary_structure(width=width, kernel=kernel)

        if use_fft:
            return self.convolve(structure.squeeze(), method="fft") > 1

        data = ndi.binary_dilation(self.data, structure=structure)
        return self._init_copy(data=data)

    def convolve(self, kernel, method="fft", mode="same"):
        """Convolve map with a kernel.

        If the kernel is two dimensional, it is applied to all image planes likewise.
        If the kernel is higher dimensional, it should either match the map in the number of
        dimensions or the map must be an image (no non-spatial axes). In that case, the
        corresponding kernel is selected and applied to every image plane or to the single
        input image respectively.

        Parameters
        ----------
        kernel : `~gammapy.irf.PSFKernel` or `numpy.ndarray`
            Convolution kernel.
        method : str
            The method used by `~scipy.signal.convolve`. Default is 'fft'.
        mode : str
            The convolution mode used by `~scipy.signal.convolve`. Default is 'same'.

        Returns
        -------
        map : `WcsNDMap`
            Convolved map.
        """
        from gammapy.irf import PSFKernel

        convolve = scipy.signal.convolve

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
                geom = geom.to_cube(kmap.geom.axes)

        if mode == "full":
            pad_width = [0.5 * (width - 1) for width in kernel.shape[-2:]]
            geom = geom.pad(pad_width, axis_name=None)
        elif mode == "valid":
            raise NotImplementedError(
                "WcsNDMap.convolve: mode='valid' is not supported."
            )

        data = np.empty(geom.data_shape, dtype=np.float32)

        shape_axes_kernel = kernel.shape[slice(0, -2)]

        if len(shape_axes_kernel) > 0:
            if not geom.shape_axes == shape_axes_kernel:
                raise ValueError(
                    f"Incompatible shape between data {geom.shape_axes}"
                    " and kernel {shape_axes_kernel}"
                )

        if self.geom.is_image and kernel.ndim == 3:
            for idx in range(kernel.shape[0]):
                data[idx] = convolve(
                    self.data.astype(np.float32), kernel[idx], method=method, mode=mode
                )
        else:
            for img, idx in self.iter_by_image_data():
                ikern = Ellipsis if kernel.ndim == 2 else idx
                data[idx] = convolve(
                    img.astype(np.float32), kernel[ikern], method=method, mode=mode
                )
        return self._init_copy(data=data, geom=geom)

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
            Keyword arguments passed to `~ndi.uniform_filter`
            ('box'), `~ndi.gaussian_filter` ('gauss') or
            `~ndi.convolve` ('disk').

        Returns
        -------
        image : `WcsNDMap`
            Smoothed image (a copy, the original object is unchanged).
        """
        if isinstance(width, (u.Quantity, str)):
            width = u.Quantity(width) / self.geom.pixel_scales.mean()
            width = width.to_value("")

        smoothed_data = np.empty(self.data.shape, dtype=float)

        for img, idx in self.iter_by_image_data():
            img = img.astype(float)
            if kernel == "gauss":
                data = ndi.gaussian_filter(img, width, **kwargs)
            elif kernel == "disk":
                disk = Tophat2DKernel(width)
                disk.normalize("integral")
                data = ndi.convolve(img, disk.array, **kwargs)
            elif kernel == "box":
                data = ndi.uniform_filter(img, width, **kwargs)
            else:
                raise ValueError(f"Invalid kernel: {kernel!r}")
            smoothed_data[idx] = data

        return self._init_copy(data=smoothed_data)

    def cutout(self, position, width, mode="trim", odd_npix=False):
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
        odd_npix : bool
            Force width to odd number of pixels.

        Returns
        -------
        cutout : `~gammapy.maps.WcsNDMap`
            Cutout map
        """
        geom_cutout = self.geom.cutout(
            position=position, width=width, mode=mode, odd_npix=odd_npix
        )
        cutout_info = geom_cutout.cutout_slices(self.geom, mode=mode)

        slices = cutout_info["parent-slices"]
        parent_slices = Ellipsis, slices[0], slices[1]

        slices = cutout_info["cutout-slices"]
        cutout_slices = Ellipsis, slices[0], slices[1]

        data = np.zeros(shape=geom_cutout.data_shape, dtype=self.data.dtype)
        data[cutout_slices] = self.data[parent_slices]

        return self._init_copy(geom=geom_cutout, data=data)

    def stack(self, other, weights=None, nan_to_num=True):
        """Stack cutout into map.

        Parameters
        ----------
        other : `WcsNDMap`
            Other map to stack
        weights : `WcsNDMap`
            Array to be used as weights. The spatial geometry must be equivalent
            to `other` and additional axes must be broadcastable.
        nan_to_num: bool
            Non-finite values are replaced by zero if True (default).
        """
        if self.geom == other.geom:
            parent_slices, cutout_slices = None, None
        elif self.geom.is_aligned(other.geom):
            cutout_slices = other.geom.cutout_slices(self.geom)

            slices = cutout_slices["parent-slices"]
            parent_slices = Ellipsis, slices[0], slices[1]

            slices = cutout_slices["cutout-slices"]
            cutout_slices = Ellipsis, slices[0], slices[1]
        else:
            raise ValueError(
                "Can only stack equivalent maps or cutout of the same map."
            )

        data = other.quantity[cutout_slices].to_value(self.unit)
        if nan_to_num:
            not_finite = ~np.isfinite(data)
            if np.any(not_finite):
                data = data.copy()
                data[not_finite] = 0
        if weights is not None:
            if not other.geom.to_image() == weights.geom.to_image():
                raise ValueError("Incompatible spatial geoms between map and weights")
            data = data * weights.data[cutout_slices]
        self.data[parent_slices] += data
