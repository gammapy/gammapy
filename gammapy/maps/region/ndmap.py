from itertools import product
import numpy as np
from scipy.ndimage.measurements import label as ndi_label
from astropy import units as u
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.table import Table
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from gammapy.utils.interpolation import ScaledRegularGridInterpolator, StatProfileScale
from gammapy.utils.scripts import make_path
from ..axes import MapAxes
from ..core import Map
from ..geom import pix_tuple_to_idx
from ..region import RegionGeom
from ..utils import INVALID_INDEX

__all__ = ["RegionNDMap"]


class RegionNDMap(Map):
    """N-dimensional region map.
    A `~RegionNDMap` owns a `~RegionGeom` instance as well as a data array
    containing the values associated to that region in the sky along the non-spatial
    axis, usually an energy axis. The spatial dimensions of a `~RegionNDMap`
    are reduced to a single spatial bin with an arbitrary shape,
    and any extra dimensions are described by an arbitrary number of non-spatial axes.

    Parameters
    ----------
    geom : `~gammapy.maps.RegionGeom`
        Region geometry object.
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
        if data is None:
            data = np.zeros(geom.data_shape, dtype=dtype)

        if meta is None:
            meta = {}

        self._geom = geom
        self.data = data
        self.meta = meta
        self._unit = u.Unit(unit)

    def plot(self, ax=None, axis_name=None, **kwargs):
        """Plot the data contained in region map along the non-spatial axis.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axis`
            Axis used for plotting
        axis_name : str
            Which axis to plot on the x axis. Extra axes will be plotted as
            additional lines.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.errorbar`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axis`
            Axis used for plotting
        """
        ax = ax or plt.gca()

        if axis_name is None:
            if self.geom.axes.is_unidimensional:
                axis_name = self.geom.axes.primary_axis.name
            else:
                raise ValueError(
                    "Plotting a region map with multiple extra axes requires "
                    "specifying the 'axis_name' keyword."
                )

        axis = self.geom.axes[axis_name]

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("ls", "None")
        kwargs.setdefault("xerr", axis.as_plot_xerr)

        yerr_nd, yerr = kwargs.pop("yerr", None), None
        uplims_nd, uplims = kwargs.pop("uplims", None), None
        label_default = kwargs.pop("label", None)

        labels = product(
            *[ax.as_plot_labels for ax in self.geom.axes if ax.name != axis.name]
        )

        for label_axis, (idx, quantity) in zip(
            labels, self.iter_by_axis_data(axis_name=axis.name)
        ):
            if isinstance(yerr_nd, tuple):
                yerr = yerr_nd[0][idx], yerr_nd[1][idx]
            elif isinstance(yerr_nd, np.ndarray):
                yerr = yerr_nd[idx]

            if uplims_nd is not None:
                uplims = uplims_nd[idx]

            label = " ".join(label_axis) if label_default is None else label_default

            with quantity_support():
                ax.errorbar(
                    x=axis.as_plot_center,
                    y=quantity,
                    yerr=yerr,
                    uplims=uplims,
                    label=label,
                    **kwargs,
                )

        axis.format_plot_xaxis(ax=ax)

        if "energy" in axis_name:
            ax.set_yscale("log", nonpositive="clip")

        if len(self.geom.axes) > 1:
            plt.legend()

        return ax

    def plot_hist(self, ax=None, **kwargs):
        """Plot as histogram.

        kwargs are forwarded to `~matplotlib.pyplot.hist`

        Parameters
        ----------
        ax : `~matplotlib.axis` (optional)
            Axis instance to be used for the plot
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.hist`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axis`
            Axis used for plotting
        """
        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("histtype", "step")
        kwargs.setdefault("lw", 1)

        if not self.geom.axes.is_unidimensional:
            raise ValueError("Plotting is only supported for unidimensional maps")

        axis = self.geom.axes[0]

        with quantity_support():
            weights = self.data[:, 0, 0]
            ax.hist(
                axis.as_plot_center, bins=axis.as_plot_edges, weights=weights, **kwargs
            )

        if not self.unit.is_unity():
            ax.set_ylabel(f"Data [{self.unit}]")

        axis.format_plot_xaxis(ax=ax)
        ax.set_yscale("log")
        return ax

    def plot_interactive(self):
        raise NotImplementedError(
            "Interactive plotting currently not support for RegionNDMap"
        )

    def plot_region(self, ax=None, **kwargs):
        """Plot region

        Parameters
        ----------
        ax : `~astropy.visualization.WCSAxes`
            Axes to plot on. If no axes are given,
            the region is shown using the minimal
            equivalent WCS geometry.
        **kwargs : dict
            Keyword arguments forwarded to `~regions.PixelRegion.as_artist`
        """
        ax = self.geom.plot_region(ax, **kwargs)
        return ax

    def plot_mask(self, ax=None, **kwargs):
        """Plot the mask as a shaded area in a xmin-xmax range

        Parameters
        ----------
        ax : `~matplotlib.axis`
            Axis instance to be used for the plot.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.axvspan`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axis`
            Axis used for plotting
        """
        if not self.is_mask:
            raise ValueError("This is not a mask and cannot be plotted")

        kwargs.setdefault("color", "k")
        kwargs.setdefault("alpha", 0.05)
        kwargs.setdefault("label", "mask")

        ax = plt.gca() if ax is None else ax

        edges = self.geom.axes["energy"].edges.reshape((-1, 1, 1))

        labels, nlabels = ndi_label(self.data)

        for idx in range(1, nlabels + 1):
            mask = labels == idx
            xmin = edges[:-1][mask].min().value
            xmax = edges[1:][mask].max().value
            ax.axvspan(xmin, xmax, **kwargs)

        return ax

    @classmethod
    def create(
        cls,
        region,
        axes=None,
        dtype="float32",
        meta=None,
        unit="",
        wcs=None,
        binsz_wcs="0.1deg",
        data=None,
    ):
        """Create an empty region map object.

        Parameters
        ----------
        region : str or `~regions.SkyRegion`
            Region specification
        axes : list of `MapAxis`
            Non spatial axes.
        dtype : str
            Data type, default is 'float32'
        unit : str or `~astropy.units.Unit`
            Data unit.
        meta : `dict`
            Dictionary to store meta data.
        wcs : `~astropy.wcs.WCS`
            WCS projection to use for local projections of the region
        binsz_wcs: `~astropy.units.Quantity` or str
            Bin size used for the default WCS, if wcs=None.
        data : `~numpy.ndarray`
            Data array

        Returns
        -------
        map : `RegionNDMap`
            Region map
        """
        geom = RegionGeom.create(region=region, axes=axes, wcs=wcs, binsz_wcs=binsz_wcs)
        return cls(geom=geom, dtype=dtype, unit=unit, meta=meta, data=data)

    def downsample(self, factor, preserve_counts=True, axis_name=None, weights=None):
        """Downsample the non-spatial dimension by a given factor.

        By default the first axes is downsampled.

        Parameters
        ----------
        factor : int
            Downsampling factor.
        preserve_counts : bool
            Preserve the integral over each bin.  This should be true
            if the map is an integral quantity (e.g. counts) and false if
            the map is a differential quantity (e.g. intensity).
        axis_name : str
            Which axis to downsample. Default is "energy".
        weights : `RegionNDMap`
            Contains the weights to apply to the axis to reduce. Default
            is just weighs of one.

        Returns
        -------
        map : `RegionNDMap`
            Downsampled region map.
        """
        if axis_name is None:
            axis_name = self.geom.axes[0].name

        geom = self.geom.downsample(factor=factor, axis_name=axis_name)

        block_size = [1] * self.data.ndim
        idx = self.geom.axes.index_data(axis_name)
        block_size[idx] = factor

        if weights is None:
            weights = 1
        else:
            weights = weights.data

        func = np.nansum if preserve_counts else np.nanmean
        if self.is_mask:
            func = np.all
        data = block_reduce(self.data * weights, tuple(block_size), func=func)

        return self._init_copy(geom=geom, data=data)

    def upsample(self, factor, order=0, preserve_counts=True, axis_name=None):
        """Upsample the non-spatial dimension by a given factor.

        By default the first axes is upsampled.

        Parameters
        ----------
        factor : int
            Upsampling factor.
        order : int
            Order of the interpolation used for upsampling.
        preserve_counts : bool
            Preserve the integral over each bin.  This should be true
            if the RegionNDMap is an integral quantity (e.g. counts) and false if
            the RegionNDMap is a differential quantity (e.g. intensity).
        axis_name : str
            Which axis to upsample. Default is "energy".

        Returns
        -------
        map : `RegionNDMap`
            Upsampled region map.
        """
        if axis_name is None:
            axis_name = self.geom.axes[0].name

        geom = self.geom.upsample(factor=factor, axis_name=axis_name)
        data = self.interp_by_coord(geom.get_coord())

        if preserve_counts:
            data /= factor

        return self._init_copy(geom=geom, data=data)

    def iter_by_axis_data(self, axis_name):
        """Iterate data by axis

        Parameters
        ----------
        axis_name : str
            Axis name

        Returns
        -------
        idx, data : tuple, `~astropy.units.Quantity`
            Data and index
        """
        idx_axis = self.geom.axes.index_data(axis_name)
        shape = list(self.data.shape)
        shape[idx_axis] = 1

        for idx in np.ndindex(*shape):
            idx = list(idx)
            idx[idx_axis] = slice(None)
            yield tuple(idx), self.quantity[tuple(idx)]

    def _resample_by_idx(self, idx, weights=None, preserve_counts=False):
        # inherited docstring
        # TODO: too complex, simplify!
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

    def get_by_idx(self, idxs):
        # inherited docstring
        return self.data[idxs[::-1]]

    def interp_by_coord(self, coords, **kwargs):
        """Interpolate map values at the given map coordinates.

        Parameters
        ----------
        coords : tuple, dict or `~gammapy.maps.MapCoord`
            Coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.
            "lon" and "lat" are optional and will be taken at the center
            of the region by default.
        method : {"linear", "nearest"}
            Method to interpolate data values. By default linear
            interpolation is performed.
        fill_value : None or float value
            The value to use for points outside of the interpolation domain.
            If None, values outside the domain are extrapolated.
        values_scale : {"lin", "log", "sqrt"}
            Optional value scaling.

        Returns
        -------
        vals : `~numpy.ndarray`
            Interpolated pixel values.
        """
        pix = self.geom.coord_to_pix(coords=coords)
        return self.interp_by_pix(pix, **kwargs)

    def interp_by_pix(self, pix, **kwargs):
        # inherited docstring
        grid_pix = [np.arange(n, dtype=float) for n in self.data.shape[::-1]]

        if np.any(np.isfinite(self.data)):
            data = self.data.copy().T
            data[~np.isfinite(data)] = 0.0
        else:
            data = self.data.T

        scale = kwargs.get("values_scale", "lin")

        if scale == "stat-profile":
            axis = 2 + self.geom.axes.index("norm")
            kwargs["values_scale"] = StatProfileScale(axis=axis)

        fn = ScaledRegularGridInterpolator(grid_pix, data, **kwargs)
        return fn(tuple(pix), clip=False)

    def set_by_idx(self, idx, value):
        # inherited docstring
        self.data[idx[::-1]] = value

    @classmethod
    def read(cls, filename, format="gadf", ogip_column=None, hdu=None):
        """Read from file.

        Parameters
        ----------
        filename : `pathlib.Path` or str
            Filename.
        format : {"gadf", "ogip", "ogip-arf"}
            Which format to use.
        ogip_column : {None, "COUNTS", "QUALITY", "BACKSCAL"}
            If format 'ogip' is chosen which table hdu column to read.
        hdu : str
            Name or index of the HDU with the map data.

        Returns
        -------
        region_map : `RegionNDMap`
            Region nd map
        """
        filename = make_path(filename)
        with fits.open(filename, memmap=False) as hdulist:
            return cls.from_hdulist(
                hdulist, format=format, ogip_column=ogip_column, hdu=hdu
            )

    def write(self, filename, overwrite=False, format="gadf", hdu="SKYMAP"):
        """Write map to file

        Parameters
        ----------
        filename : `pathlib.Path` or str
            Filename.
        format : {"gadf", "ogip", "ogip-sherpa", "ogip-arf", "ogip-arf-sherpa"}
            Which format to use.
        overwrite : bool
            Overwrite existing files?
        """
        filename = make_path(filename)
        self.to_hdulist(format=format, hdu=hdu).writeto(filename, overwrite=overwrite)

    def to_hdulist(self, format="gadf", hdu="SKYMAP", hdu_bands=None, hdu_region=None):
        """Convert to `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        format : {"gadf", "ogip", "ogip-sherpa", "ogip-arf", "ogip-arf-sherpa"}
            Format specification
        hdu : str
            Name of the HDU with the map data, used for "gadf" format.
        hdu_bands : str
            Name or index of the HDU with the BANDS table, used for "gadf" format.
        hdu_region : str
            Name or index of the HDU with the region table.

        Returns
        -------
        hdulist : `~astropy.fits.HDUList`
            HDU list
        """
        hdulist = fits.HDUList()
        table = self.to_table(format=format)

        if hdu_bands is None:
            hdu_bands = f"{hdu.upper()}_BANDS"
        if hdu_region is None:
            hdu_region = f"{hdu.upper()}_REGION"

        if format in ["ogip", "ogip-sherpa", "ogip-arf", "ogip-arf-sherpa"]:
            hdulist.append(fits.BinTableHDU(table))
        elif format == "gadf":
            table.meta.update(self.geom.axes.to_header())
            hdulist.append(fits.BinTableHDU(table, name=hdu))
        else:
            raise ValueError(f"Unsupported format '{format}'")

        if format in ["ogip", "ogip-sherpa", "gadf"]:
            hdulist_geom = self.geom.to_hdulist(
                format=format, hdu_bands=hdu_bands, hdu_region=hdu_region
            )
            hdulist.extend(hdulist_geom[1:])

        return hdulist

    @classmethod
    def from_table(cls, table, format="", colname=None):
        """Create region map from table

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table with input data
        format : {"gadf-sed", "lightcurve", "profile"}
            Format to use
        colname : str
            Column name to take the data from.

        Returns
        -------
        region_map : `RegionNDMap`
            Region map
        """
        if format == "gadf-sed":
            if colname is None:
                raise ValueError("Column name required")

            axes = MapAxes.from_table(table=table, format=format)

            if colname == "stat_scan":
                names = ["norm", "energy"]
            # TODO: this is not officially supported by GADF...
            elif colname in ["counts", "npred", "npred_excess"]:
                names = ["dataset", "energy"]
            else:
                names = ["energy"]

            axes = axes[names]
            data = table[colname].data
            unit = table[colname].unit or ""
        elif format == "lightcurve":
            axes = MapAxes.from_table(table=table, format=format)

            if colname == "stat_scan":
                names = ["norm", "energy", "time"]
            # TODO: this is not officially supported by GADF...
            elif colname in ["counts", "npred", "npred_excess"]:
                names = ["dataset", "energy", "time"]
            else:
                names = ["energy", "time"]

            axes = axes[names]
            data = table[colname].data
            unit = table[colname].unit or ""
        elif format == "profile":
            axes = MapAxes.from_table(table=table, format=format)

            if colname == "stat_scan":
                names = ["norm", "energy", "projected-distance"]
            # TODO: this is not officially supported by GADF...
            elif colname in ["counts", "npred", "npred_excess"]:
                names = ["dataset", "energy", "projected-distance"]
            else:
                names = ["energy", "projected-distance"]

            axes = axes[names]
            data = table[colname].data
            unit = table[colname].unit or ""
        else:
            raise ValueError(f"Format not supported {format}")

        geom = RegionGeom.create(region=None, axes=axes)
        return cls(geom=geom, data=data, unit=unit, meta=table.meta, dtype=data.dtype)

    @classmethod
    def from_hdulist(cls, hdulist, format="gadf", ogip_column=None, hdu=None, **kwargs):
        """Create from `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list.
        format : {"gadf", "ogip", "ogip-arf"}
            Format specification
        ogip_column : {"COUNTS", "QUALITY", "BACKSCAL"}
            If format 'ogip' is chosen which table hdu column to read.
        hdu : str
            Name or index of the HDU with the map data.

        Returns
        -------
        region_nd_map : `RegionNDMap`
            Region map.
        """
        defaults = {
            "ogip": {"hdu": "SPECTRUM", "column": "COUNTS"},
            "ogip-arf": {"hdu": "SPECRESP", "column": "SPECRESP"},
            "gadf": {"hdu": "SKYMAP", "column": "DATA"},
        }

        if hdu is None:
            hdu = defaults[format]["hdu"]

        if ogip_column is None:
            ogip_column = defaults[format]["column"]

        geom = RegionGeom.from_hdulist(hdulist, format=format, hdu=hdu)

        table = Table.read(hdulist[hdu])
        quantity = table[ogip_column].quantity

        if ogip_column == "QUALITY":
            data, unit = np.logical_not(quantity.value.astype(bool)), ""
        else:
            data, unit = quantity.value, quantity.unit

        return cls(geom=geom, data=data, meta=table.meta, unit=unit, dtype=data.dtype)

    def _pad_spatial(self, *args, **kwargs):
        raise NotImplementedError("Spatial padding is not supported by RegionNDMap")

    def crop(self):
        raise NotImplementedError("Crop is not supported by RegionNDMap")

    def stack(self, other, weights=None, nan_to_num=True):
        """Stack other region map into map.

        Parameters
        ----------
        other : `RegionNDMap`
            Other map to stack
        weights : `RegionNDMap`
            Array to be used as weights. The spatial geometry must be equivalent
            to `other` and additional axes must be broadcastable.
        nan_to_num: bool
            Non-finite values are replaced by zero if True (default).
        """
        data = other.quantity.to_value(self.unit).astype(self.data.dtype)

        # TODO: re-think stacking of regions. Is making the union reasonable?
        # self.geom.union(other.geom)
        if nan_to_num:
            not_finite = ~np.isfinite(data)
            if np.any(not_finite):
                data = data.copy()
                data[not_finite] = 0
        if weights is not None:
            if not other.geom.to_image() == weights.geom.to_image():
                raise ValueError("Incompatible geoms between map and weights")
            data = data * weights.data

        self.data += data

    def to_table(self, format="gadf"):
        """Convert to `~astropy.table.Table`.

        Data format specification: :ref:`gadf:ogip-pha`

        Parameters
        ----------
        format : {"gadf", "ogip", "ogip-arf", "ogip-arf-sherpa"}
            Format specification

        Returns
        -------
        table : `~astropy.table.Table`
            Table
        """
        data = np.nan_to_num(self.quantity[:, 0, 0])

        if format == "ogip":
            if len(self.geom.axes) > 1:
                raise ValueError(
                    f"Writing to format '{format}' only supports a "
                    f"single energy axis. Got {self.geom.axes.names}"
                )

            energy_axis = self.geom.axes[0]
            energy_axis.assert_name("energy")
            table = Table()
            table["CHANNEL"] = np.arange(energy_axis.nbin, dtype=np.int16)
            table["COUNTS"] = np.array(data, dtype=np.int32)

            # see https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node6.html  # noqa: E501
            table.meta = {
                "EXTNAME": "SPECTRUM",
                "telescop": "unknown",
                "instrume": "unknown",
                "filter": "None",
                "exposure": 0,
                "corrfile": "",
                "corrscal": "",
                "ancrfile": "",
                "hduclass": "OGIP",
                "hduclas1": "SPECTRUM",
                "hduvers": "1.2.1",
                "poisserr": True,
                "chantype": "PHA",
                "detchans": energy_axis.nbin,
                "quality": 0,
                "backscal": 0,
                "grouping": 0,
                "areascal": 1,
            }

        elif format in ["ogip-arf", "ogip-arf-sherpa"]:
            if len(self.geom.axes) > 1:
                raise ValueError(
                    f"Writing to format '{format}' only supports a "
                    f"single energy axis. Got {self.geom.axes.names}"
                )

            energy_axis = self.geom.axes[0]
            table = energy_axis.to_table(format=format)
            table.meta = {
                "EXTNAME": "SPECRESP",
                "telescop": "unknown",
                "instrume": "unknown",
                "filter": "None",
                "hduclass": "OGIP",
                "hduclas1": "RESPONSE",
                "hduclas2": "SPECRESP",
                "hduvers": "1.1.0",
            }

            if format == "ogip-arf-sherpa":
                data = data.to("cm2")

            table["SPECRESP"] = data

        elif format == "gadf":
            table = Table()
            data = self.quantity.flatten()
            table["CHANNEL"] = np.arange(len(data), dtype=np.int16)
            table["DATA"] = data
        else:
            raise ValueError(f"Unsupported format: '{format}'")

        meta = {k: self.meta.get(k, v) for k, v in table.meta.items()}
        table.meta.update(meta)
        return table

    def get_spectrum(self, *args, **kwargs):
        """Return self"""
        return self

    def to_region_nd_map(self, *args, **kwargs):
        return self

    def cutout(self, *args, **kwargs):
        """Return self"""
        return self
