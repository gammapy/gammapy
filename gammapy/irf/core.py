from copy import deepcopy
import abc
import logging
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.utils import lazyproperty
from astropy.table import Table
from gammapy.maps import Map, MapAxes, MapAxis, RegionGeom
from gammapy.utils.interpolation import (
    ScaledRegularGridInterpolator,
    interpolation_scale,
)
from gammapy.utils.integrate import trapz_loglog
from gammapy.utils.scripts import make_path
from .io import IRF_DL3_HDU_SPECIFICATION, IRF_MAP_HDU_SPECIFICATION

log = logging.getLogger(__name__)


class IRF:
    """IRF base class for DL3 instrument response functions

    Parameters
    -----------
    axes : list of `MapAxis` or `MapAxes`
        Axes
    data : `~numpy.ndarray`
        Data
    unit : str or `~astropy.units.Unit`
        Unit
    meta : dict
        Meta data
    """

    default_interp_kwargs = dict(bounds_error=False, fill_value=None,)

    def __init__(self, axes, data=0, unit="", meta=None):
        axes = MapAxes(axes)
        axes.assert_names(self.required_axes)
        self._axes = axes
        self.data = data
        self.unit = unit
        self.meta = meta or {}

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @property
    @abc.abstractmethod
    def required_axes(self):
        pass

    @property
    def is_offset_dependent(self):
        """Whether the IRF depends on offset"""
        return "offset" in self.required_axes

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        """Set data

        Parameters
        ----------
        value : `~astropy.units.Quantity`, array-like
            Data array
        """
        required_shape = self.axes.shape

        if np.isscalar(value):
            value = value * np.ones(required_shape)

        if isinstance(value, u.Quantity):
            raise TypeError("Map data must be a Numpy array. Set unit separately")

        if np.shape(value) != required_shape:
            raise ValueError(
                f"data shape {value.shape} does not match"
                f"axes shape {required_shape}"
            )

        self._data = value

        # reset cached interpolators
        self.__dict__.pop("_interpolate", None)
        self.__dict__.pop("_integrate_rad", None)

    def interp_missing_data(self, axis_name):
        """Interpolate missing data along a given axis"""
        data = self.data.copy()
        values_scale = self.default_interp_kwargs.get("values_scale", "lin")
        scale = interpolation_scale(values_scale)

        axis = self.axes.index(axis_name)
        mask = ~np.isfinite(data) | (data == 0.0)

        coords = np.where(mask)
        xp = np.arange(data.shape[axis])

        for coord in zip(*coords):
            idx = list(coord)
            idx[axis] = slice(None)
            fp = data[tuple(idx)]
            valid = ~mask[tuple(idx)]

            if np.any(valid):
                value = np.interp(
                    x=coord[axis],
                    xp=xp[valid],
                    fp=scale(fp[valid]),
                    left=np.nan,
                    right=np.nan,
                )
                if not np.isnan(value):
                    data[coord] = scale.inverse(value)
        self.data = data  # reset cached values

    @property
    def unit(self):
        """Map unit (`~astropy.units.Unit`)"""
        return self._unit

    @unit.setter
    def unit(self, val):
        self._unit = u.Unit(val)

    @lazyproperty
    def _interpolate(self):
        points = [a.center for a in self.axes]
        points_scale = tuple([a.interp for a in self.axes])
        return ScaledRegularGridInterpolator(
            points,
            self.quantity,
            points_scale=points_scale,
            **self.default_interp_kwargs,
        )

    @property
    def quantity(self):
        """`~astropy.units.Quantity`"""
        return u.Quantity(self.data, unit=self.unit, copy=False)

    @quantity.setter
    def quantity(self, val):
        val = u.Quantity(val, copy=False)
        self.data = val.value
        self.unit = val.unit

    @property
    def axes(self):
        """`MapAxes`"""
        return self._axes

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n\n"
        str_ += f"\taxes  : {self.axes.names}\n"
        str_ += f"\tshape : {self.data.shape}\n"
        str_ += f"\tndim  : {len(self.axes)}\n"
        str_ += f"\tunit  : {self.unit}\n"
        str_ += f"\tdtype : {self.data.dtype}\n"
        return str_.expandtabs(tabsize=2)

    def evaluate(self, method=None, **kwargs):
        """Evaluate IRF

        Parameters
        ----------
        **kwargs : dict
            Coordinates at which to evaluate the IRF
        method : str {'linear', 'nearest'}, optional
            Interpolation method

        Returns
        -------
        array : `~astropy.units.Quantity`
            Interpolated values
        """
        # TODO: change to coord dict?
        non_valid_axis = set(kwargs).difference(self.axes.names)
        if non_valid_axis:
            raise ValueError(
                f"Not a valid coordinate axis {non_valid_axis}"
                f" Choose from: {self.axes.names}"
            )

        coords_default = self.axes.get_coord()

        for key, value in kwargs.items():
            coord = kwargs.get(key, value)
            if coord is not None:
                coords_default[key] = u.Quantity(coord, copy=False)

        return self._interpolate(coords_default.values(), method=method)

    def integrate_log_log(self, axis_name, **kwargs):
        """Integrate along a given axis.

        This method uses log-log trapezoidal integration.

        Parameters
        ----------
        axis_name : str
            Along which axis to integrate.
        **kwargs : dict
            Coordinates at which to evaluate the IRF

        Returns
        -------
        array : `~astropy.units.Quantity`
            Returns 2D array with axes offset
        """
        axis = self.axes.index(axis_name)
        data = self.evaluate(**kwargs, method="linear")
        values = kwargs[axis_name]
        return trapz_loglog(data, values, axis=axis)

    def cumsum(self, axis_name):
        """Compute cumsum along a given axis

        Parameters
        ----------
        axis_name : str
            Along which axis to integrate.

        Returns
        -------
        irf : `~IRF`
            Cumsum IRF

        """
        axis = self.axes[axis_name]
        axis_idx = self.axes.index(axis_name)

        # TODO: the broadcasting should be done by axis.center, axis.bin_width etc.
        shape = [1] * len(self.axes)
        shape[axis_idx] = -1

        values = self.quantity * axis.bin_width.reshape(shape)

        if axis_name == "rad":
            # take Jacobian into account
            values = 2 * np.pi * axis.center.reshape(shape) * values

        data = values.cumsum(axis=axis_idx)

        axis_shifted = MapAxis.from_nodes(
            axis.edges[1:], name=axis.name, interp=axis.interp
        )
        axes = self.axes.replace(axis_shifted)
        return self.__class__(axes=axes, data=data.value, unit=data.unit)

    def integral(self, axis_name, **kwargs):
        """Compute integral along a given axis

        This method uses interpolation of the cumulative sum.

        Parameters
        ----------
        axis_name : str
            Along which axis to integrate.
        **kwargs : dict
            Coordinates at which to evaluate the IRF

        Returns
        -------
        array : `~astropy.units.Quantity`
            Returns 2D array with axes offset

        """
        cumsum = self.cumsum(axis_name=axis_name)
        return cumsum.evaluate(**kwargs)

    def normalize(self, axis_name):
        """Normalise data in place along a given axis.

        Parameters
        ----------
        axis_name : str
            Along which axis to normalize.

        """
        cumsum = self.cumsum(axis_name=axis_name).quantity

        with np.errstate(invalid="ignore", divide="ignore"):
            axis = self.axes.index(axis_name=axis_name)
            normed = self.quantity / cumsum.max(axis=axis, keepdims=True)

        self.quantity = np.nan_to_num(normed)

    @classmethod
    def from_hdulist(cls, hdulist, hdu=None, format="gadf-dl3"):
        """Create from `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdulist : `~astropy.io.HDUList`
            HDU list
        hdu : str
            HDU name
        format : {"gadf-dl3"}
            Format specification

        Returns
        -------
        irf : `IRF`
            IRF class
        """
        if hdu is None:
            hdu = IRF_DL3_HDU_SPECIFICATION[cls.tag]["extname"]

        return cls.from_table(Table.read(hdulist[hdu]), format=format)

    @classmethod
    def read(cls, filename, hdu=None, format="gadf-dl3"):
        """Read from file.

        Parameters
        ----------
        filename : str or `Path`
            Filename
        hdu : str
            HDU name
        format : {"gadf-dl3"}
            Format specification

        Returns
        -------
        irf : `IRF`
            IRF class
        """
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu=hdu)

    @classmethod
    def from_table(cls, table, format="gadf-dl3"):
        """Read from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table with irf data
        format : {"gadf-dl3"}
            Format specification

        Returns
        -------
        irf : `IRF`
            IRF class.
        """
        axes = MapAxes.from_table(table=table, format=format)[cls.required_axes]
        column_name = IRF_DL3_HDU_SPECIFICATION[cls.tag]["column_name"]
        data = table[column_name].quantity[0].transpose()
        return cls(axes=axes, data=data.value, meta=table.meta, unit=data.unit)

    def to_table(self, format="gadf-dl3"):
        """Convert to table

        Parameters
        ----------
        format : {"gadf-dl3"}
            Format specification

        Returns
        -------
        table : `~astropy.table.Table`
            IRF data table
        """
        table = self.axes.to_table(format=format)

        if format == "gadf-dl3":
            table.meta = self.meta.copy()
            spec = IRF_DL3_HDU_SPECIFICATION[self.tag]
            # TODO: add missing required meta data!
            table.meta["HDUCLAS2"] = spec["hduclas2"]
            table[spec["column_name"]] = self.quantity.T[np.newaxis]
        else:
            raise ValueError(f"Not a valid supported format: '{format}'")

        return table

    def to_table_hdu(self, format="gadf-dl3"):
        """Convert to `~astropy.io.fits.BinTableHDU`.

        Parameters
        ----------
        format : {"gadf-dl3"}
            Format specification

        Returns
        -------
        hdu : `~astropy.io.fits.BinTableHDU`
            IRF data table hdu
        """
        name = IRF_DL3_HDU_SPECIFICATION[self.tag]["extname"]
        return fits.BinTableHDU(self.to_table(format=format), name=name)

    def to_hdulist(self, format="gadf-dl3"):
        """"""
        hdu = self.to_table_hdu(format=format)
        return fits.HDUList([fits.PrimaryHDU(), hdu])

    def write(self, filename, *args, **kwargs):
        """Write PSF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_hdulist().writeto(str(make_path(filename)), *args, **kwargs)


class IRFMap:
    """IRF map base class for DL4 instrument response functions"""

    def __init__(self, irf_map, exposure_map):
        self._irf_map = irf_map
        self.exposure_map = exposure_map
        irf_map.geom.axes.assert_names(self.required_axes)

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @property
    @abc.abstractmethod
    def required_axes(self):
        pass

    # TODO: add mask safe to IRFMap as a regular attribute and don't derive it from the data
    @property
    def mask_safe_image(self):
        """Mask safe for the map"""
        mask = self._irf_map > (0 * self._irf_map.unit)
        return mask.reduce_over_axes(func=np.logical_or)

    def to_region_nd_map(self, region):
        """Extract IRFMap in a given region or position

        If a region is given a mean IRF is computed, if a position is given the
        IRF is interpolated.

        Parameters
        ----------
        region : `SkyRegion` or `SkyCoord`
            Region or position where to get the map.

        Returns
        -------
        irf : `IRFMap`
            IRF map with region geometry.
        """
        if region is None:
            region = self._irf_map.geom.center_skydir

        # TODO: compute an exposure weighted mean PSF here
        kwargs = {"region": region, "func": np.nanmean}

        if "energy" in self._irf_map.geom.axes.names:
            kwargs["method"] = "nearest"

        irf_map = self._irf_map.to_region_nd_map(**kwargs)

        if self.exposure_map:
            exposure_map = self.exposure_map.to_region_nd_map(**kwargs)
        else:
            exposure_map = None

        return self.__class__(irf_map, exposure_map=exposure_map)

    def _get_nearest_valid_position(self, position):
        """Get nearest valid position"""
        is_valid = np.nan_to_num(self.mask_safe_image.get_by_coord(position))[0]

        if not is_valid:
            log.warning(
                f"Position {position} is outside "
                "valid IRF map range, using nearest IRF defined within"
            )

            position = self.mask_safe_image.mask_nearest_position(position)
        return position

    @classmethod
    def from_hdulist(
        cls,
        hdulist,
        hdu=None,
        hdu_bands=None,
        exposure_hdu=None,
        exposure_hdu_bands=None,
        format="gadf",
    ):
        """Create from `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdulist : `~astropy.fits.HDUList`
            HDU list.
        hdu : str
            Name or index of the HDU with the IRF map.
        hdu_bands : str
            Name or index of the HDU with the IRF map BANDS table.
        exposure_hdu : str
            Name or index of the HDU with the exposure map data.
        exposure_hdu_bands : str
            Name or index of the HDU with the exposure map BANDS table.
        format : {"gadf", "gtpsf"}
            File format

        Returns
        -------
        irf_map : `IRFMap`
            IRF map.
        """
        if format == "gadf":
            if hdu is None:
                hdu = IRF_MAP_HDU_SPECIFICATION[cls.tag]

            irf_map = Map.from_hdulist(hdulist, hdu=hdu, hdu_bands=hdu_bands, format=format)

            if exposure_hdu is None:
                exposure_hdu = IRF_MAP_HDU_SPECIFICATION[cls.tag] + "_exposure"

            if exposure_hdu in hdulist:
                exposure_map = Map.from_hdulist(
                    hdulist, hdu=exposure_hdu, hdu_bands=exposure_hdu_bands, format=format
                )
            else:
                exposure_map = None
        elif format == "gtpsf":
            rad_axis = MapAxis.from_table_hdu(hdulist["THETA"], format=format)

            table = Table.read(hdulist["PSF"])
            energy_axis_true = MapAxis.from_table(table, format=format)

            geom_psf = RegionGeom.create(region=None, axes=[rad_axis, energy_axis_true])

            psf_map = Map.from_geom(geom=geom_psf, data=table["Psf"].data, unit="sr-1")

            geom_exposure = geom_psf.squash("rad")
            exposure_map = Map.from_geom(
                geom=geom_exposure, data=table["Exposure"].data, unit="cm2 s"
            )
            return cls(psf_map=psf_map, exposure_map=exposure_map)
        else:
            raise ValueError(f"Format {format} not supported")

        return cls(irf_map, exposure_map)

    @classmethod
    def read(cls, filename, format="gadf", hdu=None):
        """Read an IRF_map from file and create corresponding object"

        Parameters
        ----------
        filename : str or `Path`
            File name
        format : {"gadf", "gtpsf"}
            File format
        hdu : str or int
            HDU location

        Returns
        -------
        irf_map : `PSFMap`, `EDispMap` or `EDispKernelMap`
            IRF map

        """
        filename = make_path(filename)
        with fits.open(filename, memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, format=format, hdu=hdu)

    def to_hdulist(self, format="gadf"):
        """Convert to `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        format : {"gadf", "gtpsf"}
            File format

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list.
        """
        if format == "gadf":
            hdu = IRF_MAP_HDU_SPECIFICATION[self.tag]
            hdulist = self._irf_map.to_hdulist(hdu=hdu, format=format)
            exposure_hdu = hdu + "_exposure"

            if self.exposure_map is not None:
                new_hdulist = self.exposure_map.to_hdulist(hdu=exposure_hdu, format=format)
                hdulist.extend(new_hdulist[1:])

        elif format == "gtpsf":
            if not self._irf_map.geom.is_region:
                raise ValueError(
                    "Format 'gtpsf' is only supported for region geometries"
                )

            rad_hdu = self._irf_map.geom.axes["rad"].to_table_hdu(format=format)
            psf_table = self._irf_map.geom.axes["energy_true"].to_table(format=format)

            psf_table["Exposure"] = self.exposure_map.quantity[..., 0, 0].to("cm^2 s")
            psf_table["Psf"] = self._irf_map.quantity[..., 0, 0].to("sr^-1")
            psf_hdu = fits.BinTableHDU(data=psf_table, name="PSF")
            hdulist = fits.HDUList([fits.PrimaryHDU(), rad_hdu, psf_hdu])
        else:
            raise ValueError(f"Format {format} not supported")

        return hdulist

    def write(self, filename, overwrite=False, format="gadf"):
        """Write IRF map to fits

        Parameters
        ----------
        filename : str or `Path`
            Filename to write to
        overwrite : bool
            Whether to overwrite
        format : {"gadf", "gtpsf"}
            File format
        """
        hdulist = self.to_hdulist(format=format)
        hdulist.writeto(str(filename), overwrite=overwrite)

    def stack(self, other, weights=None):
        """Stack IRF map with another one in place.

        Parameters
        ----------
        other : `~gammapy.irf.IRFMap`
            IRF map to be stacked with this one.
        weights : `~gammapy.maps.Map`
            Map with stacking weights.

        """
        if self.exposure_map is None or other.exposure_map is None:
            raise ValueError(
                f"Missing exposure map for {self.__class__.__name__}.stack"
            )

        cutout_info = getattr(other._irf_map.geom, "cutout_info", None)

        if cutout_info is not None:
            slices = cutout_info["parent-slices"]
            parent_slices = Ellipsis, slices[0], slices[1]
        else:
            parent_slices = slice(None)

        self._irf_map.data[parent_slices] *= self.exposure_map.data[parent_slices]
        self._irf_map.stack(other._irf_map * other.exposure_map.data, weights=weights)

        # stack exposure map
        if weights and "energy" in weights.geom.axes.names:
            weights = weights.reduce(
                axis_name="energy", func=np.logical_or, keepdims=True
            )
        self.exposure_map.stack(other.exposure_map, weights=weights)

        with np.errstate(invalid="ignore"):
            self._irf_map.data[parent_slices] /= self.exposure_map.data[parent_slices]
            self._irf_map.data = np.nan_to_num(self._irf_map.data)

    def copy(self):
        """Copy IRF map"""
        return deepcopy(self)

    def cutout(self, position, width, mode="trim"):
        """Cutout IRF map.

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
        cutout : `IRFMap`
            Cutout IRF map.
        """
        irf_map = self._irf_map.cutout(position, width, mode)
        exposure_map = self.exposure_map.cutout(position, width, mode)
        return self.__class__(irf_map, exposure_map=exposure_map)

    def downsample(self, factor, axis_name=None, weights=None):
        """Downsample the spatial dimension by a given factor.

        Parameters
        ----------
        factor : int
            Downsampling factor.
        axis_name : str
            Which axis to downsample. By default spatial axes are downsampled.
        weights : `~gammapy.maps.Map`
            Map with weights downsampling.

        Returns
        -------
        map : `IRFMap`
            Downsampled irf map.
        """
        irf_map = self._irf_map.downsample(
            factor=factor, axis_name=axis_name, preserve_counts=True, weights=weights
        )
        if axis_name is None:
            exposure_map = self.exposure_map.downsample(
                factor=factor, preserve_counts=False
            )
        else:
            exposure_map = self.exposure_map.copy()

        return self.__class__(irf_map, exposure_map=exposure_map)

    def slice_by_idx(self, slices):
        """Slice sub dataset.

        The slicing only applies to the maps that define the corresponding axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.

        Returns
        -------
        map_out : `IRFMap`
            Sliced irf map object.
        """
        irf_map = self._irf_map.slice_by_idx(slices=slices)

        if "energy_true" in slices:
            exposure_map = self.exposure_map.slice_by_idx(slices=slices)
        else:
            exposure_map = self.exposure_map

        return self.__class__(irf_map, exposure_map=exposure_map)
