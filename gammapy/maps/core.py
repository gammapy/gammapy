# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import copy
import inspect
import json
import numpy as np
from astropy import units as u
from astropy.io import fits
from gammapy.maps.hpx import HpxGeom
from gammapy.utils.scripts import make_path
from .geom import MapAxis, MapCoord, pix_tuple_to_idx
from .utils import INVALID_VALUE

__all__ = ["Map"]


class Map(abc.ABC):
    """Abstract map class.

    This can represent WCS- or HEALPIX-based maps
    with 2 spatial dimensions and N non-spatial dimensions.

    Parameters
    ----------
    geom : `~gammapy.maps.Geom`
        Geometry
    data : `~numpy.ndarray`
        Data array
    meta : `dict`
        Dictionary to store meta data
    unit : str or `~astropy.units.Unit`
        Data unit
    """

    tag = "map"

    def __init__(self, geom, data, meta=None, unit=""):
        self._geom = geom
        self.data = data
        self.unit = unit

        if meta is None:
            self.meta = {}
        else:
            self.meta = meta

    def _init_copy(self, **kwargs):
        """Init map instance by copying missing init arguments from self.
        """
        argnames = inspect.getfullargspec(self.__init__).args
        argnames.remove("self")
        argnames.remove("dtype")

        for arg in argnames:
            value = getattr(self, "_" + arg)
            kwargs.setdefault(arg, copy.deepcopy(value))

        return self.from_geom(**kwargs)

    @property
    def geom(self):
        """Map geometry (`~gammapy.maps.Geom`)"""
        return self._geom

    @property
    def data(self):
        """Data array (`~numpy.ndarray`)"""
        return self._data

    @data.setter
    def data(self, value):
        if np.isscalar(value):
            value = value * np.ones(self.geom.data_shape)

        if isinstance(value, u.Quantity):
            raise TypeError("Map data must be a Numpy array. Set unit separately")

        if not value.shape == self.geom.data_shape:
            value = value.reshape(self.geom.data_shape)

        self._data = value

    @property
    def unit(self):
        """Map unit (`~astropy.units.Unit`)"""
        return self._unit

    @unit.setter
    def unit(self, val):
        self._unit = u.Unit(val)

    @property
    def meta(self):
        """Map meta (`dict`)"""
        return self._meta

    @meta.setter
    def meta(self, val):
        self._meta = val

    @property
    def quantity(self):
        """Map data times unit (`~astropy.units.Quantity`)"""
        return u.Quantity(self.data, self.unit, copy=False)

    @quantity.setter
    def quantity(self, val):
        val = u.Quantity(val, copy=False)
        self.data = val.value
        self.unit = val.unit

    @staticmethod
    def create(**kwargs):
        """Create an empty map object.

        This method accepts generic options listed below, as well as options
        for `HpxMap` and `WcsMap` objects. For WCS-specific options, see
        `WcsMap.create` and for HPX-specific options, see `HpxMap.create`.

        Parameters
        ----------
        frame : str
            Coordinate system, either Galactic ("galactic") or Equatorial
            ("icrs").
        map_type : {'wcs', 'wcs-sparse', 'hpx', 'hpx-sparse', 'region'}
            Map type.  Selects the class that will be used to
            instantiate the map.
        binsz : float or `~numpy.ndarray`
            Pixel size in degrees.
        skydir : `~astropy.coordinates.SkyCoord`
            Coordinate of map center.
        axes : list
            List of `~MapAxis` objects for each non-spatial dimension.
            If None then the map will be a 2D image.
        dtype : str
            Data type, default is 'float32'
        unit : str or `~astropy.units.Unit`
            Data unit.
        meta : `dict`
            Dictionary to store meta data.
        region : `~regions.SkyRegion`
            Sky region used for the region map.

        Returns
        -------
        map : `Map`
            Empty map object.
        """
        from .hpxmap import HpxMap
        from .wcsmap import WcsMap
        from .regionnd import RegionNDMap

        map_type = kwargs.setdefault("map_type", "wcs")
        if "wcs" in map_type.lower():
            return WcsMap.create(**kwargs)
        elif "hpx" in map_type.lower():
            return HpxMap.create(**kwargs)
        elif map_type == "region":
            _ = kwargs.pop("map_type")
            return RegionNDMap.create(**kwargs)
        else:
            raise ValueError(f"Unrecognized map type: {map_type!r}")

    @staticmethod
    def read(filename, hdu=None, hdu_bands=None, map_type="auto", format=None):
        """Read a map from a FITS file.

        Parameters
        ----------
        filename : str or `~pathlib.Path`
            Name of the FITS file.
        hdu : str
            Name or index of the HDU with the map data.
        hdu_bands : str
            Name or index of the HDU with the BANDS table.  If not
            defined this will be inferred from the FITS header of the
            map HDU.
        map_type : {'wcs', 'wcs-sparse', 'hpx', 'hpx-sparse', 'auto'}
            Map type.  Selects the class that will be used to
            instantiate the map.  The map type should be consistent
            with the format of the input file.  If map_type is 'auto'
            then an appropriate map type will be inferred from the
            input file.

        Returns
        -------
        map_out : `Map`
            Map object
        """
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return Map.from_hdulist(hdulist, hdu, hdu_bands, map_type, format=format)

    @staticmethod
    def from_geom(geom, meta=None, data=None, unit="", dtype="float32"):
        """Generate an empty map from a `Geom` instance.

        Parameters
        ----------
        geom : `Geom`
            Map geometry.
        data : `numpy.ndarray`
            data array
        meta : `dict`
            Dictionary to store meta data.
        unit : str or `~astropy.units.Unit`
            Data unit.

        Returns
        -------
        map_out : `Map`
            Map object

        """
        from .hpx import HpxGeom
        from .wcs import WcsGeom
        from .region import RegionGeom

        if isinstance(geom, HpxGeom):
            map_type = "hpx"
        elif isinstance(geom, WcsGeom):
            map_type = "wcs"
        elif isinstance(geom, RegionGeom):
            map_type = "region"
        else:
            raise ValueError("Unrecognized geom type.")

        cls_out = Map._get_map_cls(map_type)
        return cls_out(geom, data=data, meta=meta, unit=unit, dtype=dtype)

    @staticmethod
    def from_hdulist(hdulist, hdu=None, hdu_bands=None, map_type="auto", format=None):
        """Create from `astropy.io.fits.HDUList`."""
        if map_type == "auto":
            map_type = Map._get_map_type(hdulist, hdu)
        cls_out = Map._get_map_cls(map_type)
        return cls_out.from_hdulist(
            hdulist, hdu=hdu, hdu_bands=hdu_bands, format=format
        )

    @staticmethod
    def _get_meta_from_header(header):
        """Load meta data from a FITS header."""
        if "META" in header:
            return json.loads(header["META"])
        else:
            return {}

    @staticmethod
    def _get_map_type(hdu_list, hdu_name):
        """Infer map type from a FITS HDU.

        Only read header, never data, to have good performance.
        """
        if hdu_name is None:
            # Find the header of the first non-empty HDU
            header = hdu_list[0].header
            if header["NAXIS"] == 0:
                header = hdu_list[1].header
        else:
            header = hdu_list[hdu_name].header

        if ("PIXTYPE" in header) and (header["PIXTYPE"] == "HEALPIX"):
            return "hpx"
        else:
            return "wcs"

    @staticmethod
    def _get_map_cls(map_type):
        """Get map class for given `map_type` string.

        This should probably be a registry dict so that users
        can add supported map types to the `gammapy.maps` I/O
        (see e.g. the Astropy table format I/O registry),
        but that's non-trivial to implement without avoiding circular imports.
        """
        if map_type == "wcs":
            from .wcsnd import WcsNDMap

            return WcsNDMap
        elif map_type == "wcs-sparse":
            raise NotImplementedError()
        elif map_type == "hpx":
            from .hpxnd import HpxNDMap

            return HpxNDMap
        elif map_type == "hpx-sparse":
            raise NotImplementedError()
        elif map_type == "region":
            from .regionnd import RegionNDMap

            return RegionNDMap
        else:
            raise ValueError(f"Unrecognized map type: {map_type!r}")

    def write(self, filename, overwrite=False, **kwargs):
        """Write to a FITS file.

        Parameters
        ----------
        filename : str
            Output file name.
        overwrite : bool
            Overwrite existing file?
        hdu : str
            Set the name of the image extension.  By default this will
            be set to SKYMAP (for BINTABLE HDU) or PRIMARY (for IMAGE
            HDU).
        hdu_bands : str
            Set the name of the bands table extension.  By default this will
            be set to BANDS.
        format : str, optional
            FITS format convention.  By default files will be written
            to the gamma-astro-data-formats (GADF) format.  This
            option can be used to write files that are compliant with
            format conventions required by specific software (e.g. the
            Fermi Science Tools). The following formats are supported:

                - "gadf" (default)
                - "fgst-ccube"
                - "fgst-ltcube"
                - "fgst-bexpcube"
                - "fgst-srcmap"
                - "fgst-template"
                - "fgst-srcmap-sparse"
                - "galprop"
                - "galprop2"

        sparse : bool
            Sparsify the map by dropping pixels with zero amplitude.
            This option is only compatible with the 'gadf' format.
        """
        hdulist = self.to_hdulist(**kwargs)
        hdulist.writeto(str(make_path(filename)), overwrite=overwrite)

    def iter_by_image(self):
        """Iterate over image planes of the map.

        This is a generator yielding ``(data, idx)`` tuples,
        where ``data`` is a `numpy.ndarray` view of the image plane data,
        and ``idx`` is a tuple of int, the index of the image plane.

        The image plane index is in data order, so that the data array can be
        indexed directly. See :ref:`mapiter` for further information.
        """
        for idx in np.ndindex(self.geom.shape_axes):
            yield self.data[idx[::-1]], idx[::-1]

    def coadd(self, map_in, weights=None):
        """Add the contents of ``map_in`` to this map.

        This method can be used to combine maps containing integral quantities (e.g. counts)
        or differential quantities if the maps have the same binning.

        Parameters
        ----------
        map_in : `Map`
            Input map.
        weights: `Map` or `~numpy.ndarray`
            The weight factors while adding
        """
        if not self.unit.is_equivalent(map_in.unit):
            raise ValueError("Incompatible units")

        # TODO: Check whether geometries are aligned and if so sum the
        # data vectors directly
        if weights is not None:
            map_in = map_in * weights
        idx = map_in.geom.get_idx()
        coords = map_in.geom.get_coord()
        vals = u.Quantity(map_in.get_by_idx(idx), map_in.unit)
        self.fill_by_coord(coords, vals)

    @abc.abstractmethod
    def pad(self, pad_width, mode="constant", cval=0, order=1):
        """Pad the spatial dimensions of the map.

        Parameters
        ----------
        pad_width : {sequence, array_like, int}
            Number of pixels padded to the edges of each axis.
        mode : {'edge', 'constant', 'interp'}
            Padding mode.  'edge' pads with the closest edge value.
            'constant' pads with a constant value. 'interp' pads with
            an extrapolated value.
        cval : float
            Padding value when mode='consant'.
        order : int
            Order of interpolation when mode='constant' (0 =
            nearest-neighbor, 1 = linear, 2 = quadratic, 3 = cubic).

        Returns
        -------
        map : `Map`
            Padded map.

        """
        pass

    @abc.abstractmethod
    def crop(self, crop_width):
        """Crop the spatial dimensions of the map.

        Parameters
        ----------
        crop_width : {sequence, array_like, int}
            Number of pixels cropped from the edges of each axis.
            Defined analogously to ``pad_with`` from `numpy.pad`.

        Returns
        -------
        map : `Map`
            Cropped map.
        """
        pass

    @abc.abstractmethod
    def downsample(self, factor, preserve_counts=True, axis=None):
        """Downsample the spatial dimension by a given factor.

        Parameters
        ----------
        factor : int
            Downsampling factor.
        preserve_counts : bool
            Preserve the integral over each bin.  This should be true
            if the map is an integral quantity (e.g. counts) and false if
            the map is a differential quantity (e.g. intensity).
        axis : str
            Which axis to downsample. By default spatial axes are downsampled.

        Returns
        -------
        map : `Map`
            Downsampled map.
        """
        pass

    @abc.abstractmethod
    def upsample(self, factor, order=0, preserve_counts=True, axis=None):
        """Upsample the spatial dimension by a given factor.

        Parameters
        ----------
        factor : int
            Upsampling factor.
        order : int
            Order of the interpolation used for upsampling.
        preserve_counts : bool
            Preserve the integral over each bin.  This should be true
            if the map is an integral quantity (e.g. counts) and false if
            the map is a differential quantity (e.g. intensity).
        axis : str
            Which axis to upsample. By default spatial axes are upsampled.


        Returns
        -------
        map : `Map`
            Upsampled map.
        """
        pass

    def resample_axis(self, axis, weights=None, ufunc=np.add):
        """Resample map to a new axis binning by grouping over smaller bins and apply ufunc to the bin contents.

        By default, the map content are summed over the smaller bins. Other numpy ufunc can be used,
        e.g. np.logical_and, np.logical_or

        Parameters
        ----------
        axis : `MapAxis`
            New map axis.
        weights : `Map`
            Array to be used as weights. The spatial geometry must be equivalent
            to `other` and additional axes must be broadcastable.
        ufunc : `~numpy.ufunc`
            ufunc to use to resample the axis. Default is numpy.add.


        Returns
        -------
        map : `Map`
            Map with resampled axis.
        """
        geom = self.geom.resample_axis(axis)

        axis_self = self.geom.axes[axis.name]
        axis_resampled = geom.axes[axis.name]

        # We don't use MapAxis.coord_to_idx because is does not behave as needed with boundaries
        coord = axis_resampled.edges.value
        edges = axis_self.edges.value
        indices = np.digitize(coord, edges) - 1

        idx = self.geom.axes.index_data(axis.name)

        weights = 1 if weights is None else weights.data

        if not isinstance(self.geom, HpxGeom):
            shape = self.geom._shape[:2]
        else:
            shape = (self.geom.data_shape[-1],)
        shape += tuple([ax.nbin if ax != axis else 1 for ax in self.geom.axes])

        padded_array = np.append(self.data * weights, np.zeros(shape[::-1]), axis=idx)

        slices = tuple([slice(0, _) for _ in geom.data_shape])
        data = ufunc.reduceat(padded_array, indices=indices, axis=idx)[slices]

        return self._init_copy(data=data, geom=geom)

    def slice_by_idx(
        self, slices,
    ):
        """Slice sub map from map object.

        For usage examples, see :ref:`mapslicing`.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.

        Returns
        -------
        map_out : `Map`
            Sliced map object.
        """
        geom = self.geom.slice_by_idx(slices)
        slices = tuple([slices.get(ax.name, slice(None)) for ax in self.geom.axes])
        data = self.data[slices[::-1]]
        return self.__class__(geom=geom, data=data, unit=self.unit, meta=self.meta)

    def get_image_by_coord(self, coords):
        """Return spatial map at the given axis coordinates.

        Parameters
        ----------
        coords : tuple or dict
            Tuple should be ordered as (x_0, ..., x_n) where x_i are coordinates
            for non-spatial dimensions of the map. Dict should specify the axis
            names of the non-spatial axes such as {'axes0': x_0, ..., 'axesn': x_n}.

        Returns
        -------
        map_out : `Map`
            Map with spatial dimensions only.

        See Also
        --------
        get_image_by_idx, get_image_by_pix

        Examples
        --------
        ::

            import numpy as np
            from gammapy.maps import Map, MapAxis
            from astropy.coordinates import SkyCoord
            from astropy import units as u

            # Define map axes
            energy_axis = MapAxis.from_edges(
                np.logspace(-1., 1., 4), unit='TeV', name='energy',
            )

            time_axis = MapAxis.from_edges(
                np.linspace(0., 10, 20), unit='h', name='time',
            )

            # Define map center
            skydir = SkyCoord(0, 0, frame='galactic', unit='deg')

            # Create map
            m_wcs = Map.create(
                map_type='wcs',
                binsz=0.02,
                skydir=skydir,
                width=10.0,
                axes=[energy_axis, time_axis],
            )

            # Get image by coord tuple
            image = m_wcs.get_image_by_coord(('500 GeV', '1 h'))

            # Get image by coord dict with strings
            image = m_wcs.get_image_by_coord({'energy': '500 GeV', 'time': '1 h'})

            # Get image by coord dict with quantities
            image = m_wcs.get_image_by_coord({'energy': 0.5 * u.TeV, 'time': 1 * u.h})
        """
        if isinstance(coords, tuple):
            coords = dict(zip(self.geom.axes.names, coords))

        idx = self.geom.axes.coord_to_idx(coords)
        return self.get_image_by_idx(idx)

    def get_image_by_pix(self, pix):
        """Return spatial map at the given axis pixel coordinates

        Parameters
        ----------
        pix : tuple
            Tuple of scalar pixel coordinates for each non-spatial dimension of
            the map. Tuple should be ordered as (I_0, ..., I_n). Pixel coordinates
            can be either float or integer type.

        See Also
        --------
        get_image_by_coord, get_image_by_idx

        Returns
        -------
        map_out : `Map`
            Map with spatial dimensions only.
        """
        idx = self.geom.pix_to_idx(pix)
        return self.get_image_by_idx(idx)

    def get_image_by_idx(self, idx):
        """Return spatial map at the given axis pixel indices.

        Parameters
        ----------
        idx : tuple
            Tuple of scalar indices for each non spatial dimension of the map.
            Tuple should be ordered as (I_0, ..., I_n).

        See Also
        --------
        get_image_by_coord, get_image_by_pix

        Returns
        -------
        map_out : `Map`
            Map with spatial dimensions only.
        """
        if len(idx) != len(self.geom.axes):
            raise ValueError("Tuple length must equal number of non-spatial dimensions")

        # Only support scalar indices per axis
        idx = tuple([int(_) for _ in idx])

        geom = self.geom.to_image()
        data = self.data[idx[::-1]]
        return self.__class__(geom=geom, data=data, unit=self.unit, meta=self.meta)

    def get_by_coord(self, coords):
        """Return map values at the given map coordinates.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            Coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        Returns
        -------
        vals : `~numpy.ndarray`
           Values of pixels in the map.  np.nan used to flag coords
           outside of map.
        """
        coords = MapCoord.create(
            coords, frame=self.geom.frame, axis_names=self.geom.axes.names
        )
        pix = self.geom.coord_to_pix(coords)
        vals = self.get_by_pix(pix)
        return vals

    def get_by_pix(self, pix):
        """Return map values at the given pixel coordinates.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
            Pixel indices can be either float or integer type.

        Returns
        -------
        vals : `~numpy.ndarray`
           Array of pixel values.  np.nan used to flag coordinates
           outside of map
        """
        # FIXME: Support local indexing here?
        # FIXME: Support slicing?
        pix = np.broadcast_arrays(*pix)
        idx = self.geom.pix_to_idx(pix)
        vals = self.get_by_idx(idx)
        mask = self.geom.contains_pix(pix)

        if not mask.all():
            invalid = INVALID_VALUE[self.data.dtype]
            vals = vals.astype(type(invalid))
            vals[~mask] = invalid

        return vals

    @abc.abstractmethod
    def get_by_idx(self, idx):
        """Return map values at the given pixel indices.

        Parameters
        ----------
        idx : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.

        Returns
        -------
        vals : `~numpy.ndarray`
           Array of pixel values.
           np.nan used to flag coordinate outside of map
        """
        pass

    @abc.abstractmethod
    def interp_by_coord(self, coords, interp=None, fill_value=None):
        """Interpolate map values at the given map coordinates.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            Coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        interp : {None, 'nearest', 'linear', 'cubic', 0, 1, 2, 3}
            Method to interpolate data values.  By default no
            interpolation is performed and the return value will be
            the amplitude of the pixel encompassing the given
            coordinate.  Integer values can be used in lieu of strings
            to choose the interpolation method of the given order
            (0='nearest', 1='linear', 2='quadratic', 3='cubic').  Note
            that only 'nearest' and 'linear' methods are supported for
            all map types.
        fill_value : None or float value
            The value to use for points outside of the interpolation domain.
            If None, values outside the domain are extrapolated.

        Returns
        -------
        vals : `~numpy.ndarray`
            Interpolated pixel values.
        """
        pass

    @abc.abstractmethod
    def interp_by_pix(self, pix, interp=None, fill_value=None):
        """Interpolate map values at the given pixel coordinates.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinate arrays for each dimension of the
            map.  Tuple should be ordered as (p_lon, p_lat, p_0, ...,
            p_n) where p_i are pixel coordinates for non-spatial
            dimensions of the map.

        interp : {None, 'nearest', 'linear', 'cubic', 0, 1, 2, 3}
            Method to interpolate data values.  By default no
            interpolation is performed and the return value will be
            the amplitude of the pixel encompassing the given
            coordinate.  Integer values can be used in lieu of strings
            to choose the interpolation method of the given order
            (0='nearest', 1='linear', 2='quadratic', 3='cubic').  Note
            that only 'nearest' and 'linear' methods are supported for
            all map types.
        fill_value : None or float value
            The value to use for points outside of the interpolation domain.
            If None, values outside the domain are extrapolated.

        Returns
        -------
        vals : `~numpy.ndarray`
            Interpolated pixel values.
        """
        pass

    def interp_to_geom(self, geom, preserve_counts=False, **kwargs):
        """Interpolate map to input geometry.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Target Map geometry
        preserve_counts : bool
            Preserve the integral over each bin.  This should be true
            if the map is an integral quantity (e.g. counts) and false if
            the map is a differential quantity (e.g. intensity)
        **kwargs : dict
            Keyword arguments passed to `Map.interp_by_coord`

        Returns
        -------
        interp_map : `Map`
            Interpolated Map
        """

        coords = geom.get_coord()

        # set nearest neighbour interpolation for mask as default
        if self.data.dtype == bool:
            kwargs.setdefault("interp", 0)

        if preserve_counts:
            if geom.ndim > 2:
                assert self.geom.axes[0] == geom.axes[0]  # Energy axis has to match
            old_map_copy = self.copy()
            old_map_copy.data /= self.geom.solid_angle().to_value("deg2")
            data = old_map_copy.interp_by_coord(coords, **kwargs)
            data *= geom.solid_angle().to_value("deg2")
        else:
            data = self.interp_by_coord(coords, **kwargs)

        if self.data.dtype == bool:
            data = data.astype(bool)
        return Map.from_geom(geom, data=data, unit=self.unit)

    def fill_events(self, events):
        """Fill event coordinates (`~gammapy.data.EventList`)."""
        self.fill_by_coord(events.map_coord(self.geom))

    def fill_by_coord(self, coords, weights=None):
        """Fill pixels at ``coords`` with given ``weights``.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            Coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.
        weights : `~numpy.ndarray`
            Weights vector. Default is weight of one.
        """
        idx = self.geom.coord_to_idx(coords)
        self.fill_by_idx(idx, weights)

    def fill_by_pix(self, pix, weights=None):
        """Fill pixels at ``pix`` with given ``weights``.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
            Pixel indices can be either float or integer type.  Float
            indices will be rounded to the nearest integer.
        weights : `~numpy.ndarray`
            Weights vector. Default is weight of one.
        """
        idx = pix_tuple_to_idx(pix)
        return self.fill_by_idx(idx, weights=weights)

    @abc.abstractmethod
    def fill_by_idx(self, idx, weights=None):
        """Fill pixels at ``idx`` with given ``weights``.

        Parameters
        ----------
        idx : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
        weights : `~numpy.ndarray`
            Weights vector. Default is weight of one.
        """
        pass

    def set_by_coord(self, coords, vals):
        """Set pixels at ``coords`` with given ``vals``.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            Coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.
        vals : `~numpy.ndarray`
            Values vector.
        """
        idx = self.geom.coord_to_pix(coords)
        self.set_by_pix(idx, vals)

    def set_by_pix(self, pix, vals):
        """Set pixels at ``pix`` with given ``vals``.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
            Pixel indices can be either float or integer type.  Float
            indices will be rounded to the nearest integer.
        vals : `~numpy.ndarray`
            Values vector.
        """
        idx = pix_tuple_to_idx(pix)
        return self.set_by_idx(idx, vals)

    @abc.abstractmethod
    def set_by_idx(self, idx, vals):
        """Set pixels at ``idx`` with given ``vals``.

        Parameters
        ----------
        idx : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
        vals : `~numpy.ndarray`
            Values vector.
        """
        pass

    def plot_grid(self, figsize=None, ncols=3, **kwargs):
        """Plot map as a grid of subplots for non-spatial axes

        Parameters
        ----------
        figsize : tuple of int
            Figsize to plot on
        ncols : int
            Number of columns to plot
        **kwargs : dict
            Keyword arguments passed to `Map.plot`.

        Returns
        -------
        axes : `~numpy.ndarray` of `~matplotlib.pyplot.Axes`
            Axes grid
        """
        import matplotlib.pyplot as plt

        if len(self.geom.axes) > 1:
            raise ValueError("Grid plotting is only supported for one non spatial axis")

        axis = self.geom.axes[0]

        cols = min(ncols, axis.nbin)
        rows = 1 + (axis.nbin - 1) // cols

        if figsize is None:
            width = 12
            figsize = (width, width * rows / cols)

        if self.geom.is_hpx:
            wcs = self.geom.to_wcs_geom().wcs
        else:
            wcs = self.geom.wcs

        fig, axes = plt.subplots(
            ncols=cols,
            nrows=rows,
            subplot_kw={"projection": wcs},
            figsize=figsize,
            gridspec_kw={"hspace": 0.1, "wspace": 0.1},
        )

        for idx in range(cols * rows):
            ax = axes.flat[idx]

            try:
                image = self.get_image_by_idx((idx,))
            except IndexError:
                ax.set_visible(False)
                continue

            if image.geom.is_hpx:
                image_wcs = (image.to_wcs(normalize=False, proj="AIT", oversample=2,),)
            else:
                image_wcs = image

            image_wcs.plot(ax=ax, **kwargs)

            if axis.node_type == "center":
                info = f"{axis.center[idx]:.1f}"
            else:
                info = f"{axis.edges[idx]:.1f} - {axis.edges[idx + 1]:.1f} "

            ax.set_title(f"{axis.name.capitalize()} " + info)
            lon, lat = ax.coords[0], ax.coords[1]
            lon.set_ticks_position("b")
            lat.set_ticks_position("l")

            row, col = np.unravel_index(idx, shape=(rows, cols))

            if col > 0:
                lat.set_ticklabel_visible(False)
                lat.set_axislabel("")

            if row < (rows - 1):
                lon.set_ticklabel_visible(False)
                lon.set_axislabel("")

        return axes

    def plot_interactive(self, rc_params=None, **kwargs):
        """
        Plot map with interactive widgets to explore the non spatial axes.

        Parameters
        ----------
        rc_params : dict
            Passed to ``matplotlib.rc_context(rc=rc_params)`` to style the plot.
        **kwargs : dict
            Keyword arguments passed to `WcsNDMap.plot`.

        Examples
        --------
        You can try this out e.g. using a Fermi-LAT diffuse model cube with an energy axis::

            from gammapy.maps import Map

            m = Map.read("$GAMMAPY_DATA/fermi_3fhl/gll_iem_v06_cutout.fits")
            m.plot_interactive(add_cbar=True, stretch="sqrt")

        If you would like to adjust the figure size you can use the ``rc_params`` argument::

            rc_params = {'figure.figsize': (12, 6), 'font.size': 12}
            m.plot_interactive(rc_params=rc_params)
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from ipywidgets.widgets.interaction import interact, fixed
        from ipywidgets import SelectionSlider, RadioButtons

        if self.geom.is_image:
            raise TypeError("Use .plot() for 2D Maps")

        kwargs.setdefault("interpolation", "nearest")
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("cmap", "afmhot")

        rc_params = rc_params or {}
        stretch = kwargs.pop("stretch", "sqrt")

        interact_kwargs = {}

        for axis in self.geom.axes:
            if axis.node_type == "edges":
                options = [
                    f"{val_min:.2e} - {val_max:.2e} {axis.unit}"
                    for val_min, val_max in zip(axis.edges[:-1], axis.edges[1:])
                ]
            else:
                options = [f"{val:.2e} {axis.unit}" for val in axis.center]

            interact_kwargs[axis.name] = SelectionSlider(
                options=options,
                description=f"Select {axis.name}:",
                continuous_update=False,
                style={"description_width": "initial"},
                layout={"width": "50%"},
            )
            interact_kwargs[axis.name + "_options"] = fixed(options)

        interact_kwargs["stretch"] = RadioButtons(
            options=["linear", "sqrt", "log"],
            value=stretch,
            description="Select stretch:",
            style={"description_width": "initial"},
        )

        @interact(**interact_kwargs)
        def _plot_interactive(**ikwargs):
            idx = [
                ikwargs[ax.name + "_options"].index(ikwargs[ax.name])
                for ax in self.geom.axes
            ]
            img = self.get_image_by_idx(idx)
            stretch = ikwargs["stretch"]
            with mpl.rc_context(rc=rc_params):
                fig, ax, cbar = img.plot(stretch=stretch, **kwargs)
                plt.show()

    def copy(self, **kwargs):
        """Copy map instance and overwrite given attributes, except for geometry.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to overwrite in the map constructor.

        Returns
        -------
        copy : `Map`
            Copied Map.
        """
        if "geom" in kwargs:
            raise ValueError("Can't copy and change geometry of the map.")
        return self._init_copy(**kwargs)

    def apply_edisp(self, edisp):
        """Apply energy dispersion to map. Requires energy axis.

        Parameters
        ----------
        edisp : `gammapy.irf.EDispKernel`
            Energy dispersion matrix

        Returns
        -------
        map : `WcsNDMap`
            Map with energy dispersion applied.
        """
        # TODO: either use sparse matrix mutiplication or something like edisp.is_diagonal
        if edisp is not None:
            loc = self.geom.axes.index("energy_true")
            data = np.rollaxis(self.data, loc, len(self.data.shape))
            data = np.dot(data, edisp.pdf_matrix)
            data = np.rollaxis(data, -1, loc)
            energy_axis = edisp.energy_axis.copy(name="energy")
        else:
            data = self.data
            energy_axis = self.geom.axes["energy_true"].copy(name="energy")

        geom = self.geom.to_image().to_cube(axes=[energy_axis])
        return self._init_copy(geom=geom, data=data)

    def sum_over_axes(self, axes_names=None, keepdims=True, weights=None):
        """To sum map values over all non-spatial axes.

        Parameters
        ----------
        keepdims : bool, optional
            If this is set to true, the axes which are summed over are left in
            the map with a single bin
        axes_names: list of str
            Names of MapAxis to reduce over. If None, all will summed over
        weights : `Map`
            Weights to be applied. The Map should have the same geometry.

        Returns
        -------
        map_out : `~Map`
            Map with non-spatial axes summed over
        """
        return self.reduce_over_axes(
            func=np.add, axes_names=axes_names, keepdims=keepdims, weights=weights
        )

    def reduce_over_axes(
        self, func=np.add, keepdims=False, axes_names=None, weights=None
    ):
        """Reduce map over non-spatial axes

        Parameters
        ----------
        func : `~numpy.ufunc`
            Function to use for reducing the data.
        keepdims : bool, optional
            If this is set to true, the axes which are summed over are left in
            the map with a single bin
        axes_names: list
            Names of MapAxis to reduce over
            If None, all will reduced
        weights : `Map`
            Weights to be applied.

        Returns
        -------
        map_out : `~Map`
            Map with non-spatial axes reduced
        """
        if axes_names is None:
            axes_names = self.geom.axes.names

        map_out = self.copy()
        for axis_name in axes_names:
            map_out = map_out.reduce(
                axis_name, func=func, keepdims=keepdims, weights=weights
            )

        return map_out

    def reduce(self, axis_name, func=np.add, keepdims=False, weights=None):
        """Reduce map over a single non-spatial axis

        Parameters
        ----------
        axis_name: str
            The name of the axis to reduce over
        func : `~numpy.ufunc`
            Function to use for reducing the data.
        keepdims : bool, optional
            If this is set to true, the axes which are summed over are left in
            the map with a single bin
        weights : `Map`
            Weights to be applied.

        Returns
        -------
        map_out : `~Map`
            Map with the given non-spatial axes reduced
        """
        if keepdims:
            geom = self.geom.squash(axis_name=axis_name)
        else:
            geom = self.geom.drop(axis_name=axis_name)

        idx = self.geom.axes.index_data(axis_name)

        data = self.data

        if weights is not None:
            data = data * weights

        data = func.reduce(data, axis=idx, keepdims=keepdims, where=~np.isnan(data))
        return self._init_copy(geom=geom, data=data)

    @classmethod
    def from_images(cls, images, axis=None):
        """Create Map from list of images and a non-spatial axis.

        If the images have a non-spatial axis of length 1 a new axes is generated
        from by merging the individual axes. The image geometries must be aligned.

        Parameters
        ----------
        images : list of `Map` objects
            Images
        axis : `MapAxis`
            Map axis

        Returns
        -------
        map : `Map`
            Map with additional non-spatial axis.

        """
        geom_ref = images[0].geom.to_image()

        data = []

        for image in images:
            if not image.geom.to_image() == geom_ref:
                raise ValueError("Image geometries not aligned")
            data.append(image.data)

        if axis is None:
            try:
                axis = MapAxis.from_stack(axes=[image.geom.axes[0] for image in images])
            except IndexError:
                ValueError(
                    "Images don't have a non-spatial axis. Please provide"
                    " the axis separately"
                )

        return cls.from_geom(
            data=np.stack(data), geom=geom_ref.to_cube(axes=[axis]), unit=images[0].unit
        )

    def to_cube(self, axes):
        """Append non-spatial axes to create a higher-dimensional Map.

        This will result in a Map with a new geometry with
        N+M dimensions where N is the number of current dimensions and
        M is the number of axes in the list. The data is reshaped onto the
        new geometry

        Parameters
        ----------
        axes : list
            Axes that will be appended to this Map.
            The axes should have only one bin

        Returns
        -------
        map : `~gammapy.maps.WcsNDMap`
            new map
        """
        for ax in axes:
            if ax.nbin > 1:
                raise ValueError(ax.name, "should have only one bin")
        geom = self.geom.to_cube(axes)
        data = self.data.reshape((1,) * len(axes) + self.data.shape)
        return self.from_geom(data=data, geom=geom, unit=self.unit)

    def __repr__(self):
        geom = self.geom.__class__.__name__
        axes = ["skycoord"] if self.geom.is_hpx else ["lon", "lat"]
        axes = axes + [_.name for _ in self.geom.axes]

        return (
            f"{self.__class__.__name__}\n\n"
            f"\tgeom  : {geom} \n "
            f"\taxes  : {axes}\n"
            f"\tshape : {self.geom.data_shape[::-1]}\n"
            f"\tndim  : {self.geom.ndim}\n"
            f"\tunit  : {self.unit}\n"
            f"\tdtype : {self.data.dtype}\n"
        )

    def _arithmetics(self, operator, other, copy):
        """Perform arithmetics on maps after checking geometry consistency."""
        if isinstance(other, Map):
            if self.geom == other.geom:
                q = other.quantity
            else:
                raise ValueError("Map Arithmetics: Inconsistent geometries.")
        else:
            q = u.Quantity(other, copy=False)

        out = self.copy() if copy else self
        out.quantity = operator(out.quantity, q)
        return out

    def _boolean_arithmetics(self, operator, other, copy):
        """Perform arithmetics on maps after checking geometry consistency."""
        if operator == np.logical_not:
            out = self.copy()
            out.data = operator(out.data)
            return out

        if isinstance(other, Map):
            if self.geom == other.geom:
                other = other.data
            else:
                raise ValueError("Map Arithmetics: Inconsistent geometries.")

        out = self.copy() if copy else self
        out.data = operator(out.data, other)
        return out

    def __add__(self, other):
        return self._arithmetics(np.add, other, copy=True)

    def __iadd__(self, other):
        return self._arithmetics(np.add, other, copy=False)

    def __sub__(self, other):
        return self._arithmetics(np.subtract, other, copy=True)

    def __isub__(self, other):
        return self._arithmetics(np.subtract, other, copy=False)

    def __mul__(self, other):
        return self._arithmetics(np.multiply, other, copy=True)

    def __imul__(self, other):
        return self._arithmetics(np.multiply, other, copy=False)

    def __truediv__(self, other):
        return self._arithmetics(np.true_divide, other, copy=True)

    def __itruediv__(self, other):
        return self._arithmetics(np.true_divide, other, copy=False)

    def __le__(self, other):
        return self._arithmetics(np.less_equal, other, copy=True)

    def __lt__(self, other):
        return self._arithmetics(np.less, other, copy=True)

    def __ge__(self, other):
        return self._arithmetics(np.greater_equal, other, copy=True)

    def __gt__(self, other):
        return self._arithmetics(np.greater, other, copy=True)

    def __eq__(self, other):
        return self._arithmetics(np.equal, other, copy=True)

    def __ne__(self, other):
        return self._arithmetics(np.not_equal, other, copy=True)

    def __and__(self, other):
        return self._boolean_arithmetics(np.logical_and, other, copy=True)

    def __or__(self, other):
        return self._boolean_arithmetics(np.logical_or, other, copy=True)

    def __invert__(self):
        return self._boolean_arithmetics(np.logical_not, None, copy=True)

    def __xor__(self, other):
        return self._boolean_arithmetics(np.logical_xor, other, copy=True)

    def __iand__(self, other):
        return self._boolean_arithmetics(np.logical_and, other, copy=False)

    def __ior__(self, other):
        return self._boolean_arithmetics(np.logical_or, other, copy=False)

    def __ixor__(self, other):
        return self._boolean_arithmetics(np.logical_xor, other, copy=False)

    def __array__(self):
        return self.data
