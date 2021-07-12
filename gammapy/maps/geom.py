# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import copy
import inspect
import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from .utils import INVALID_INDEX, find_bands_hdu, find_hdu

__all__ = ["Geom", "MapCoord"]

log = logging.getLogger(__name__)


def get_shape(param):
    if param is None:
        return tuple()

    if not isinstance(param, tuple):
        param = [param]

    return max([np.array(p, ndmin=1).shape for p in param])


def skycoord_to_lonlat(skycoord, frame=None):
    """Convert SkyCoord to lon, lat, frame.

    Returns
    -------
    lon : `~numpy.ndarray`
        Longitude in degrees.
    lat : `~numpy.ndarray`
        Latitude in degrees.
    """
    if frame:
        skycoord = skycoord.transform_to(frame)

    return skycoord.data.lon.deg, skycoord.data.lat.deg, skycoord.frame.name


def pix_tuple_to_idx(pix):
    """Convert a tuple of pixel coordinate arrays to a tuple of pixel indices.

    Pixel coordinates are rounded to the closest integer value.

    Parameters
    ----------
    pix : tuple
        Tuple of pixel coordinates with one element for each dimension

    Returns
    -------
    idx : `~numpy.ndarray`
        Array of pixel indices
    """
    idx = []
    for p in pix:
        p = np.array(p, ndmin=1)
        if np.issubdtype(p.dtype, np.integer):
            idx += [p]
        else:
            p_idx = np.rint(p).astype(int)
            p_idx[~np.isfinite(p)] = INVALID_INDEX.int
            idx += [p_idx]

    return tuple(idx)


class MapCoord:
    """Represents a sequence of n-dimensional map coordinates.

    Contains coordinates for 2 spatial dimensions and an arbitrary
    number of additional non-spatial dimensions.

    For further information see :ref:`mapcoord`.

    Parameters
    ----------
    data : `dict` of `~numpy.ndarray`
        Dictionary of coordinate arrays.
    frame : {"icrs", "galactic", None}
        Spatial coordinate system.  If None then the coordinate system
        will be set to the native coordinate system of the geometry.
    match_by_name : bool
        Match coordinates to axes by name?
        If false coordinates will be matched by index.
    """

    def __init__(self, data, frame=None, match_by_name=True):
        if "lon" not in data or "lat" not in data:
            raise ValueError("data dictionary must contain axes named 'lon' and 'lat'.")

        self._data = {k: np.atleast_1d(v) for k, v in data.items()}
        self._frame = frame
        self._match_by_name = match_by_name

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        else:
            return list(self._data.values())[key]

    def __iter__(self):
        return iter(self._data.values())

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self._data)

    @property
    def shape(self):
        """Coordinate array shape."""
        arrays = [_ for _ in self._data.values()]
        return np.broadcast(*arrays).shape

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def lon(self):
        """Longitude coordinate in degrees."""
        return self._data["lon"]

    @property
    def lat(self):
        """Latitude coordinate in degrees."""
        return self._data["lat"]

    @property
    def theta(self):
        """Theta co-latitude angle in radians."""
        theta = u.Quantity(self.lat, unit="deg", copy=False).to_value("rad")
        return np.pi / 2.0 - theta

    @property
    def phi(self):
        """Phi longitude angle in radians."""
        phi = u.Quantity(self.lon, unit="deg", copy=False).to_value("rad")
        return phi

    @property
    def frame(self):
        """Coordinate system (str)."""
        return self._frame

    @property
    def match_by_name(self):
        """Boolean flag: axis lookup by name (True) or index (False)."""
        return self._match_by_name

    @property
    def skycoord(self):
        return SkyCoord(self.lon, self.lat, unit="deg", frame=self.frame)

    @classmethod
    def _from_lonlat(cls, coords, frame=None, axis_names=None):
        """Create a `~MapCoord` from a tuple of coordinate vectors.

        The first two elements of the tuple should be longitude and latitude in degrees.

        Parameters
        ----------
        coords : tuple
            Tuple of `~numpy.ndarray`.

        Returns
        -------
        coord : `~MapCoord`
            A coordinates object.
        """
        if axis_names is None:
            axis_names = [f"axis{idx}" for idx in range(len(coords) - 2)]

        if isinstance(coords, (list, tuple)):
            coords_dict = {"lon": coords[0], "lat": coords[1]}
            for name, c in zip(axis_names, coords[2:]):
                coords_dict[name] = c
        else:
            raise ValueError("Unrecognized input type.")

        return cls(coords_dict, frame=frame, match_by_name=False)

    @classmethod
    def _from_tuple(cls, coords, frame=None, axis_names=None):
        """Create from tuple of coordinate vectors."""
        if isinstance(coords[0], (list, np.ndarray)) or np.isscalar(coords[0]):
            return cls._from_lonlat(coords, frame=frame, axis_names=axis_names)
        elif isinstance(coords[0], SkyCoord):
            lon, lat, frame = skycoord_to_lonlat(coords[0], frame=frame)
            coords = (lon, lat) + coords[1:]
            return cls._from_lonlat(coords, frame=frame, axis_names=axis_names)
        else:
            raise TypeError(f"Type not supported: {type(coords)!r}")

    @classmethod
    def _from_dict(cls, coords, frame=None):
        """Create from a dictionary of coordinate vectors."""
        if "lon" in coords and "lat" in coords:
            return cls(coords, frame=frame)
        elif "skycoord" in coords:
            lon, lat, frame = skycoord_to_lonlat(coords["skycoord"], frame=frame)
            coords_dict = {"lon": lon, "lat": lat}
            for k, v in coords.items():
                if k == "skycoord":
                    continue
                coords_dict[k] = v
            return cls(coords_dict, frame=frame)
        else:
            raise ValueError("coords dict must contain 'lon'/'lat' or 'skycoord'.")

    @classmethod
    def create(cls, data, frame=None, axis_names=None):
        """Create a new `~MapCoord` object.

        This method can be used to create either unnamed (with tuple input)
        or named (via dict input) axes.

        Parameters
        ----------
        data : tuple, dict, `MapCoord` or `~astropy.coordinates.SkyCoord`
            Object containing coordinate arrays.
        frame : {"icrs", "galactic", None}, optional
            Set the coordinate system for longitude and latitude. If
            None longitude and latitude will be assumed to be in
            the coordinate system native to a given map geometry.
        axis_names : list of str
            Axis names use if a tuple is provided

        Examples
        --------
        >>> from astropy.coordinates import SkyCoord
        >>> from gammapy.maps import MapCoord

        >>> lon, lat = [1, 2], [2, 3]
        >>> skycoord = SkyCoord(lon, lat, unit='deg')
        >>> energy = [1000]
        >>> c = MapCoord.create((lon,lat))
        >>> c = MapCoord.create((skycoord,))
        >>> c = MapCoord.create((lon,lat,energy))
        >>> c = MapCoord.create(dict(lon=lon,lat=lat))
        >>> c = MapCoord.create(dict(lon=lon,lat=lat,energy=energy))
        >>> c = MapCoord.create(dict(skycoord=skycoord,energy=energy))
        """
        if isinstance(data, cls):
            if data.frame is None or frame == data.frame:
                return data
            else:
                return data.to_frame(frame)
        elif isinstance(data, dict):
            return cls._from_dict(data, frame=frame)
        elif isinstance(data, (list, tuple)):
            return cls._from_tuple(data, frame=frame, axis_names=axis_names)
        elif isinstance(data, SkyCoord):
            return cls._from_tuple((data,), frame=frame, axis_names=axis_names)
        else:
            raise TypeError(f"Unsupported input type: {type(data)!r}")

    def to_frame(self, frame):
        """Convert to a different coordinate frame.

        Parameters
        ----------
        frame : {"icrs", "galactic"}
            Coordinate system, either Galactic ("galactic") or Equatorial ("icrs").

        Returns
        -------
        coords : `~MapCoord`
            A coordinates object.
        """
        if frame == self.frame:
            return copy.deepcopy(self)
        else:
            lon, lat, frame = skycoord_to_lonlat(self.skycoord, frame=frame)
            data = copy.deepcopy(self._data)
            if isinstance(self.lon, u.Quantity):
                lon = u.Quantity(lon, unit="deg", copy=False)

            if isinstance(self.lon, u.Quantity):
                lat = u.Quantity(lat, unit="deg", copy=False)

            data["lon"] = lon
            data["lat"] = lat
            return self.__class__(data, frame, self._match_by_name)

    def apply_mask(self, mask):
        """Return a masked copy of this coordinate object.

        Parameters
        ----------
        mask : `~numpy.ndarray`
            Boolean mask.

        Returns
        -------
        coords : `~MapCoord`
            A coordinates object.
        """
        try:
            data = {k: v[mask] for k, v in self._data.items()}
        except IndexError:
            data = {}

            for name, coord in self._data.items():
                if name in ["lon", "lat"]:
                    data[name] = np.squeeze(coord)[mask]
                else:
                    data[name] = np.squeeze(coord, axis=-1)

        return self.__class__(data, self.frame, self._match_by_name)

    @property
    def flat(self):
        """Return flattened, valid coordinates"""
        coords = self.broadcasted
        is_finite = np.isfinite(coords[0])
        return coords.apply_mask(is_finite)

    @property
    def broadcasted(self):
        """Return broadcasted coords"""
        vals = np.broadcast_arrays(*self._data.values(), subok=True)
        data = dict(zip(self._data.keys(), vals))
        return self.__class__(data=data, frame=self.frame, match_by_name=self._match_by_name)

    def copy(self):
        """Copy `MapCoord` object."""
        return copy.deepcopy(self)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n\n"
            f"\taxes     : {list(self._data.keys())}\n"
            f"\tshape    : {self.shape[::-1]}\n"
            f"\tndim     : {self.ndim}\n"
            f"\tframe : {self.frame}\n"
        )


class Geom(abc.ABC):
    """Map geometry base class.

    See also: `~gammapy.maps.WcsGeom` and `~gammapy.maps.HpxGeom`
    """

    @property
    @abc.abstractmethod
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        pass

    def data_nbytes(self, dtype="float32"):
        """Estimate memory usage in megabytes of the Numpy data array
        matching this geometry depending on the given type.

        Parameters
        ----------
        dtype : data-type
            The desired data-type for the array. Default is "float32"
            
        Returns
        -------
        memory : `~astropy.units.Quantity`
            Estimated memory usage in megabytes (MB)
        """
        return (np.empty(self.data_shape, dtype).nbytes * u.byte).to("MB")

    @property
    @abc.abstractmethod
    def is_allsky(self):
        pass

    @property
    @abc.abstractmethod
    def center_coord(self):
        pass

    @property
    @abc.abstractmethod
    def center_pix(self):
        pass

    @property
    @abc.abstractmethod
    def center_skydir(self):
        pass

    @classmethod
    def from_hdulist(cls, hdulist, hdu=None, hdu_bands=None):
        """Load a geometry object from a FITS HDUList.

        Parameters
        ----------
        hdulist :  `~astropy.io.fits.HDUList`
            HDU list containing HDUs for map data and bands.
        hdu : str
            Name or index of the HDU with the map data.
        hdu_bands : str
            Name or index of the HDU with the BANDS table.  If not
            defined this will be inferred from the FITS header of the
            map HDU.

        Returns
        -------
        geom : `~Geom`
            Geometry object.
        """
        if hdu is None:
            hdu = find_hdu(hdulist)
        else:
            hdu = hdulist[hdu]

        if hdu_bands is None:
            hdu_bands = find_bands_hdu(hdulist, hdu)

        if hdu_bands is not None:
            hdu_bands = hdulist[hdu_bands]

        return cls.from_header(hdu.header, hdu_bands)

    def to_bands_hdu(self, hdu_bands=None, format="gadf"):
        table_hdu = self.axes.to_table_hdu(format=format, hdu_bands=hdu_bands)
        cols = table_hdu.columns.columns
        cols.extend(self._make_bands_cols())
        return fits.BinTableHDU.from_columns(
            cols, header=table_hdu.header, name=table_hdu.name
        )

    @abc.abstractmethod
    def _make_bands_cols(self):
        pass

    @abc.abstractmethod
    def get_idx(self, idx=None, local=False, flat=False):
        """Get tuple of pixel indices for this geometry.

        Returns all pixels in the geometry by default. Pixel indices
        for a single image plane can be accessed by setting ``idx``
        to the index tuple of a plane.

        Parameters
        ----------
        idx : tuple, optional
            A tuple of indices with one index for each non-spatial
            dimension.  If defined only pixels for the image plane with
            this index will be returned.  If none then all pixels
            will be returned.
        local : bool
            Flag to return local or global pixel indices.  Local
            indices run from 0 to the number of pixels in a given
            image plane.
        flat : bool, optional
            Return a flattened array containing only indices for
            pixels contained in the geometry.

        Returns
        -------
        idx : tuple
            Tuple of pixel index vectors with one vector for each
            dimension.
        """
        pass

    @abc.abstractmethod
    def get_coord(self, idx=None, flat=False):
        """Get the coordinate array for this geometry.

        Returns a coordinate array with the same shape as the data
        array.  Pixels outside the geometry are set to NaN.
        Coordinates for a single image plane can be accessed by
        setting ``idx`` to the index tuple of a plane.

        Parameters
        ----------
        idx : tuple, optional
            A tuple of indices with one index for each non-spatial
            dimension.  If defined only coordinates for the image
            plane with this index will be returned.  If none then
            coordinates for all pixels will be returned.
        flat : bool, optional
            Return a flattened array containing only coordinates for
            pixels contained in the geometry.

        Returns
        -------
        coords : tuple
            Tuple of coordinate vectors with one vector for each
            dimension.
        """
        pass

    @abc.abstractmethod
    def coord_to_pix(self, coords):
        """Convert map coordinates to pixel coordinates.

        Parameters
        ----------
        coords : tuple
            Coordinate values in each dimension of the map.  This can
            either be a tuple of numpy arrays or a MapCoord object.
            If passed as a tuple then the ordering should be
            (longitude, latitude, c_0, ..., c_N) where c_i is the
            coordinate vector for axis i.

        Returns
        -------
        pix : tuple
            Tuple of pixel coordinates in image and band dimensions.
        """
        pass

    def coord_to_idx(self, coords, clip=False):
        """Convert map coordinates to pixel indices.

        Parameters
        ----------
        coords : tuple or `~MapCoord`
            Coordinate values in each dimension of the map.  This can
            either be a tuple of numpy arrays or a MapCoord object.
            If passed as a tuple then the ordering should be
            (longitude, latitude, c_0, ..., c_N) where c_i is the
            coordinate vector for axis i.
        clip : bool
            Choose whether to clip indices to the valid range of the
            geometry.  If false then indices for coordinates outside
            the geometry range will be set -1.

        Returns
        -------
        pix : tuple
            Tuple of pixel indices in image and band dimensions.
            Elements set to -1 correspond to coordinates outside the
            map.
        """
        pix = self.coord_to_pix(coords)
        return self.pix_to_idx(pix, clip=clip)

    @abc.abstractmethod
    def pix_to_coord(self, pix):
        """Convert pixel coordinates to map coordinates.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.

        Returns
        -------
        coords : tuple
            Tuple of map coordinates.
        """
        pass

    @abc.abstractmethod
    def pix_to_idx(self, pix, clip=False):
        """Convert pixel coordinates to pixel indices.

        Returns -1 for pixel coordinates that lie outside of the map.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.
        clip : bool
            Choose whether to clip indices to the valid range of the
            geometry.  If false then indices for coordinates outside
            the geometry range will be set -1.

        Returns
        -------
        idx : tuple
            Tuple of pixel indices.
        """
        pass

    @abc.abstractmethod
    def contains(self, coords):
        """Check if a given map coordinate is contained in the geometry.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            Tuple of map coordinates.

        Returns
        -------
        containment : `~numpy.ndarray`
            Bool array.
        """
        pass

    def contains_pix(self, pix):
        """Check if a given pixel coordinate is contained in the geometry.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.

        Returns
        -------
        containment : `~numpy.ndarray`
            Bool array.
        """
        idx = self.pix_to_idx(pix)
        return np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)

    def slice_by_idx(self, slices):
        """Create a new geometry by slicing the non-spatial axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.

        Returns
        -------
        geom : `~Geom`
            Sliced geometry.
        """
        axes = self.axes.slice_by_idx(slices)
        return self._init_copy(axes=axes)

    @property
    def as_energy_true(self):
        """If the geom contains an energy axis rename it to energy true"""
        energy_axis = self.axes["energy"].copy(name="energy_true")
        return self.to_image().to_cube([energy_axis])

    @property
    def has_energy_axis(self):
        """Whether geom has an energy axis"""
        return ("energy" in self.axes.names) ^ ("energy_true" in self.axes.names)

    @abc.abstractmethod
    def to_image(self):
        """Create 2D image geometry (drop non-spatial dimensions).

        Returns
        -------
        geom : `~Geom`
            Image geometry.
        """
        pass

    @abc.abstractmethod
    def to_cube(self, axes):
        """Append non-spatial axes to create a higher-dimensional geometry.

        This will result in a new geometry with
        N+M dimensions where N is the number of current dimensions and
        M is the number of axes in the list.

        Parameters
        ----------
        axes : list
            Axes that will be appended to this geometry.

        Returns
        -------
        geom : `~Geom`
            Map geometry.
        """
        pass

    def squash(self, axis_name):
        """Squash geom axis.

        Parameters
        ----------
        axis_name : str
            Axis to squash.

        Returns
        -------
        geom : `Geom`
            Geom with squashed axis.
        """
        axes = self.axes.squash(axis_name=axis_name)
        return self.to_image().to_cube(axes=axes)

    def drop(self, axis_name):
        """Drop an axis from the geom.

        Parameters
        ----------
        axis_name : str
            Name of the axis to remove.

        Returns
            -------
        geom : `Geom`
            New geom with the axis removed.
        """
        axes = self.axes.drop(axis_name=axis_name)
        return self.to_image().to_cube(axes=axes)

    def pad(self, pad_width, axis_name):
        """
        Pad the geometry at the edges.

        Parameters
        ----------
        pad_width : {sequence, array_like, int}
            Number of values padded to the edges of each axis.
        axis_name : str
            Name of the axis to pad.

        Returns
        -------
        geom : `~Geom`
            Padded geometry.
        """
        if axis_name is None:
            return self._pad_spatial(pad_width)
        else:
            axes = self.axes.pad(axis_name=axis_name, pad_width=pad_width)
            return self.to_image().to_cube(axes)

    @abc.abstractmethod
    def _pad_spatial(self, pad_width):
        pass

    @abc.abstractmethod
    def crop(self, crop_width):
        """
        Crop the geometry at the edges.

        Parameters
        ----------
        crop_width : {sequence, array_like, int}
            Number of values cropped from the edges of each axis.

        Returns
        -------
        geom : `~Geom`
            Cropped geometry.
        """
        pass

    @abc.abstractmethod
    def downsample(self, factor, axis_name):
        """Downsample the spatial dimension of the geometry by a given factor.

        Parameters
        ----------
        factor : int
            Downsampling factor.
        axis_name : str
            Axis to downsample.

        Returns
        -------
        geom : `~Geom`
            Downsampled geometry.

        """
        pass

    @abc.abstractmethod
    def upsample(self, factor, axis_name):
        """Upsample the spatial dimension of the geometry by a given factor.

        Parameters
        ----------
        factor : int
            Upsampling factor.
        axis_name : str
            Axis to upsample.

        Returns
        -------
        geom : `~Geom`
            Upsampled geometry.

        """
        pass

    def resample_axis(self, axis):
        """Resample geom to a new axis binning.

        This method groups the existing bins into a new binning.

        Parameters
        ----------
        axis : `MapAxis`
            New map axis.

        Returns
        -------
        map : `Geom`
            Geom with resampled axis.
        """
        axes = self.axes.resample(axis=axis)
        return self._init_copy(axes=axes)

    @abc.abstractmethod
    def solid_angle(self):
        """Solid angle (`~astropy.units.Quantity` in ``sr``)."""
        pass

    @property
    def is_image(self):
        """Whether the geom is an image without extra dimensions."""
        if self.axes is None:
            return True
        return len(self.axes) == 0

    @property
    def is_flat(self):
        """Whether the geom non spatial axes have length 1, i.e. if the geom is equivalent to an image."""
        if self.is_image:
            return True
        else:
            valid = True
            for axis in self.axes:
                valid = valid and (axis.nbin == 1)
            return valid

    def _init_copy(self, **kwargs):
        """Init map geom instance by copying missing init arguments from self.
        """
        argnames = inspect.getfullargspec(self.__init__).args
        argnames.remove("self")

        for arg in argnames:
            value = getattr(self, "_" + arg)
            kwargs.setdefault(arg, copy.deepcopy(value))

        return self.__class__(**kwargs)

    def copy(self, **kwargs):
        """Copy and overwrite given attributes.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to overwrite in the map geometry constructor.

        Returns
        -------
        copy : `Geom`
            Copied map geometry.
        """
        return self._init_copy(**kwargs)

    def energy_mask(self, energy_min=None, energy_max=None, round_to_edge=False):
        """Create a mask for a given energy range.

        The energy bin must be fully contained to be included in the mask.

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy range

        Returns
        -------
        mask : `~numpy.ndarray`
            Energy mask
        """
        from . import Map

        # get energy axes and values
        energy_axis = self.axes["energy"]

        if round_to_edge:
            energy_min, energy_max = energy_axis.round([energy_min, energy_max])

        # TODO: make this more general
        shape = (-1, 1) if self.is_hpx else (-1, 1, 1)
        energy_edges = energy_axis.edges.reshape(shape)

        # set default values
        energy_min = energy_min if energy_min is not None else energy_edges[0]
        energy_max = energy_max if energy_max is not None else energy_edges[-1]

        mask = (energy_edges[:-1] >= energy_min) & (energy_edges[1:] <= energy_max)
        data = np.broadcast_to(mask, shape=self.data_shape)
        return Map.from_geom(geom=self, data=data)
