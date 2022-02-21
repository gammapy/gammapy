import copy
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

__all__ = ["MapCoord"]


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

    def __setitem__(self, key, value):
        # TODO: check for broadcastability?
        self._data[key] = value

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
        data : tuple, dict, `~gammapy.maps.MapCoord` or `~astropy.coordinates.SkyCoord`
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
        return self.__class__(
            data=data, frame=self.frame, match_by_name=self._match_by_name
        )

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
