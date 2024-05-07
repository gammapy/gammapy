# Licensed under a 3-clause BSD style license - see LICENSE.rst
import html
import logging
import sys
import astropy.units as u
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.units import Quantity
from .scripts import make_path

log = logging.getLogger(__name__)

__all__ = ["earth_location_from_dict", "LazyFitsData", "HDULocation"]


class HDULocation:
    """HDU localisation, loading and Gammapy object mapper.

    This represents one row in `HDUIndexTable`.

    It's more a helper class, that is wrapped by `~gammapy.data.Observation`,
    usually those objects will be used to access data.

    See also :ref:`gadf:hdu-index`.
    """

    def __init__(
        self,
        hdu_class,
        base_dir=".",
        file_dir=None,
        file_name=None,
        hdu_name=None,
        cache=True,
        format=None,
    ):
        self.hdu_class = hdu_class
        self.base_dir = base_dir
        self.file_dir = file_dir
        self.file_name = file_name
        self.hdu_name = hdu_name
        self.cache = cache
        self.format = format

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def info(self, file=None):
        """Print some summary information to stdout."""
        if not file:
            file = sys.stdout
        print(f"HDU_CLASS = {self.hdu_class}", file=file)
        print(f"BASE_DIR = {self.base_dir}", file=file)
        print(f"FILE_DIR = {self.file_dir}", file=file)
        print(f"FILE_NAME = {self.file_name}", file=file)
        print(f"HDU_NAME = {self.hdu_name}", file=file)

    def path(self, abs_path=True):
        """Full filename path.

        Include ``base_dir`` if ``abs_path`` is True.
        """
        path = make_path(self.base_dir) / self.file_dir / self.file_name

        if abs_path and path.exists():
            return path
        else:
            return make_path(self.file_dir) / self.file_name

    def get_hdu(self):
        """Get HDU."""
        filename = self.path(abs_path=True)
        # Here we're intentionally not calling `with fits.open`
        # because we don't want the file to remain open.
        hdu_list = fits.open(str(filename), memmap=False)
        return hdu_list[self.hdu_name]

    def load(self):
        """Load HDU as appropriate class."""
        from gammapy.irf import IRF_REGISTRY

        hdu_class = self.hdu_class
        filename = self.path()
        hdu = self.hdu_name

        if hdu_class == "events":
            from gammapy.data import EventList

            return EventList.read(filename, hdu=hdu)
        elif hdu_class == "gti":
            from gammapy.data.gti import GTI

            return GTI.read(filename, hdu=hdu)
        elif hdu_class == "map":
            from gammapy.maps import Map

            return Map.read(filename, hdu=hdu, format=self.format)
        elif hdu_class == "pointing":
            # FIXME: support loading the pointing table
            from gammapy.data import FixedPointingInfo

            return FixedPointingInfo.read(filename, hdu=hdu)
        else:
            cls = IRF_REGISTRY.get_cls(hdu_class)

            return cls.read(filename, hdu=hdu)


class LazyFitsData(object):
    """A lazy FITS data descriptor.

    Parameters
    ----------
    cache : bool
        Whether to cache the data.
    """

    def __init__(self, cache=True):
        self.cache = cache

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, objtype):
        if instance is None:
            # Accessed on a class, not an instance
            return self

        try:
            return instance.__dict__[self.name]
        except KeyError:
            hdu_loc = instance.__dict__[f"_{self.name}_hdu"]
            try:
                value = hdu_loc.load()
            except KeyError:
                value = None
                log.warning(f"HDU '{hdu_loc.hdu_name}' not found")
            if self.cache and hdu_loc.cache:
                instance.__dict__[self.name] = value
            return value

    def __set__(self, instance, value):
        if isinstance(value, HDULocation):
            instance.__dict__[f"_{self.name}_hdu"] = value
        else:
            instance.__dict__[self.name] = value


def earth_location_from_dict(meta):
    """Create `~astropy.coordinates.EarthLocation` from FITS header dictionary."""
    lon = Angle(meta["GEOLON"], "deg")
    lat = Angle(meta["GEOLAT"], "deg")
    if "GEOALT" in meta:
        height = Quantity(meta["GEOALT"], "meter")
    elif "ALTITUDE" in meta:
        height = Quantity(meta["ALTITUDE"], "meter")
    else:
        raise KeyError("The GEOALT or ALTITUDE header keyword must be set")

    return EarthLocation(lon=lon, lat=lat, height=height)


def earth_location_to_dict(location):
    """Convert `~astropy.coordinates.EarthLocation` to FITS header dictionary."""
    return {
        "GEOLON": location.lon.deg,
        "GEOLAT": location.lat.deg,
        "ALTITUDE": location.height.to_value(u.m),
    }


def skycoord_from_dict(header, frame="icrs", ext="PNT"):
    """Create `~astropy.coordinates.SkyCoord` from a dictionary of FITS keywords.

    Parameters
    ----------
    header : dict
        The input dictionary.
    frame : {"icrs", "galactic", "altaz"}
        The frame to use. Default is 'icrs'.
    ext: str, optional
        The keyword extension to apply to the keywords names. Default is 'PNT'.

    Returns
    -------
    skycoord : `~astropy.coordinates.skycoord`
        The input SkyCoord.
    """

    ext = "_" + ext if ext != "" else ""

    if frame == "altaz":
        alt = header.get("ALT" + ext, None)
        az = header.get("AZ" + ext, None)
        return (
            AltAz(alt=alt * u.deg, az=az * u.deg)
            if (alt is not None and az is not None)
            else None
        )
    elif frame == "icrs":
        coords = header.get("RA" + ext, None), header.get("DEC" + ext, None)
    elif frame == "galactic":
        coords = header.get("GLON" + ext, None), header.get("GLAT" + ext, None)
    else:
        raise ValueError(
            f"Unsupported frame {frame}. Select in 'icrs', 'galactic', 'altaz'."
        )
    if coords[0] is not None and coords[1] is not None:
        return SkyCoord(coords[0], coords[1], unit="deg", frame=frame)
    else:
        return None
