# Licensed under a 3-clause BSD style license - see LICENSE.rst
import html
import logging
import warnings
from enum import Enum, auto
import numpy as np
import scipy.interpolate
import astropy.units as u
from astropy.coordinates import (
    ICRS,
    AltAz,
    BaseCoordinateFrame,
    CartesianRepresentation,
    SkyCoord,
    UnitSphericalRepresentation,
)
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.utils import lazyproperty
from gammapy.utils.compat import COPY_IF_NEEDED
from gammapy.utils.deprecation import GammapyDeprecationWarning
from gammapy.utils.fits import earth_location_from_dict, earth_location_to_dict
from gammapy.utils.scripts import make_path
from gammapy.utils.time import time_ref_from_dict, time_ref_to_dict, time_to_fits_header

log = logging.getLogger(__name__)

__all__ = ["FixedPointingInfo", "PointingInfo", "PointingMode"]


def _check_coord_frame(coord_or_frame, expected_frame, name):
    """Check if a skycoord or frame is given in expected_frame."""
    is_coord = isinstance(coord_or_frame, SkyCoord)
    is_frame = isinstance(coord_or_frame, BaseCoordinateFrame)

    if not (is_frame or is_coord):
        raise TypeError(
            f"{name} must be a 'astropy.coordinates.SkyCoord'"
            "or {expected_frame} instance"
        )

    if is_coord:
        frame = coord_or_frame.frame
    else:
        frame = coord_or_frame

    if not isinstance(frame, expected_frame):
        raise ValueError(
            f"{name} is in wrong frame, expected {expected_frame}, got {frame}"
        )


class PointingMode(Enum):
    """
    Describes the behavior of the pointing during the observation.

    See :ref:`gadf:iact-events-obs-mode`.

    For ground-based instruments, the most common options will be:
    * POINTING: The telescope observes a fixed position in the ICRS frame
    * DRIFT: The telescope observes a fixed position in the alt-az frame

    Gammapy only supports fixed pointing positions over the whole observation
    (either in equatorial or horizontal coordinates).
    OGIP also defines RASTER, SLEW and SCAN. These cannot be treated using
    a fixed pointing position in either frame, so they would require the
    pointing table, which is at the moment not supported by gammapy.

    Data releases based on gadf v0.2 do not have consistent OBS_MODE keyword
    e.g. the H.E.S.S. data releases uses the not-defined value "WOBBLE".
    For all gadf data, we assume OBS_MODE to be the same as "POINTING",
    unless it is set to "DRIFT", making the assumption that one observation
    only contains a single fixed position.
    """

    POINTING = auto()
    DRIFT = auto()

    @staticmethod
    def from_gadf_string(val):
        """Parse a string from the GADF header into a PointingMode."""
        # OBS_MODE is not well-defined and not mandatory in GADF 0.2
        # We always assume that the observations are pointing observations
        # unless the OBS_MODE is set to DRIFT
        if val.upper() == "DRIFT":
            return PointingMode.DRIFT
        else:
            return PointingMode.POINTING


class FixedPointingInfo:
    """IACT array pointing info.

    Data format specification: :ref:`gadf:iact-pnt`

    Parameters
    ----------
    meta : `~astropy.table.Table.meta`
        Meta header info from Table on pointing.
        Passing this is deprecated, provide ``mode`` and ``fixed_icrs`` or ``fixed_altaz``
        instead or use `FixedPointingInfo.from_fits_header` instead.
    mode : `PointingMode`
        How the telescope was pointing during the observation.
    fixed_icrs : `~astropy.coordinates.SkyCoord`, optional
        The coordinates of the observation in ICRS as a `~astropy.coordinates.SkyCoord` object. Default is None.
        Required if mode is `PointingMode.POINTING`.
    fixed_altaz : `~astropy.coordinates.SkyCoord`, optional
        The coordinates of the observation in alt-az as a `~astropy.coordinates.SkyCoord` object. Default is None.
        Required if mode is `PointingMode.DRIFT`.

    Examples
    --------
    >>> from gammapy.data import FixedPointingInfo, PointingMode
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> fixed_icrs = SkyCoord(83.633 * u.deg, 22.014 * u.deg, frame="icrs")
    >>> pointing_info = FixedPointingInfo(fixed_icrs=fixed_icrs)
    >>> print(pointing_info)
    FixedPointingInfo:
    <BLANKLINE>
    mode:        PointingMode.POINTING
    coordinates: <SkyCoord (ICRS): (ra, dec) in deg
        (83.633, 22.014)>
    """

    def __init__(
        self,
        meta=None,
        *,
        mode=None,
        fixed_icrs=None,
        fixed_altaz=None,
        # these have nothing really to do with pointing_info
        # needed for backwards compatibility but should be removed and accessed
        # from the observation, not the pointing info.
        location=None,
        time_start=None,
        time_stop=None,
        time_ref=None,
        legacy_altaz=None,  # store altaz given for mode=POINTING separately so it can be removed easily
    ):
        self._meta = None

        # TODO: for backwards compatibility, remove in 2.0
        # and make other keywards required
        if meta is not None:
            warnings.warn(
                "Initializing a FixedPointingInfo using a `meta` dict is deprecated",
                GammapyDeprecationWarning,
            )
            self._meta = meta
            self.__dict__.update(self.from_fits_header(meta).__dict__)
            return

        if mode is not None:
            warnings.warn(
                "Passing mode is deprecated and the argument will be removed in Gammapy 1.3."
                " pointing mode is deduced from whether fixed_icrs or fixed_altaz is given",
                GammapyDeprecationWarning,
            )

        self._location = location
        self._time_start = time_start
        self._time_stop = time_stop
        self._time_ref = time_ref
        self._legacy_altaz = legacy_altaz or AltAz(np.nan * u.deg, np.nan * u.deg)

        if fixed_icrs is not None and fixed_altaz is not None:
            raise ValueError("fixed_icrs and fixed_altaz are mutually exclusive")

        if fixed_icrs is not None:
            _check_coord_frame(fixed_icrs, ICRS, "fixed_icrs")

            if np.isnan(fixed_icrs.ra.value) or np.isnan(fixed_icrs.dec.value):
                warnings.warn(
                    "In future, fixed_icrs must have non-nan values",
                    GammapyDeprecationWarning,
                )

            self._mode = PointingMode.POINTING
            self._fixed_icrs = fixed_icrs
            self._fixed_altaz = None

        else:
            _check_coord_frame(fixed_altaz, AltAz, "fixed_altaz")
            self._mode = PointingMode.DRIFT
            self._fixed_icrs = None
            self._fixed_altaz = fixed_altaz

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    @classmethod
    def from_fits_header(cls, header):
        """
        Parse `~gammapy.data.FixedPointingInfo` from the given FITS header.

        Parameters
        ----------
        header : `astropy.fits.Header`
            Header to parse, e.g. from a GADF EVENTS HDU.
            Currently, only the GADF format is supported.

        Returns
        -------
        pointing : `~gammapy.data.FixedPointingInfo`
            The FixedPointingInfo instance filled from the given header.
        """
        obs_mode = header.get("OBS_MODE", "POINTING")
        mode = PointingMode.from_gadf_string(obs_mode)
        try:
            location = earth_location_from_dict(header)
        except KeyError:
            location = None

        # we allow missing RA_PNT / DEC_PNT in POINTING for some reason...
        # FIXME: actually enforce this to be present instead of using nan
        ra = u.Quantity(header.get("RA_PNT", np.nan), u.deg)
        dec = u.Quantity(header.get("DEC_PNT", np.nan), u.deg)
        alt = header.get("ALT_PNT")
        az = header.get("AZ_PNT")
        legacy_altaz = None

        fixed_icrs = None
        pointing_altaz = None

        # we can be more strict with DRIFT, as support was only added recently
        if mode is PointingMode.DRIFT:
            if alt is None or az is None:
                raise IOError(
                    "Keywords ALT_PNT and AZ_PNT are required for OBSMODE=DRIFT"
                )
            pointing_altaz = AltAz(alt=alt * u.deg, az=az * u.deg)
        else:
            fixed_icrs = SkyCoord(ra, dec)
            if np.isnan(ra.value) or np.isnan(dec.value):
                warnings.warn(
                    "RA_PNT / DEC_PNT will be required in a future version of"
                    " gammapy for pointing-mode POINTING",
                    GammapyDeprecationWarning,
                )
            # store given altaz also for POINTING for backwards compatibility,
            # FIXME: remove in 2.0
            if alt is not None and az is not None:
                legacy_altaz = AltAz(alt=alt * u.deg, az=az * u.deg)

        time_start = header.get("TSTART")
        time_stop = header.get("TSTOP")
        time_ref = None

        if time_start is not None or time_stop is not None:
            time_ref = time_ref_from_dict(header)
            time_unit = u.Unit(header.get("TIMEUNIT", "s"), format="fits")

            if time_start is not None:
                time_start = time_ref + u.Quantity(time_start, time_unit)

            if time_stop is not None:
                time_stop = time_ref + u.Quantity(time_stop, time_unit)

        return cls(
            location=location,
            fixed_icrs=fixed_icrs,
            fixed_altaz=pointing_altaz,
            time_start=time_start,
            time_stop=time_stop,
            time_ref=time_ref,
            legacy_altaz=legacy_altaz,
        )

    def to_fits_header(self, format="gadf", version="0.3", time_ref=None):
        """
        Convert this FixedPointingInfo object into a fits header for the given format.

        Parameters
        ----------
        format : {"gadf"}
            Format, currently only "gadf" is supported. Default is "gadf".
        version : str, optional
            Version of the ``format``, this function currently supports
            gadf versions 0.2 and 0.3. Default is "0.3".
        time_ref : `astropy.time.Time`, optional
            Reference time for storing the time related information in fits format. Default is None.

        Returns
        -------
        header : `astropy.fits.Header`
            Header with fixed pointing information filled for the requested format.
        """
        if format != "gadf":
            raise ValueError(f'Only the "gadf" format supported, got {format}')

        if version not in {"0.2", "0.3"}:
            raise ValueError(f"Unsupported version {version} for format {format}")

        if self.mode == PointingMode.DRIFT and version == "0.2":
            raise ValueError("mode=DRIFT is only supported by GADF 0.3")

        header = fits.Header()
        if self.mode is PointingMode.POINTING:
            header["OBS_MODE"] = "POINTING"
            header["RA_PNT"] = self.fixed_icrs.ra.deg, u.deg.to_string("fits")
            header["DEC_PNT"] = self.fixed_icrs.dec.deg, u.deg.to_string("fits")
        elif self.mode is PointingMode.DRIFT:
            header["OBS_MODE"] = "DRIFT"
            header["AZ_PNT"] = self.fixed_altaz.az.deg, u.deg.to_string("fits")
            header["ALT_PNT"] = self.fixed_altaz.alt.deg, u.deg.to_string("fits")

        # FIXME: remove in 2.0
        if self._legacy_altaz is not None and not np.isnan(
            self._legacy_altaz.alt.value
        ):
            header["AZ_PNT"] = self._legacy_altaz.az.deg, u.deg.to_string("fits")
            header["ALT_PNT"] = self._legacy_altaz.alt.deg, u.deg.to_string("fits")

        if self._time_start is not None:
            header["TSTART"] = time_to_fits_header(self._time_start, epoch=time_ref)
        if self._time_stop is not None:
            header["TSTOP"] = time_to_fits_header(self._time_stop, epoch=time_ref)

        if self._time_start is not None or self._time_stop is not None:
            header.update(time_ref_to_dict(time_ref))

        if self._location is not None:
            header.update(earth_location_to_dict(self._location))

        return header

    @classmethod
    def read(cls, filename, hdu="EVENTS"):
        """Read pointing information table from file to obtain the metadata.

        Parameters
        ----------
        filename : str
            Filename.
        hdu : int or str, optional
            HDU number or name. Default is "EVENTS".

        Returns
        -------
        pointing_info : `PointingInfo`
            Pointing information.
        """
        filename = make_path(filename)
        header = fits.getheader(filename, extname=hdu)
        return cls.from_fits_header(header)

    @property
    def mode(self):
        """See `PointingMode`, if not present, assume POINTING."""
        return self._mode

    @property
    def fixed_altaz(self):
        """The fixed coordinates of the observation in alt-az as a `~astropy.coordinates.SkyCoord` object.

        None if not a DRIFT observation.
        """
        return self._fixed_altaz

    @property
    def fixed_icrs(self):
        """
        The fixed coordinates of the observation in ICRS as a `~astropy.coordinates.SkyCoord` object.

        None if not a POINTING observation.
        """
        return self._fixed_icrs

    def get_icrs(self, obstime=None, location=None) -> SkyCoord:
        """
        Get the pointing position in ICRS frame for a given time.

        If the observation was performed tracking a fixed position in ICRS,
        the icrs pointing is returned with the given obstime attached.

        If the observation was performed in drift mode, the fixed alt-az coordinates
        are transformed to ICRS using the observation location and the given time.


        Parameters
        ----------
        obstime : `astropy.time.Time`, optional
            Time for which to get the pointing position in ICRS frame. Default is None.
        location : `astropy.coordinates.EarthLocation`, optional
            Observatory location, only needed for drift observations to transform
            from horizontal coordinates to ICRS. Default is None.

        Returns
        -------
        icrs : `astropy.coordinates.SkyCoord`
            Pointing position in ICRS frame.
        """
        if self.mode == PointingMode.POINTING:
            return SkyCoord(self._fixed_icrs.data, location=location, obstime=obstime)

        if self.mode == PointingMode.DRIFT:
            if obstime is None:
                obstime = self.obstime

            return self.get_altaz(obstime, location=location).icrs

        raise ValueError(f"Unsupported pointing mode: {self.mode}.")

    def get_altaz(self, obstime=None, location=None) -> SkyCoord:
        """
        Get the pointing position in alt-az frame for a given time.

        If the observation was performed tracking a fixed position in ICRS,
        the icrs pointing is transformed at the given time using the location
        of the observation.

        If the observation was performed in drift mode,
        the fixed alt-az coordinate is returned with `obstime` attached.

        Parameters
        ----------
        obstime : `astropy.time.Time`, optional
            Time for which to get the pointing position in alt-az frame. Default is None.
        location : `astropy.coordinates.EarthLocation`, optional
            Observatory location, only needed for pointing observations to transform
            from ICRS to horizontal coordinates. Default is None.

        Returns
        -------
        altaz : `astropy.coordinates.SkyCoord`
            Pointing position in alt-az frame.
        """
        location = location if location is not None else self._location

        frame = AltAz(location=location, obstime=obstime)

        if self.mode == PointingMode.POINTING:
            return self.fixed_icrs.transform_to(frame)

        if self.mode == PointingMode.DRIFT:
            # see https://github.com/astropy/astropy/issues/12965
            alt = self.fixed_altaz.alt
            az = self.fixed_altaz.az
            return SkyCoord(
                alt=u.Quantity(
                    np.full(obstime.shape, alt.deg), u.deg, copy=COPY_IF_NEEDED
                ),
                az=u.Quantity(
                    np.full(obstime.shape, az.deg), u.deg, copy=COPY_IF_NEEDED
                ),
                frame=frame,
            )

        raise ValueError(f"Unsupported pointing mode: {self.mode}.")

    def __str__(self):
        coordinates = (
            self.fixed_icrs if self.mode == PointingMode.POINTING else self.fixed_altaz
        )
        return (
            "FixedPointingInfo:\n\n"
            f"mode:        {self.mode}\n"
            f"coordinates: {coordinates}"
        )


class PointingInfo:
    """IACT array pointing info.

    Data format specification: :ref:`gadf:iact-pnt`.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table (with meta header information) on pointing.

    Examples
    --------
    >>> from gammapy.data import PointingInfo
    >>> pointing_info = PointingInfo.read('$GAMMAPY_DATA/tests/pointing_table.fits.gz')
    >>> print(pointing_info)
    Pointing info:
    <BLANKLINE>
    Location:     GeodeticLocation(lon=<Longitude 16.50022222 deg>, lat=<Latitude -23.27177778 deg>, height=<Quantity 1835. m>)
    MJDREFI, MJDREFF, TIMESYS = (51910, 0.000742870370370241, 'TT')
    Time ref:     2001-01-01T00:01:04.184
    Time ref:     51910.00074287037 MJD (TT)
    Duration:     1586.0000000000018 sec = 0.44055555555555603 hours
    Table length: 100
    <BLANKLINE>
    START:
    Time:  2004-01-21T19:50:02.184
    Time:  53025.826414166666 MJD (TT)
    RADEC: 83.6333 24.5144 deg
    ALTAZ: 11.4575 41.3409 deg
    <BLANKLINE>
    <BLANKLINE>
    END:
    Time:  2004-01-21T20:16:28.184
    Time:  53025.844770648146 MJD (TT)
    RADEC: 83.6333 24.5144 deg
    ALTAZ: 3.44573 42.1319 deg
    <BLANKLINE>
    <BLANKLINE>

    Note: In order to reproduce the example you need the tests datasets folder.
    You may download it with the command
    ``gammapy download datasets --tests --out $GAMMAPY_DATA``
    """

    def __init__(self, table):
        self.table = table

    @classmethod
    def read(cls, filename, hdu="POINTING"):
        """Read `PointingInfo` table from file.

        Parameters
        ----------
        filename : str
            Filename.
        hdu : int or str, optional
            HDU number or name. Default is "POINTING".

        Returns
        -------
        pointing_info : `PointingInfo`
            Pointing information.
        """
        filename = make_path(filename)
        table = Table.read(filename, hdu=hdu)
        return cls(table=table)

    def __str__(self):
        ss = "Pointing info:\n\n"
        ss += f"Location:     {self.location.geodetic}\n"
        m = self.table.meta
        ss += "MJDREFI, MJDREFF, TIMESYS = {}\n".format(
            (m["MJDREFI"], m["MJDREFF"], m["TIMESYS"])
        )
        ss += f"Time ref:     {self.time_ref.fits}\n"
        ss += f"Time ref:     {self.time_ref.mjd} MJD (TT)\n"
        sec = self.duration.to("second").value
        hour = self.duration.to("hour").value
        ss += f"Duration:     {sec} sec = {hour} hours\n"
        ss += "Table length: {}\n".format(len(self.table))

        ss += "\nSTART:\n" + self._str_for_index(0) + "\n"
        ss += "\nEND:\n" + self._str_for_index(-1) + "\n"

        return ss

    def _str_for_index(self, idx):
        """Information for one point in the pointing table."""
        ss = "Time:  {}\n".format(self.time[idx].fits)
        ss += "Time:  {} MJD (TT)\n".format(self.time[idx].mjd)
        ss += "RADEC: {} deg\n".format(self.radec[idx].to_string())
        ss += "ALTAZ: {} deg\n".format(self.altaz[idx].to_string())
        return ss

    @lazyproperty
    def location(self):
        """Observatory location as an `~astropy.coordinates.EarthLocation` object."""
        return earth_location_from_dict(self.table.meta)

    @lazyproperty
    def time_ref(self):
        """Time reference as a `~astropy.time.Time` object."""
        return time_ref_from_dict(self.table.meta)

    @lazyproperty
    def duration(self):
        """Pointing table duration as a `~astropy.time.TimeDelta` object.

        The time difference between the first and last entry.
        """
        return self.time[-1] - self.time[0]

    @lazyproperty
    def time(self):
        """Time array as a `~astropy.time.Time` object."""
        met = Quantity(self.table["TIME"].astype("float64"), "second")
        time = self.time_ref + met
        return time.tt

    @lazyproperty
    def radec(self):
        """RA / DEC position from table as a `~astropy.coordinates.SkyCoord`."""
        lon = self.table["RA_PNT"]
        lat = self.table["DEC_PNT"]
        return SkyCoord(lon, lat, unit="deg", frame="icrs")

    @lazyproperty
    def altaz_frame(self):
        """ALT / AZ frame as a `~astropy.coordinates.AltAz` object."""
        return AltAz(obstime=self.time, location=self.location)

    @lazyproperty
    def altaz(self):
        """ALT / AZ position computed from RA / DEC as a`~astropy.coordinates.SkyCoord`."""
        return self.radec.transform_to(self.altaz_frame)

    @lazyproperty
    def altaz_from_table(self):
        """ALT / AZ position from table as a `~astropy.coordinates.SkyCoord`."""
        lon = self.table["AZ_PNT"]
        lat = self.table["ALT_PNT"]
        return SkyCoord(lon, lat, unit="deg", frame=self.altaz_frame)

    @staticmethod
    def _interpolate_cartesian(mjd_support, coord_support, mjd):
        xyz = coord_support.cartesian
        x_new = scipy.interpolate.interp1d(mjd_support, xyz.x)(mjd)
        y_new = scipy.interpolate.interp1d(mjd_support, xyz.y)(mjd)
        z_new = scipy.interpolate.interp1d(mjd_support, xyz.z)(mjd)
        return CartesianRepresentation(x_new, y_new, z_new).represent_as(
            UnitSphericalRepresentation
        )

    def altaz_interpolate(self, time):
        """Interpolate pointing for a given time."""
        altaz_frame = AltAz(obstime=time, location=self.location)
        return SkyCoord(
            self._interpolate_cartesian(self.time.mjd, self.altaz, time.mjd),
            frame=altaz_frame,
        )

    def get_icrs(self, obstime):
        """
        Get the pointing position in ICRS frame for a given time.

        Parameters
        ----------
        obstime : `astropy.time.Time`
            Time for which to get the pointing position in ICRS frame.

        Returns
        -------
        icrs : `astropy.coordinates.SkyCoord`
            Pointing position in ICRS frame.
        """
        return SkyCoord(
            self._interpolate_cartesian(self.time.mjd, self.radec, obstime.mjd),
            obstime=obstime,
            frame="icrs",
        )

    def get_altaz(self, obstime):
        """
        Get the pointing position in alt-az frame for a given time.

        If the observation was performed tracking a fixed position in ICRS,
        the icrs pointing is transformed at the given time using the location
        of the observation.

        If the observation was performed in drift mode,
        the fixed alt-az coordinate is returned with `obstime` attached.

        Parameters
        ----------
        obstime : `astropy.time.Time`
            Time for which to get the pointing position in alt-az frame.

        Returns
        -------
        altaz : `astropy.coordinates.SkyCoord`
            Pointing position in alt-az frame.
        """
        # give precedence to ALT_PNT / AZ_PNT if present
        if "ALT_PNT" in self.table and "AZ_PNT" in self.table:
            altaz = self.altaz_from_table
            frame = AltAz(obstime=obstime, location=self.location)
            return SkyCoord(
                self._interpolate_cartesian(self.time.mjd, altaz, obstime.mjd),
                frame=frame,
            )

        # fallback to transformation from required ICRS if not
        return self.altaz_interpolate(time=obstime)
