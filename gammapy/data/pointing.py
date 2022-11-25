# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from enum import Enum, auto
import numpy as np
import scipy.interpolate
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import (
    AltAz,
    CartesianRepresentation,
    SkyCoord,
    UnitSphericalRepresentation,
)
from astropy.table import Table
from astropy.units import Quantity
from astropy.utils import lazyproperty
from gammapy.utils.fits import earth_location_from_dict
from gammapy.utils.scripts import make_path
from gammapy.utils.time import time_ref_from_dict
import warnings


log = logging.getLogger(__name__)

__all__ = ["FixedPointingInfo", "PointingInfo", "PointingMode"]


class PointingMode(Enum):
    """
    Describes the behavior of the pointing during the observation.

    See :ref:`gadf:obs-mode`.

    For ground-based instruments, the most common options will be:
    * POINTING: The telescope observes a fixed position in the ICRS frame
    * DRIFT: The telescope observes a fixed position in the AltAz frame

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
        Meta header info from Table on pointing

    Examples
    --------
    >>> from gammapy.data import PointingInfo
    >>> path = '$GAMMAPY_DATA/tests/pointing_table.fits.gz'
    >>> pointing_info = PointingInfo.read(path)
    >>> print(pointing_info)
    Pointing info:
    <BLANKLINE>
    Location:     GeodeticLocation(lon=<Longitude 16.50022222 deg>, lat=<Latitude -23.27177778 deg>, height=<Quantity 1835. m>)  # noqa: E501
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

    def __init__(
        self,
        meta=None,
        mode=None,
        pointing_icrs=None,
        pointing_altaz=None,
        location=None,
        time_start=None,
        time_stop=None,
        time_ref=None,
    ):
        self._meta = None

        # TODO: for backwards compatibility, remove in 2.0
        # and make other keywards required
        if meta is not None:
            warnings.warn("Initializing a FixedPointingInfo using a `meta` dict is deprecated")
            self._meta = meta
            self.__dict__.update(self.from_gadf_header(meta).__dict__)
            return

        if not isinstance(mode, PointingMode):
            raise TypeError(f'mode must be an instance of PointingMode, got {mode!r}')

        self._mode = mode
        self._location = location
        self._time_start = time_start
        self._time_stop = time_stop
        self._time_ref = time_ref

        if mode is PointingMode.POINTING:
            if pointing_icrs is None:
                raise ValueError("pointing_icrs is mandatory for PointingMode.POINTING")

            if np.isnan(pointing_icrs.ra.value) or np.isnan(pointing_icrs.dec.value):
                warnings.warn("In future, pointing_icrs must have non-nan values", DeprecationWarning)

            if pointing_altaz is not None:
                warnings.warn("In future, pointing_altaz will be mutually exclusive with mode=POINTING", DeprecationWarning)

            self._pointing_icrs = pointing_icrs
            self._pointing_altaz = pointing_altaz

        elif mode is PointingMode.DRIFT:
            if pointing_altaz is None:
                raise ValueError("pointing_altaz is required for PointingMode.DRIFT")

            if pointing_icrs is not None:
                raise ValueError("pointing_icrs is excluded for PointingMode.DRIFT")

            self._pointing_altaz = pointing_altaz
            self._pointing_icrs = None
        else:
            raise ValueError(f"Unsupported pointing mode for FixedPointingInfo: {mode}")

    @classmethod
    def from_gadf_header(cls, header):
        obs_mode = header.get("OBS_MODE", "POINTING")
        mode = PointingMode.from_gadf_string(obs_mode)
        try:
            location = earth_location_from_dict(header)
        except KeyError:
            location = None

        # we allow missing RA_PNT / DEC_PNT in POINTING for some reason...
        # FIXME: actually enforce this to be present instead of using nan
        ra = header.get("RA_PNT", np.nan)
        dec = header.get("DEC_PNT", np.nan)

        # we can be more strict with DRIFT, as support was only added recently
        if mode is PointingMode.DRIFT:
            pointing_icrs = None
        else:
            pointing_icrs = SkyCoord(ra, dec, unit=u.deg, frame="icrs")

        alt = header.get("ALT_PNT")
        az = header.get("AZ_PNT")
        if alt is not None and az is not None:
            pointing_altaz = SkyCoord(alt=alt, az=az, unit=u.deg, frame=AltAz(location=location))
        else:
            pointing_altaz = None

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
            mode=mode,
            location=location,
            pointing_icrs=pointing_icrs,
            pointing_altaz=pointing_altaz,
            time_start=time_start,
            time_stop=time_stop,
            time_ref=time_ref,
        )

    @property
    def meta(self):
        if self._meta is not None:
            return self.meta

    @classmethod
    def read(cls, filename, hdu="EVENTS"):
        """Read pointing information table from file to obtain the metadata.

        Parameters
        ----------
        filename : str
            File name
        hdu : int or str
            HDU number or name

        Returns
        -------
        pointing_info : `PointingInfo`
            Pointing info object
        """
        filename = make_path(filename)
        header = fits.getheader(filename, extname=hdu)
        return cls.from_gadf_header(header)

    @property
    def mode(self):
        """See `PointingMode`, if not present, assume POINTING"""
        return self._mode

    @property
    def location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)."""
        warnings.warn("Accessing observatory location through pointing is deprecated", DeprecationWarning)
        return self._location

    @property
    def time_start(self):
        """Start time (`~astropy.time.Time`)."""
        warnings.warn("Accessing time information through pointing is deprecated", DeprecationWarning)
        return self._time_start

    @property
    def time_stop(self):
        """Stop time (`~astropy.time.Time`)."""
        warnings.warn("Accessing time information through pointing is deprecated", DeprecationWarning)
        return self._time_stop

    @property
    def time_ref(self):
        """Reference time (`~astropy.time.Time`)."""
        warnings.warn("Accessing time information through pointing is deprecated", DeprecationWarning)
        return self._time_ref

    @lazyproperty
    def obstime(self):
        """Average observation time for the observation (`~astropy.time.Time`)."""
        warnings.warn("Accessing time information through pointing is deprecated", DeprecationWarning)
        if self.time_start is None or self.duration is None:
            return None
        return self.time_start + self.duration / 2

    @lazyproperty
    def duration(self):
        """Pointing duration (`~astropy.time.TimeDelta`).

        The time difference between the TSTART and TSTOP.
        """
        warnings.warn("Accessing time information through pointing is deprecated", DeprecationWarning)
        if self.time_start is None or self.time_stop is None:
            return None
        return self.time_stop - self.time_start

    @lazyproperty
    def radec(self):
        """
        RA/DEC pointing position from table (`~astropy.coordinates.SkyCoord`).

        Use `get_icrs` to get the pointing at a specific time, correctly
        handling different pointing modes.
        """
        warnings.warn("The radec property is deprecated, use `get_icrs(time, location)` to obtain pointing in ICRS", DeprecationWarning)
        return self._pointing_icrs

    @lazyproperty
    def altaz_frame(self):
        """ALT / AZ frame (`~astropy.coordinates.AltAz`)."""
        warnings.warn("The altaz_frame property is deprecated, use `get_alt(time, location)` to obtain pointing in AltAz", DeprecationWarning)
        return AltAz(obstime=self.obstime, location=self.location)

    @lazyproperty
    def altaz(self):
        """
        ALT/AZ pointing position computed from RA/DEC (`~astropy.coordinates.SkyCoord`)
        for the midpoint of the run.

        Use `get_altaz` to get the pointing at a specific time, correctly
        handling different pointing modes.
        """
        warnings.warn("The altaz property is deprecated, use `get_altaz(time, location)` to obtain pointing in AltAz", DeprecationWarning)
        frame = AltAz(location=self.location, obstime=self.obstime)
        if frame.location is None or frame.obstime is None:
            log.warning(
                "Location or time information missing,"
                " using ALT_PNT/AZ_PNT and incomplete frame"
            )
            if self._pointing_altaz is None:
                return SkyCoord(np.nan, np.nan, unit=u.deg, frame=frame)

            return self._pointing_altaz.copy()

        return self.radec.transform_to(frame)

    @property
    def fixed_altaz(self):
        """The fixed coordinates in AltAz of the observation.

        None if not a DRIFT observation
        """
        if self.mode is PointingMode.POINTING:
            return None
        return self._pointing_altaz

    @lazyproperty
    def fixed_icrs(self):
        """
        The fixed coordinates in ICRS of the observation.

        None if not a POINTING observation
        """
        return self._pointing_icrs

    def get_icrs(self, obstime, location=None) -> SkyCoord:
        """
        Get the pointing position in ICRS frame for a given time.

        If the observation was performed tracking a fixed position in ICRS,
        the icrs pointing is returned with the given obstime attached.

        If the observation was performed in drift mode, the fixed altaz coordinates
        are transformed to ICRS using the observation location and the given time.


        Parameters
        ----------
        obstime: `astropy.time.Time`
            Time for which to get the pointing position in ICRS frame
        location: `astropy.coordinates.EarthLocation`
            Observatory location, only needed for drift observations to transform
            from horizontal coordinates to ICRS.
        """
        if self.mode == PointingMode.POINTING:
            location = location if location is not None else self.location
            return SkyCoord(self._pointing_icrs, location=location, obstime=obstime)

        if self.mode == PointingMode.DRIFT:
            return self.get_altaz(obstime, location=location).icrs

        raise ValueError(f"Unsupported pointing mode: {self.mode}.")

    def get_altaz(self, obstime, location=None) -> SkyCoord:
        """
        Get the pointing position in AltAz frame for a given time.

        If the observation was performed tracking a fixed position in ICRS,
        the icrs pointing is transformed at the given time using the location
        of the observation.

        If the observation was performed in drift mode,
        the fixed altaz coordinate is returned with `obstime` attached.

        Parameters
        ----------
        obstime: `astropy.time.Time`
            Time for which to get the pointing position in AltAz frame
        location: `astropy.coordinates.EarthLocation`
            Observatory location, only needed for pointing observations to transform
            from ICRS to horizontal coordinates.
        """
        location = location if location is not None else self.location
        frame = AltAz(location=location, obstime=obstime)

        if self.mode == PointingMode.POINTING:
            return self.fixed_icrs.transform_to(frame)

        if self.mode == PointingMode.DRIFT:
            # see https://github.com/astropy/astropy/issues/12965
            alt = self.fixed_altaz.alt
            az = self.fixed_altaz.az
            return SkyCoord(
                alt=u.Quantity(np.full(obstime.shape, alt.deg), u.deg, copy=False),
                az=u.Quantity(np.full(obstime.shape, az.deg), u.deg, copy=False),
                frame=frame,
            )

        raise ValueError(f"Unsupported pointing mode: {self.mode}.")


class PointingInfo:
    """IACT array pointing info.

    Data format specification: :ref:`gadf:iact-pnt`

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table (with meta header info) on pointing

    Examples
    --------
    >>> from gammapy.data import PointingInfo
    >>> pointing_info = PointingInfo.read('$GAMMAPY_DATA/tests/pointing_table.fits.gz')
    >>> print(pointing_info)
    Pointing info:
    <BLANKLINE>
    Location:     GeodeticLocation(lon=<Longitude 16.50022222 deg>, lat=<Latitude -23.27177778 deg>, height=<Quantity 1835. m>) # noqa: E501
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
            File name
        hdu : int or str
            HDU number or name

        Returns
        -------
        pointing_info : `PointingInfo`
            Pointing info object
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
        """Observatory location (`~astropy.coordinates.EarthLocation`)."""
        return earth_location_from_dict(self.table.meta)

    @lazyproperty
    def time_ref(self):
        """Time reference (`~astropy.time.Time`)."""
        return time_ref_from_dict(self.table.meta)

    @lazyproperty
    def duration(self):
        """Pointing table duration (`~astropy.time.TimeDelta`).

        The time difference between the first and last entry.
        """
        return self.time[-1] - self.time[0]

    @lazyproperty
    def time(self):
        """Time array (`~astropy.time.Time`)."""
        met = Quantity(self.table["TIME"].astype("float64"), "second")
        time = self.time_ref + met
        return time.tt

    @lazyproperty
    def radec(self):
        """RA / DEC position from table (`~astropy.coordinates.SkyCoord`)."""
        lon = self.table["RA_PNT"]
        lat = self.table["DEC_PNT"]
        return SkyCoord(lon, lat, unit="deg", frame="icrs")

    @lazyproperty
    def altaz_frame(self):
        """ALT / AZ frame (`~astropy.coordinates.AltAz`)."""
        return AltAz(obstime=self.time, location=self.location)

    @lazyproperty
    def altaz(self):
        """ALT / AZ position computed from RA / DEC (`~astropy.coordinates.SkyCoord`)."""
        return self.radec.transform_to(self.altaz_frame)

    @lazyproperty
    def altaz_from_table(self):
        """ALT / AZ position from table (`~astropy.coordinates.SkyCoord`)."""
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
        obstime: `astropy.time.Time`
            Time for which to get the pointing position in ICRS frame
        """
        return SkyCoord(
            self._interpolate_cartesian(self.time.mjd, self.radec, obstime.mjd),
            obstime=obstime,
            frame="icrs",
        )

    def get_altaz(self, obstime):
        """
        Get the pointing position in AltAz frame for a given time.

        If the observation was performed tracking a fixed position in ICRS,
        the icrs pointing is transformed at the given time using the location
        of the observation.

        If the observation was performed in drift mode,
        the fixed altaz coordinate is returned with `obstime` attached.

        Parameters
        ----------
        obstime: `astropy.time.Time`
            Time for which to get the pointing position in AltAz frame
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
