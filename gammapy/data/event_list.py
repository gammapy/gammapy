# Licensed under a 3-clause BSD style license - see LICENSE.rst
import collections
import logging
import numpy as np
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.coordinates.angle_utilities import angular_separation
from astropy.table import Table
from astropy.table import vstack as vstack_tables
from astropy.units import Quantity, Unit
from gammapy.maps import MapCoord, WcsNDMap
from gammapy.utils.energy import energy_logspace
from gammapy.utils.fits import earth_location_from_dict
from gammapy.utils.regions import make_region
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import Checker
from gammapy.utils.time import time_ref_from_dict

__all__ = ["EventListBase", "EventList", "EventListLAT"]

log = logging.getLogger(__name__)


class EventListBase:
    """Event list.

    This class represents the base for two different event lists:
    - EventList: targeted for IACT event lists
    - EventListLAT: targeted for Fermi-LAT event lists

    Event list data is stored in ``table`` (`~astropy.table.Table`) data member.

    The most important reconstructed event parameters
    are available as the following columns:

    - ``TIME`` - Mission elapsed time (sec)
    - ``RA``, ``DEC`` - ICRS system position (deg)
    - ``ENERGY`` - Energy (usually MeV for Fermi and TeV for IACTs)

    Other optional (columns) that are sometimes useful for high-level analysis:

    - ``GLON``, ``GLAT`` - Galactic coordinates (deg)
    - ``DETX``, ``DETY`` - Field of view coordinates (deg)

    Note that when reading data for analysis you shouldn't use those
    values directly, but access them via properties which create objects
    of the appropriate class:

    - `time` for ``TIME``
    - `radec` for ``RA``, ``DEC``
    - `energy` for ``ENERGY``
    - `galactic` for ``GLON``, ``GLAT``

    Parameters
    ----------
    table : `~astropy.table.Table`
        Event list table
    """

    def __init__(self, table):
        self.table = table

    @classmethod
    def read(cls, filename, **kwargs):
        """Read from FITS file.

        Format specification: :ref:`gadf:iact-events`

        Parameters
        ----------
        filename : `pathlib.Path`, str
            Filename
        """
        filename = make_path(filename)
        kwargs.setdefault("hdu", "EVENTS")
        table = Table.read(filename, **kwargs)
        return cls(table=table)

    @classmethod
    def stack(cls, event_lists, **kwargs):
        """Stack (concatenate) list of event lists.

        Calls `~astropy.table.vstack`.

        Parameters
        ----------
        event_lists : list
            list of `~gammapy.data.EventList` to stack
        """
        tables = [_.table for _ in event_lists]
        stacked_table = vstack_tables(tables, **kwargs)
        return cls(stacked_table)

    def __str__(self):
        ss = (
            "EventList info:\n"
            f"- Number of events: {len(self.table)}\n"
            f"- Median energy: {np.median(self.energy.value):.3g} {self.energy.unit}\n"
        )

        if "OBS_ID" in self.table.meta:
            ss += "- OBS_ID = {}".format(self.table.meta["OBS_ID"])

        # TODO: add time, RA, DEC and if present GLON, GLAT info ...

        if "AZ" in self.table.colnames:
            # TODO: azimuth should be circular median
            ss += "- Median azimuth: {}\n".format(np.median(self.table["AZ"]))

        if "ALT" in self.table.colnames:
            ss += "- Median altitude: {}\n".format(np.median(self.table["ALT"]))

        return ss

    @property
    def time_ref(self):
        """Time reference (`~astropy.time.Time`)."""
        return time_ref_from_dict(self.table.meta)

    @property
    def time(self):
        """Event times (`~astropy.time.Time`).

        Notes
        -----
        Times are automatically converted to 64-bit floats.
        With 32-bit floats times will be incorrect by a few seconds
        when e.g. adding them to the reference time.
        """
        met = Quantity(self.table["TIME"].astype("float64"), "second")
        return self.time_ref + met

    @property
    def observation_time_start(self):
        """Observation start time (`~astropy.time.Time`)."""
        return self.time_ref + Quantity(self.table.meta["TSTART"], "second")

    @property
    def observation_time_end(self):
        """Observation stop time (`~astropy.time.Time`)."""
        return self.time_ref + Quantity(self.table.meta["TSTOP"], "second")

    @property
    def radec(self):
        """Event RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        lon, lat = self.table["RA"], self.table["DEC"]
        return SkyCoord(lon, lat, unit="deg", frame="icrs")

    @property
    def galactic(self):
        """Event Galactic sky coordinates (`~astropy.coordinates.SkyCoord`).

        Always computed from RA / DEC using Astropy.
        """
        return self.radec.galactic

    @property
    def energy(self):
        """Event energies (`~astropy.units.Quantity`)."""
        return self.table["ENERGY"].quantity

    def select_row_subset(self, row_specifier):
        """Select table row subset.

        Parameters
        ----------
        row_specifier : slice, int, or array of ints
            Specification for rows to select,
            passed on to ``self.table[row_specifier]``.

        Returns
        -------
        event_list : `EventList`
            New event list with table row subset selected

        Examples
        --------
        Use a boolean mask as ``row_specifier``:

            mask = events.table['FOO'] > 42
            events2 = events.select_row_subset(mask)

        Use row index array as ``row_specifier``:

            idx = np.where(events.table['FOO'] > 42)[0]
            events2 = events.select_row_subset(idx)
        """
        table = self.table[row_specifier]
        return self.__class__(table=table)

    def select_energy(self, energy_band):
        """Select events in energy band.

        Parameters
        ----------
        energy_band : `~astropy.units.Quantity`
            Energy band ``[energy_min, energy_max)``

        Returns
        -------
        event_list : `EventList`
            Copy of event list with selection applied.

        Examples
        --------
        >>> from astropy.units import Quantity
        >>> from gammapy.data import EventList
        >>> event_list = EventList.read('events.fits')
        >>> energy_band = Quantity([1, 20], 'TeV')
        >>> event_list = event_list.select_energy()
        """
        energy = self.energy
        mask = energy_band[0] <= energy
        mask &= energy < energy_band[1]
        return self.select_row_subset(mask)

    def select_time(self, time_interval):
        """Select events in time interval.

        Parameters
        ----------
        time_interval : `astropy.time.Time`
            Start time (inclusive) and stop time (exclusive) for the selection.

        Returns
        -------
        events : `EventList`
            Copy of event list with selection applied.
        """
        time = self.time
        mask = time_interval[0] <= time
        mask &= time < time_interval[1]
        return self.select_row_subset(mask)

    def select_region(self, region, wcs=None):
        """Select events in given region.

        Parameters
        ----------
        region : `~regions.SkyRegion` or str
            Sky region or string defining a sky region
        wcs : `~astropy.wcs.WCS`
            World coordinate system transformation

        Returns
        -------
        event_list : `EventList`
            Copy of event list with selection applied.
        """
        region = make_region(region)
        mask = region.contains(self.radec, wcs)
        return self.select_row_subset(mask)

    def select_parameter(self, parameter, band):
        """Select events with respect to a specified parameter.

        Parameters
        ----------
        parameter : str
            Parameter used for the selection. Must be present in `self.table`.
        band : tuple or `astropy.units.Quantity`
            Min and max value for the parameter to be selected (min <= parameter < max).
            If parameter is not dimensionless you have to provide a Quantity.

        Returns
        -------
        event_list : `EventList`
            Copy of event list with selection applied.

        Examples
        --------
        >>> from gammapy.data import EventList
        >>> event_list = EventList.read('events.fits')
        >>> phase_region = (0.3, 0.5)
        >>> event_list = event_list.select_parameter(parameter='PHASE', band=phase_region)
        """
        mask = band[0] <= self.table[parameter].quantity
        mask &= self.table[parameter].quantity < band[1]
        return self.select_row_subset(mask)

    def _default_plot_ebounds(self):
        energy = self.energy
        return energy_logspace(energy.min(), energy.max(), 50)

    def _counts_spectrum(self, ebounds):
        from gammapy.spectrum import CountsSpectrum

        if not ebounds:
            ebounds = self._default_plot_ebounds()
        spec = CountsSpectrum(energy_lo=ebounds[:-1], energy_hi=ebounds[1:])
        spec.fill_energy(self.energy)
        return spec

    def plot_energy(self, ax=None, ebounds=None, **kwargs):
        """Plot counts as a function of energy."""
        spec = self._counts_spectrum(ebounds)
        ax = spec.plot(ax=ax, **kwargs)
        return ax

    def plot_time(self, ax=None):
        """Plots an event rate time curve.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None
            Axes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        # Note the events are not necessarily in time order
        time = self.table["TIME"]
        time = time - np.min(time)

        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Counts")
        y, x_edges = np.histogram(time, bins=30)
        # x = (x_edges[1:] + x_edges[:-1]) / 2
        xerr = np.diff(x_edges) / 2
        x = x_edges[:-1] + xerr
        yerr = np.sqrt(y)

        ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, fmt="none")

        return ax

    def plot_offset2_distribution(self, ax=None, center=None, **kwargs):
        """Plot offset^2 distribution of the events.

        The distribution shown in this plot is for this quantity::

            offset = center.separation(events.radec).deg
            offset2 = offset ** 2

        Note that this method is just for a quicklook plot.

        If you want to do computations with the offset or offset^2 values, you can
        use the line above. As an example, here's how to compute the 68% event
        containment radius using `numpy.percentile`::

            import numpy as np
            r68 = np.percentile(offset, q=68)

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` (optional)
            Axes
        center : `astropy.coordinates.SkyCoord`
            Center position for the offset^2 distribution.
            Default is the observation pointing position.
        **kwargs :
            Extra keyword arguments are passed to `matplotlib.pyplot.hist`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes

        Examples
        --------
        Load an example event list:

        >>> from gammapy.data import EventList
        >>> events = EventList.read('$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz')

        Plot the offset^2 distribution wrt. the observation pointing position
        (this is a commonly used plot to check the background spatial distribution):

        >>> events.plot_offset2_distribution()

        Plot the offset^2 distribution wrt. the Crab pulsar position
        (this is commonly used to check both the gamma-ray signal and the background spatial distribution):

        >>> import numpy as np
        >>> from astropy.coordinates import SkyCoord
        >>> center = SkyCoord(83.63307, 22.01449, unit='deg')
        >>> bins = np.linspace(start=0, stop=0.3 ** 2, num=30)
        >>> events.plot_offset2_distribution(center=center, bins=bins)

        Note how we passed the ``bins`` option of `matplotlib.pyplot.hist` to control the histogram binning,
        in this case 30 bins ranging from 0 to (0.3 deg)^2.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if center is None:
            center = self.pointing_radec

        offset2 = center.separation(self.radec).deg ** 2

        ax.hist(offset2, **kwargs)
        ax.set_xlabel("Offset^2 (deg^2)")
        ax.set_ylabel("Counts")

        return ax

    def plot_energy_offset(self, ax=None):
        """Plot counts histogram with energy and offset axes."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        ax = plt.gca() if ax is None else ax

        energy_bounds = self._default_plot_ebounds().to_value("TeV")
        offset_bounds = np.linspace(0, 4, 30)

        counts = np.histogram2d(
            x=self.energy.value,
            y=self.offset.value,
            bins=(energy_bounds, offset_bounds),
        )[0]

        ax.pcolormesh(energy_bounds, offset_bounds, counts.T, norm=LogNorm())
        ax.set_xscale("log")
        ax.set_xlabel(f"Energy ({self.energy.unit})")
        ax.set_ylabel(f"Offset ({self.offset.unit})")

    def check(self, checks="all"):
        """Run checks.

        This is a generator that yields a list of dicts.
        """
        checker = EventListChecker(self)
        return checker.run(checks=checks)

    def map_coord(self, geom):
        """Event map coordinates for a given geometry.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Geometry

        Returns
        -------
        coord : `~gammapy.maps.MapCoord`
            Coordinates
        """
        coord = {"skycoord": self.radec}

        cols = {k.upper(): v for k, v in self.table.columns.items()}

        for axis in geom.axes:
            try:
                col = cols[axis.name.upper()]
                coord[axis.name] = Quantity(col).to(axis.unit)
            except KeyError:
                raise KeyError(f"Column not found in event list: {axis.name!r}")

        return MapCoord.create(coord)

    def select_map_mask(self, mask):
        """Select events inside a mask (`EventList`).

        Parameters
        ----------
        mask : `~gammapy.maps.Map`
            Mask
        """
        coord = self.map_coord(mask.geom)
        values = mask.get_by_coord(coord)
        valid = values > 0
        return self.select_row_subset(valid)


class EventList(EventListBase):
    """Event list for IACT dataset.

    Data format specification: :ref:`gadf:iact-events`

    For further information, see the base class: `~gammapy.data.EventListBase`.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Event list table

    Examples
    --------
    To load an example H.E.S.S. event list:

    >>> from gammapy.data import EventList
    >>> filename = '$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz'
    >>> events = EventList.read(filename)
    """

    @property
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)."""
        return earth_location_from_dict(self.table.meta)

    @property
    def observation_time_duration(self):
        """Observation time duration in seconds (`~astropy.units.Quantity`).

        This is a keyword related to IACTs
        The wall time, including dead-time.
        """
        return Quantity(self.table.meta["ONTIME"], "second")

    @property
    def observation_live_time_duration(self):
        """Live-time duration in seconds (`~astropy.units.Quantity`).

        The dead-time-corrected observation time.

        - In Fermi-LAT it is automatically provided in the header of the event list.
        - In IACTs is computed as ``t_live = t_observation * (1 - f_dead)``

        where ``f_dead`` is the dead-time fraction.
        """
        return Quantity(self.table.meta["LIVETIME"], "second")

    @property
    def observation_dead_time_fraction(self):
        """Dead-time fraction (float).

        This is a keyword related to IACTs
        Defined as dead-time over observation time.

        Dead-time is defined as the time during the observation
        where the detector didn't record events:
        http://en.wikipedia.org/wiki/Dead_time
        https://ui.adsabs.harvard.edu/abs/2004APh....22..285F

        The dead-time fraction is used in the live-time computation,
        which in turn is used in the exposure and flux computation.
        """
        return 1 - self.table.meta["DEADC"]

    @property
    def altaz_frame(self):
        """ALT / AZ frame (`~astropy.coordinates.AltAz`)."""
        return AltAz(obstime=self.time, location=self.observatory_earth_location)

    @property
    def altaz(self):
        """ALT / AZ position computed from RA / DEC (`~astropy.coordinates.SkyCoord`)."""
        return self.radec.transform_to(self.altaz_frame)

    @property
    def altaz_from_table(self):
        """ALT / AZ position from table (`~astropy.coordinates.SkyCoord`)."""
        lon = self.table["AZ"]
        lat = self.table["ALT"]
        return SkyCoord(lon, lat, unit="deg", frame=self.altaz_frame)

    @property
    def pointing_radec(self):
        """Pointing RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        info = self.table.meta
        lon, lat = info["RA_PNT"], info["DEC_PNT"]
        return SkyCoord(lon, lat, unit="deg", frame="icrs")

    @property
    def offset(self):
        """Event offset from the array pointing position (`~astropy.coordinates.Angle`)."""
        position = self.radec
        center = self.pointing_radec
        offset = center.separation(position)
        return Angle(offset, unit="deg")

    def select_offset(self, offset_band):
        """Select events in offset band.

        Parameters
        ----------
        offset_band : `~astropy.coordinates.Angle`
            offset band ``[offset_min, offset_max)``

        Returns
        -------
        event_list : `EventList`
            Copy of event list with selection applied.
        """
        offset = self.offset
        mask = offset_band[0] <= offset
        mask &= offset < offset_band[1]
        return self.select_row_subset(mask)

    def peek(self):
        """Quick look plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

        self.plot_energy(ax=axes[0, 0])
        bins = np.linspace(start=0, stop=4 ** 2, num=30)
        self.plot_offset2_distribution(ax=axes[0, 1], bins=bins)
        self.plot_time(ax=axes[0, 2])

        axes[1, 0].axis("off")
        m = self._counts_image()
        ax = plt.subplot(2, 3, 4, projection=m.geom.wcs)
        m.plot(ax=ax, stretch="sqrt")

        self.plot_energy_offset(ax=axes[1, 1])

        self._plot_text_summary(ax=axes[1, 2])

        plt.tight_layout()

    def _plot_text_summary(self, ax):
        ax.axis("off")
        txt = str(self)
        ax.text(0, 1, txt, fontsize=12, verticalalignment="top")

    def _counts_image(self):
        opts = {
            "width": (7, 7),
            "binsz": 0.1,
            "proj": "TAN",
            "coordsys": "GAL",
            "skydir": self.pointing_radec,
        }
        m = WcsNDMap.create(**opts)
        m.fill_by_coord(self.radec)
        m = m.smooth(width=1)
        return m

    def plot_image(self):
        """Quick look counts map sky plot."""
        m = self._counts_image()
        m.plot(stretch="sqrt")


class EventListLAT(EventListBase):
    """Event list for Fermi-LAT dataset.

    Fermi-LAT data products
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/LAT_DP.html
    Data format specification (columns)
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/LAT_Data_Columns.html

    For further information, see the base class: `~gammapy.data.EventListBase`.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Event list table

    Examples
    --------
    To load an example Fermi-LAT event list:

    >>> from gammapy.data import EventListLAT
    >>> filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"
    >>> events = EventListLAT.read(filename)
    """

    def plot_image(self):
        """Quick look counts map sky plot."""
        from gammapy.maps import WcsNDMap

        m = WcsNDMap.create(npix=(360, 180), binsz=1.0, proj="AIT", coordsys="GAL")
        m.fill_by_coord(self.radec)
        m.plot(stretch="sqrt")


class EventListChecker(Checker):
    """Event list checker.

    Data format specification: ref:`gadf:iact-events`

    Parameters
    ----------
    event_list : `~gammapy.data.EventList`
        Event list
    """

    CHECKS = {
        "meta": "check_meta",
        "columns": "check_columns",
        "times": "check_times",
        "coordinates_galactic": "check_coordinates_galactic",
        "coordinates_altaz": "check_coordinates_altaz",
    }

    accuracy = {"angle": Angle("1 arcsec"), "time": Quantity(1, "microsecond")}

    # https://gamma-astro-data-formats.readthedocs.io/en/latest/events/events.html#mandatory-header-keywords
    meta_required = [
        "HDUCLASS",
        "HDUDOC",
        "HDUVERS",
        "HDUCLAS1",
        "OBS_ID",
        "TSTART",
        "TSTOP",
        "ONTIME",
        "LIVETIME",
        "DEADC",
        "RA_PNT",
        "DEC_PNT",
        # TODO: what to do about these?
        # They are currently listed as required in the spec,
        # but I think we should just require ICRS and those
        # are irrelevant, should not be used.
        # 'RADECSYS',
        # 'EQUINOX',
        "ORIGIN",
        "TELESCOP",
        "INSTRUME",
        "CREATOR",
        # https://gamma-astro-data-formats.readthedocs.io/en/latest/general/time.html#time-formats
        "MJDREFI",
        "MJDREFF",
        "TIMEUNIT",
        "TIMESYS",
        "TIMEREF",
        # https://gamma-astro-data-formats.readthedocs.io/en/latest/general/coordinates.html#coords-location
        "GEOLON",
        "GEOLAT",
        "ALTITUDE",
    ]

    _col = collections.namedtuple("col", ["name", "unit"])
    columns_required = [
        _col(name="EVENT_ID", unit=""),
        _col(name="TIME", unit="s"),
        _col(name="RA", unit="deg"),
        _col(name="DEC", unit="deg"),
        _col(name="ENERGY", unit="TeV"),
    ]

    def __init__(self, event_list):
        self.event_list = event_list

    def _record(self, level="info", msg=None):
        obs_id = self.event_list.table.meta["OBS_ID"]
        return {"level": level, "obs_id": obs_id, "msg": msg}

    def check_meta(self):
        meta_missing = sorted(set(self.meta_required) - set(self.event_list.table.meta))
        if meta_missing:
            yield self._record(
                level="error", msg=f"Missing meta keys: {meta_missing!r}"
            )

    def check_columns(self):
        t = self.event_list.table

        if len(t) == 0:
            yield self._record(level="error", msg="Events table has zero rows")

        for name, unit in self.columns_required:
            if name not in t.colnames:
                yield self._record(level="error", msg=f"Missing table column: {name!r}")
            else:
                if Unit(unit) != (t[name].unit or ""):
                    yield self._record(
                        level="error", msg=f"Invalid unit for column: {name!r}"
                    )

    def check_times(self):
        dt = (self.event_list.time - self.event_list.observation_time_start).sec
        if dt.min() < self.accuracy["time"].to_value("s"):
            yield self._record(level="error", msg="Event times before obs start time")

        dt = (self.event_list.time - self.event_list.observation_time_end).sec
        if dt.max() > self.accuracy["time"].to_value("s"):
            yield self._record(level="error", msg="Event times after the obs end time")

        if np.min(np.diff(dt)) <= 0:
            yield self._record(level="error", msg="Events are not time-ordered.")

    def check_coordinates_galactic(self):
        """Check if RA / DEC matches GLON / GLAT."""
        t = self.event_list.table

        if "GLON" not in t.colnames:
            return

        galactic = SkyCoord(t["GLON"], t["GLAT"], unit="deg", frame="galactic")
        separation = self.event_list.radec.separation(galactic).to("arcsec")
        if separation.max() > self.accuracy["angle"]:
            yield self._record(
                level="error", msg="GLON / GLAT not consistent with RA / DEC"
            )

    def check_coordinates_altaz(self):
        """Check if ALT / AZ matches RA / DEC."""
        t = self.event_list.table

        if "AZ" not in t.colnames:
            return

        altaz_astropy = self.event_list.altaz
        separation = angular_separation(
            altaz_astropy.data.lon,
            altaz_astropy.data.lat,
            t["AZ"].quantity,
            t["ALT"].quantity,
        )
        if separation.max() > self.accuracy["angle"]:
            yield self._record(
                level="error", msg="ALT / AZ not consistent with RA / DEC"
            )
