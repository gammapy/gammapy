# Licensed under a 3-clause BSD style license - see LICENSE.rst
import collections.abc
import copy
import logging
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from gammapy.irf import Background3D
from gammapy.utils.fits import LazyFitsData, earth_location_from_dict
from gammapy.utils.testing import Checker
from .event_list import EventListChecker
from .filters import ObservationFilter
from .gti import GTI
from .pointing import FixedPointingInfo

__all__ = ["Observation", "Observations"]

log = logging.getLogger(__name__)


class Observation:
    """In-memory observation.

    Parameters
    ----------
    obs_id : int
        Observation id
    obs_info : dict
        Observation info dict
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion2D`
        Energy dispersion
    psf : `~gammapy.irf.PSF3D`
        Point spread function
    bkg : `~gammapy.irf.Background3D`
        Background rate model
    gti : `~gammapy.data.GTI`
        Table with GTI start and stop time
    events : `~gammapy.data.EventList`
        Event list
    obs_filter : `ObservationFilter`
        Observation filter.
    """

    aeff = LazyFitsData(cache=False)
    edisp = LazyFitsData(cache=False)
    psf = LazyFitsData(cache=False)
    bkg = LazyFitsData(cache=False)
    _events = LazyFitsData(cache=False)
    _gti = LazyFitsData(cache=False)

    def __init__(
        self,
        obs_id=None,
        obs_info=None,
        gti=None,
        aeff=None,
        edisp=None,
        psf=None,
        bkg=None,
        events=None,
        obs_filter=None,
    ):
        self.obs_id = obs_id
        self.obs_info = obs_info
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.bkg = bkg
        self._gti = gti
        self._events = events
        self.obs_filter = obs_filter or ObservationFilter()

    @property
    def events(self):
        events = self.obs_filter.filter_events(self._events)
        return events

    @property
    def gti(self):
        gti = self.obs_filter.filter_gti(self._gti)
        return gti

    @staticmethod
    def _get_obs_info(pointing, deadtime_fraction):
        """Create obs info dict from in memory data"""
        return {
            "RA_PNT": pointing.icrs.ra.deg,
            "DEC_PNT": pointing.icrs.dec.deg,
            "DEADC": 1 - deadtime_fraction,
        }

    @classmethod
    def create(
        cls,
        pointing,
        obs_id=0,
        livetime=None,
        tstart=None,
        tstop=None,
        irfs=None,
        deadtime_fraction=0.0,
        reference_time="2000-01-01",
    ):
        """Create an observation.

        User must either provide the livetime, or the start and stop times.

        Parameters
        ----------
        pointing : `~astropy.coordinates.SkyCoord`
            Pointing position
        obs_id : int
            Observation ID as identifier
        livetime : ~astropy.units.Quantity`
            Livetime exposure of the simulated observation
        tstart : `~astropy.units.Quantity`
            Start time of observation w.r.t reference_time
        tstop : `~astropy.units.Quantity` w.r.t reference_time
            Stop time of observation
        irfs : dict
            IRFs used for simulating the observation: `bkg`, `aeff`, `psf`, `edisp`
        deadtime_fraction : float, optional
            Deadtime fraction, defaults to 0
        reference_time : `~astropy.time.Time`
            the reference time to use in GTI definition

        Returns
        -------
        obs : `gammapy.data.MemoryObservation`
        """
        if tstart is None:
            tstart = Quantity(0.0, "hr")

        if tstop is None:
            tstop = tstart + Quantity(livetime)

        gti = GTI.create([tstart], [tstop], reference_time=reference_time)

        obs_info = cls._get_obs_info(
            pointing=pointing, deadtime_fraction=deadtime_fraction
        )

        return cls(
            obs_id=obs_id,
            obs_info=obs_info,
            gti=gti,
            aeff=irfs.get("aeff"),
            bkg=irfs.get("bkg"),
            edisp=irfs.get("edisp"),
            psf=irfs.get("psf"),
        )

    @property
    def tstart(self):
        """Observation start time (`~astropy.time.Time`)."""
        return self.gti.time_start[0]

    @property
    def tstop(self):
        """Observation stop time (`~astropy.time.Time`)."""
        return self.gti.time_stop[0]

    @property
    def observation_time_duration(self):
        """Observation time duration in seconds (`~astropy.units.Quantity`).

        The wall time, including dead-time.
        """
        return self.gti.time_sum

    @property
    def observation_live_time_duration(self):
        """Live-time duration in seconds (`~astropy.units.Quantity`).

        The dead-time-corrected observation time.

        Computed as ``t_live = t_observation * (1 - f_dead)``
        where ``f_dead`` is the dead-time fraction.
        """
        return self.observation_time_duration * (
            1 - self.observation_dead_time_fraction
        )

    @property
    def observation_dead_time_fraction(self):
        """Dead-time fraction (float).

        Defined as dead-time over observation time.

        Dead-time is defined as the time during the observation
        where the detector didn't record events:
        https://en.wikipedia.org/wiki/Dead_time
        https://ui.adsabs.harvard.edu/abs/2004APh....22..285F

        The dead-time fraction is used in the live-time computation,
        which in turn is used in the exposure and flux computation.
        """
        return 1 - self.obs_info["DEADC"]

    @property
    def pointing_radec(self):
        """Pointing RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        lon, lat = (
            self.obs_info.get("RA_PNT", np.nan),
            self.obs_info.get("DEC_PNT", np.nan),
        )
        return SkyCoord(lon, lat, unit="deg", frame="icrs")

    @property
    def pointing_altaz(self):
        """Pointing ALT / AZ sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        alt, az = (
            self.obs_info.get("ALT_PNT", np.nan),
            self.obs_info.get("AZ_PNT", np.nan),
        )
        return SkyCoord(az, alt, unit="deg", frame="altaz")

    @property
    def pointing_zen(self):
        """Pointing zenith angle sky (`~astropy.units.Quantity`)."""
        return Quantity(self.obs_info.get("ZEN_PNT", np.nan), unit="deg")

    @property
    def fixed_pointing_info(self):
        """Fixed pointing info for this observation (`FixedPointingInfo`)."""
        return FixedPointingInfo(self.events.table.meta)

    @property
    def target_radec(self):
        """Target RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        lon, lat = (
            self.obs_info.get("RA_OBJ", np.nan),
            self.obs_info.get("DEC_OBJ", np.nan),
        )
        return SkyCoord(lon, lat, unit="deg", frame="icrs")

    @property
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)."""
        return earth_location_from_dict(self.obs_info)

    @property
    def muoneff(self):
        """Observation muon efficiency."""
        return self.obs_info.get("MUONEFF", 1)

    def __str__(self):
        ra = self.pointing_radec.ra.deg
        dec = self.pointing_radec.dec.deg

        pointing = f"{ra:.1f} deg, {dec:.1f} deg\n"
        # TODO: Which target was observed?
        # TODO: print info about available HDUs for this observation ...
        return (
            f"{self.__class__.__name__}\n\n"
            f"\tobs id            : {self.obs_id} \n "
            f"\ttstart            : {self.tstart.mjd:.2f}\n"
            f"\ttstop             : {self.tstop.mjd:.2f}\n"
            f"\tduration          : {self.observation_time_duration:.2f}\n"
            f"\tpointing (icrs)   : {pointing}\n"
            f"\tdeadtime fraction : {self.observation_dead_time_fraction:.1%}\n"
        )

    def check(self, checks="all"):
        """Run checks.

        This is a generator that yields a list of dicts.
        """
        checker = ObservationChecker(self)
        return checker.run(checks=checks)

    def peek(self, figsize=(12, 10)):
        """Quick-look plots in a few panels.

        Parameters
        ----------
        figszie : tuple
            Figure size
        """
        import matplotlib.pyplot as plt

        fig, ((ax_aeff, ax_bkg), (ax_psf, ax_edisp)) = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=figsize,
            gridspec_kw={"wspace": 0.25, "hspace": 0.25},
        )

        self.aeff.plot(ax=ax_aeff)

        try:
            if isinstance(self.bkg, Background3D):
                bkg = self.bkg.to_2d()
            else:
                bkg = self.bkg

            bkg.plot(ax=ax_bkg)
        except AttributeError:
            logging.warning(f"No background model found for obs {self.obs_id}.")

        self.psf.plot_containment_vs_energy(ax=ax_psf)
        self.edisp.plot_bias(ax=ax_edisp, add_cbar=True)

        ax_aeff.set_title("Effective area")
        ax_bkg.set_title("Background rate")
        ax_psf.set_title("Point spread function")
        ax_edisp.set_title("Energy dispersion")

    def select_time(self, time_interval):
        """Select a time interval of the observation.

        Parameters
        ----------
        time_interval : `astropy.time.Time`
            Start and stop time of the selected time interval.
            For now we only support a single time interval.

        Returns
        -------
        new_obs : `~gammapy.data.Observation`
            A new observation instance of the specified time interval
        """
        new_obs_filter = self.obs_filter.copy()
        new_obs_filter.time_filter = time_interval
        obs = copy.deepcopy(self)
        obs.obs_filter = new_obs_filter
        return obs


class Observations(collections.abc.MutableSequence):
    """Container class that holds a list of observations.

    Parameters
    ----------
    observations : list
        A list of `~gammapy.data.Observation`
    """

    def __init__(self, observations=None):
        self._observations = observations or []

    def __getitem__(self, key):
        return self._observations[self.index(key)]

    def __delitem__(self, key):
        del self._observations[self.index(key)]

    def __setitem__(self, key, obs):
        if isinstance(obs, Observation):
            self._observations[self.index(key)] = obs
        else:
            raise TypeError(f"Invalid type: {type(obs)!r}")

    def insert(self, idx, obs):
        if isinstance(obs, Observation):
            self._observations.insert(idx, obs)
        else:
            raise TypeError(f"Invalid type: {type(obs)!r}")

    def __len__(self):
        return len(self._observations)

    def __str__(self):
        s = self.__class__.__name__ + "\n"
        s += "Number of observations: {}\n".format(len(self))
        for obs in self:
            s += str(obs)
        return s

    def index(self, key):
        if isinstance(key, (int, slice)):
            return key
        elif isinstance(key, str):
            return self.ids.index(key)
        elif isinstance(key, Observation):
            return self._observations.index(key)
        else:
            raise TypeError(f"Invalid type: {type(key)!r}")

    @property
    def ids(self):
        """List of obs IDs (`list`)"""
        return [str(obs.obs_id) for obs in self]

    def select_time(self, time_intervals):
        """Select a time interval of the observations.

        Parameters
        ----------
        time_intervals : `astropy.time.Time` or list of `astropy.time.Time`
            list of Start and stop time of the time intervals or one Time interval

        Returns
        -------
        new_observations : `~gammapy.data.Observations`
            A new Observations instance of the specified time intervals
        """
        new_obs_list = []
        if isinstance(time_intervals, Time):
            time_intervals = [time_intervals]

        for time_interval in time_intervals:
            for obs in self:
                if (obs.tstart < time_interval[1]) & (obs.tstop > time_interval[0]):
                    new_obs = obs.select_time(time_interval)
                    new_obs_list.append(new_obs)

        return self.__class__(new_obs_list)

    def _ipython_key_completions_(self):
        return self.ids


class ObservationChecker(Checker):
    """Check an observation.

    Checks data format and a bit about the content.
    """

    CHECKS = {
        "events": "check_events",
        "gti": "check_gti",
        "aeff": "check_aeff",
        "edisp": "check_edisp",
        "psf": "check_psf",
    }

    def __init__(self, observation):
        self.observation = observation

    def _record(self, level="info", msg=None):
        return {"level": level, "obs_id": self.observation.obs_id, "msg": msg}

    def check_events(self):
        yield self._record(level="debug", msg="Starting events check")

        try:
            events = self.observation.events
        except Exception:
            yield self._record(level="warning", msg="Loading events failed")
            return

        yield from EventListChecker(events).run()

    # TODO: split this out into a GTIChecker
    def check_gti(self):
        yield self._record(level="debug", msg="Starting gti check")

        try:
            gti = self.observation.gti
        except Exception:
            yield self._record(level="warning", msg="Loading GTI failed")
            return

        if len(gti.table) == 0:
            yield self._record(level="error", msg="GTI table has zero rows")

        columns_required = ["START", "STOP"]
        for name in columns_required:
            if name not in gti.table.colnames:
                yield self._record(level="error", msg=f"Missing table column: {name!r}")

        # TODO: Check that header keywords agree with table entries
        # TSTART, TSTOP, MJDREFI, MJDREFF

        # Check that START and STOP times are consecutive
        # times = np.ravel(self.table['START'], self.table['STOP'])
        # # TODO: not sure this is correct ... add test with a multi-gti table from Fermi.
        # if not np.all(np.diff(times) >= 0):
        #     yield 'GTIs are not consecutive or sorted.'

    # TODO: add reference times for all instruments and check for this
    # Use TELESCOP header key to check which instrument it is.
    def _check_times(self):
        """Check if various times are consistent.

        The headers and tables of the FITS EVENTS and GTI extension
        contain various observation and event time information.
        """
        # http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/Time_in_ScienceTools.html
        # https://hess-confluence.desy.de/confluence/display/HESS/HESS+FITS+data+-+References+and+checks#HESSFITSdata-Referencesandchecks-Time
        telescope_met_refs = {
            "FERMI": Time("2001-01-01T00:00:00"),
            "HESS": Time("2001-01-01T00:00:00"),
        }

        meta = self.dset.event_list.table.meta
        telescope = meta["TELESCOP"]

        if telescope in telescope_met_refs.keys():
            dt = self.time_ref - telescope_met_refs[telescope]
            if dt > self.accuracy["time"]:
                yield self._record(
                    level="error", msg="Reference time incorrect for telescope"
                )

    def check_aeff(self):
        yield self._record(level="debug", msg="Starting aeff check")

        try:
            aeff = self.observation.aeff
        except Exception:
            yield self._record(level="warning", msg="Loading aeff failed")
            return

        # Check that thresholds are meaningful for aeff
        if (
            "LO_THRES" in aeff.meta
            and "HI_THRES" in aeff.meta
            and aeff.meta["LO_THRES"] >= aeff.meta["HI_THRES"]
        ):
            yield self._record(
                level="error", msg="LO_THRES >= HI_THRES in effective area meta data"
            )

        # Check that data isn't all null
        if np.max(aeff.data.data) <= 0:
            yield self._record(
                level="error", msg="maximum entry of effective area is <= 0"
            )

    def check_edisp(self):
        yield self._record(level="debug", msg="Starting edisp check")

        try:
            edisp = self.observation.edisp
        except Exception:
            yield self._record(level="warning", msg="Loading edisp failed")
            return

        # Check that data isn't all null
        if np.max(edisp.data.data) <= 0:
            yield self._record(level="error", msg="maximum entry of edisp is <= 0")

    def check_psf(self):
        yield self._record(level="debug", msg="Starting psf check")

        try:
            self.observation.psf
        except Exception:
            yield self._record(level="warning", msg="Loading psf failed")
            return

        # TODO: implement some basic check
        # The following doesn't work, because EnergyDependentMultiGaussPSF
        # has no attribute `data`
        # Check that data isn't all null
        # if np.max(psf.data.data) <= 0:
        #     yield self._record(
        #         level="error", msg="maximum entry of psf is <= 0"
        #     )
