# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from collections import OrderedDict
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.time import Time
from .event_list import EventListChecker
from ..utils.testing import Checker
from ..utils.fits import earth_location_from_dict
from ..utils.table import table_row_to_dict
from ..utils.time import time_ref_from_dict
from .filters import ObservationFilter
from .pointing import FixedPointingInfo

__all__ = ["DataStoreObservation", "Observations"]

log = logging.getLogger(__name__)


class DataStoreObservation:
    """IACT data store observation.

    Parameters
    ----------
    obs_id : int
        Observation ID
    data_store : `~gammapy.data.DataStore`
        Data store
    obs_filter : `~gammapy.data.ObservationFilter`, optional
        Filter for the observation
    """

    def __init__(self, obs_id, data_store, obs_filter=None):
        # Assert that `obs_id` is available
        if obs_id not in data_store.obs_table["OBS_ID"]:
            raise ValueError("OBS_ID = {} not in obs index table.".format(obs_id))
        if obs_id not in data_store.hdu_table["OBS_ID"]:
            raise ValueError("OBS_ID = {} not in HDU index table.".format(obs_id))

        self.obs_id = obs_id
        self.data_store = data_store
        self.obs_filter = obs_filter or ObservationFilter()

    def __str__(self):
        ss = "Info for OBS_ID = {}\n".format(self.obs_id)
        ss += "- Start time: {:.2f}\n".format(self.tstart.mjd)
        ss += "- Pointing pos: RA {:.2f} / Dec {:.2f}\n".format(
            self.pointing_radec.ra, self.pointing_radec.dec
        )
        ss += "- Observation duration: {}\n".format(self.observation_time_duration)
        ss += "- Dead-time fraction: {:5.3f} %\n".format(
            100 * self.observation_dead_time_fraction
        )

        # TODO: Which target was observed?
        # TODO: print info about available HDUs for this observation ...
        return ss

    def location(self, hdu_type=None, hdu_class=None):
        """HDU location object.

        Parameters
        ----------
        hdu_type : str
            HDU type (see `~gammapy.data.HDUIndexTable.VALID_HDU_TYPE`)
        hdu_class : str
            HDU class (see `~gammapy.data.HDUIndexTable.VALID_HDU_CLASS`)

        Returns
        -------
        location : `~gammapy.data.HDULocation`
            HDU location
        """
        return self.data_store.hdu_table.hdu_location(
            obs_id=self.obs_id, hdu_type=hdu_type, hdu_class=hdu_class
        )

    def load(self, hdu_type=None, hdu_class=None):
        """Load data file as appropriate object.

        Parameters
        ----------
        hdu_type : str
            HDU type (see `~gammapy.data.HDUIndexTable.VALID_HDU_TYPE`)
        hdu_class : str
            HDU class (see `~gammapy.data.HDUIndexTable.VALID_HDU_CLASS`)

        Returns
        -------
        object : object
            Object depends on type, e.g. for `events` it's a `~gammapy.data.EventList`.
        """
        location = self.location(hdu_type=hdu_type, hdu_class=hdu_class)
        return location.load()

    @property
    def events(self):
        """Load `gammapy.data.EventList` object and apply the filter."""
        events = self.load(hdu_type="events")
        return self.obs_filter.filter_events(events)

    @property
    def gti(self):
        """Load `gammapy.data.GTI` object and apply the filter."""
        try:
            gti = self.load(hdu_type="gti")
        except IndexError:
            # For now we support data without GTI HDUs
            # TODO: if GTI becomes required, we should drop this case
            # CTA discussion in https://github.com/open-gamma-ray-astro/gamma-astro-data-formats/issues/20
            # Added in Gammapy in https://github.com/gammapy/gammapy/pull/1908
            gti = self.data_store.obs_table.create_gti(obs_id=self.obs_id)

        return self.obs_filter.filter_gti(gti)

    @property
    def aeff(self):
        """Load effective area object."""
        return self.load(hdu_type="aeff")

    @property
    def edisp(self):
        """Load energy dispersion object."""
        return self.load(hdu_type="edisp")

    @property
    def psf(self):
        """Load point spread function object."""
        return self.load(hdu_type="psf")

    @property
    def bkg(self):
        """Load background object."""
        return self.load(hdu_type="bkg")

    @property
    def obs_info(self):
        """Observation information (`~collections.OrderedDict`)."""
        row = self.data_store.obs_table.select_obs_id(obs_id=self.obs_id)[0]
        return table_row_to_dict(row)

    @property
    def tstart(self):
        """Observation start time (`~astropy.time.Time`)."""
        met_ref = time_ref_from_dict(self.data_store.obs_table.meta)
        met = Quantity(self.obs_info["TSTART"].astype("float64"), "second")
        time = met_ref + met
        return time

    @property
    def tstop(self):
        """Observation stop time (`~astropy.time.Time`)."""
        met_ref = time_ref_from_dict(self.data_store.obs_table.meta)
        met = Quantity(self.obs_info["TSTOP"].astype("float64"), "second")
        time = met_ref + met
        return time

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
        return self.gti.time_sum * (1 - self.observation_dead_time_fraction)

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
        lon, lat = self.obs_info["RA_PNT"], self.obs_info["DEC_PNT"]
        return SkyCoord(lon, lat, unit="deg", frame="icrs")

    @property
    def pointing_altaz(self):
        """Pointing ALT / AZ sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        alt, az = self.obs_info["ALT_PNT"], self.obs_info["AZ_PNT"]
        return SkyCoord(az, alt, unit="deg", frame="altaz")

    @property
    def pointing_zen(self):
        """Pointing zenith angle sky (`~astropy.units.Quantity`)."""
        return Quantity(self.obs_info["ZEN_PNT"], unit="deg")

    @property
    def fixed_pointing_info(self):
        """Fixed pointing info for this observation (`FixedPointingInfo`)."""
        return FixedPointingInfo(self.events.table.meta)

    @property
    def target_radec(self):
        """Target RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        lon, lat = self.obs_info["RA_OBJ"], self.obs_info["DEC_OBJ"]
        return SkyCoord(lon, lat, unit="deg", frame="icrs")

    @property
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)."""
        return earth_location_from_dict(self.obs_info)

    @property
    def muoneff(self):
        """Observation muon efficiency."""
        return self.obs_info["MUONEFF"]

    def peek(self):
        """Quick-look plots in a few panels."""
        raise NotImplementedError

    def select_time(self, time_interval):
        """Select a time interval of the observation.

        Parameters
        ----------
        time_interval : `astropy.time.Time`
            Start and stop time of the selected time interval.
            For now we only support a single time interval.

        Returns
        -------
        new_obs : `~gammapy.data.DataStoreObservation`
            A new observation instance of the specified time interval
        """
        new_obs_filter = self.obs_filter.copy()
        new_obs_filter.time_filter = time_interval

        return self.__class__(
            obs_id=self.obs_id, data_store=self.data_store, obs_filter=new_obs_filter
        )

    def check(self, checks="all"):
        """Run checks.

        This is a generator that yields a list of dicts.
        """
        checker = ObservationChecker(self)
        return checker.run(checks=checks)


class Observations:
    """Container class that holds a list of observations.

    Parameters
    ----------
    obs_list : list
        A list of `~gammapy.data.DataStoreObservation`
    """

    def __init__(self, obs_list=None):
        self.list = obs_list or []

    def __getitem__(self, key):
        return self.list[key]

    def __len__(self):
        return len(self.list)

    def __str__(self):
        s = self.__class__.__name__ + "\n"
        s += "Number of observations: {}\n".format(len(self))
        for obs in self:
            s += str(obs)
        return s

    def select_time(self, time_interval):
        """Select a time interval of the observations.

        Parameters
        ----------
        time_interval : `astropy.time.Time`
            Start and stop time of the selected time interval.
            For now we only support a single time interval.

        Returns
        -------
        new_observations : `~gammapy.data.Observations`
            A new observations instance of the specified time interval
        """
        new_obs_list = []
        for obs in self:
            new_obs = obs.select_time(time_interval)
            if len(new_obs.gti.table) > 0:
                new_obs_list.append(new_obs)

        return self.__class__(new_obs_list)


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
            events = self.observation.load("events")
        except Exception:
            yield self._record(level="warning", msg="Loading events failed")
            return

        yield from EventListChecker(events).run()

    # TODO: split this out into a GTIChecker
    def check_gti(self):
        yield self._record(level="debug", msg="Starting gti check")

        try:
            gti = self.observation.load("gti")
        except Exception:
            yield self._record(level="warning", msg="Loading GTI failed")
            return

        if len(gti.table) == 0:
            yield self._record(level="error", msg="GTI table has zero rows")

        columns_required = ["START", "STOP"]
        for name in columns_required:
            if name not in gti.table.colnames:
                yield self._record(
                    level="error", msg="Missing table column: {!r}".format(name)
                )

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
        telescope_met_refs = OrderedDict(
            FERMI=Time("2001-01-01T00:00:00"), HESS=Time("2001-01-01T00:00:00")
        )

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
            aeff = self.observation.load("aeff")
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
            edisp = self.observation.load("edisp")
        except Exception:
            yield self._record(level="warning", msg="Loading edisp failed")
            return

        # Check that data isn't all null
        if np.max(edisp.data.data) <= 0:
            yield self._record(level="error", msg="maximum entry of edisp is <= 0")

    def check_psf(self):
        yield self._record(level="debug", msg="Starting psf check")

        try:
            self.observation.load("psf")
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
