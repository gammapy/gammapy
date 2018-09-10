# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from collections import OrderedDict
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.utils import lazyproperty
from ..extern.six.moves import UserList  # pylint:disable=import-error
from ..irf import EnergyDependentTablePSF, PSF3D, IRFStacker
from .event_list import EventListChecker
from ..utils.testing import Checker
from ..utils.energy import Energy
from ..utils.fits import earth_location_from_dict
from ..utils.table import table_row_to_dict
from ..utils.time import time_ref_from_dict

__all__ = ["ObservationCTA", "DataStoreObservation", "ObservationList"]

log = logging.getLogger(__name__)


class ObservationCTA(object):
    """Container class for an CTA observation

    Parameters follow loosely the "Required columns" here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/obs_index/index.html#required-columns

    Parameters
    ----------
    obs_id : int
        Observation ID
    gti : `~gammapy.data.GTI`
        Good Time Intervals
    events : `~gammapy.data.EventList`
        Event list
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion2D`
        Energy dispersion
    psf : `~gammapy.irf.PSF3D` or `~gammapy.irf.EnergyDependentMultiGaussPSF` or `~gammapy.irf.PSFKing`
        Tabled Point Spread Function
    bkg : `~gammapy.irf.Background2D` or `~gammapy.irf.Background3D`
        Background rate
    pointing_radec : `~astropy.coordinates.SkyCoord`
        Pointing RA / DEC sky coordinates
    observation_live_time_duration : `~astropy.units.Quantity`
        Live-time duration in seconds

        The dead-time-corrected observation time.

        Computed as ``t_live = t_observation * (1 - f_dead)``
        where ``f_dead`` is the dead-time fraction.
    observation_dead_time_fraction : float
        Dead-time fraction

        Defined as dead-time over observation time.

        Dead-time is defined as the time during the observation
        where the detector did not record events:
        https://en.wikipedia.org/wiki/Dead_time
        https://adsabs.harvard.edu/abs/2004APh....22..285F

        The dead-time fraction is used in the live-time computation,
        which in turn is used in the exposure and flux computation.
    meta : `~collections.OrderedDict`
        Dictionary to store metadata

    Examples
    --------
    A minimal working example of how to create an observation from CTA's 1DC is given in
    examples/example_observation_cta.py

    """

    def __init__(
        self,
        obs_id=None,
        gti=None,
        events=None,
        aeff=None,
        edisp=None,
        psf=None,
        bkg=None,
        pointing_radec=None,
        observation_live_time_duration=None,
        observation_dead_time_fraction=None,
        meta=None,
    ):
        self.obs_id = obs_id
        self.gti = gti
        self.events = events
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.bkg = bkg
        self.pointing_radec = pointing_radec
        self.observation_live_time_duration = observation_live_time_duration
        self.observation_dead_time_fraction = observation_dead_time_fraction
        self.meta = meta or OrderedDict()

    def __str__(self):
        ss = "Info for OBS_ID = {}\n".format(self.obs_id)

        ss += "- Pointing pos: RA {:.2f} / Dec {:.2f}\n".format(
            self.pointing_radec.ra if self.pointing_radec else "None",
            self.pointing_radec.dec if self.pointing_radec else "None",
        )

        tstart = np.atleast_1d(self.gti.time_start.fits)[0] if self.gti else "None"
        ss += "- Start time: {}\n".format(tstart)
        ss += "- Observation duration: {}\n".format(
            self.gti.time_sum if self.gti else "None"
        )
        ss += "- Dead-time fraction: {:5.3f} %\n".format(
            100 * self.observation_dead_time_fraction
        )

        return ss


class DataStoreObservation(object):
    """IACT data store observation.

    Parameters
    ----------
    obs_id : int
        Observation ID
    data_store : `~gammapy.data.DataStore`
        Data store
    """

    def __init__(self, obs_id, data_store):
        # Assert that `obs_id` is available
        if obs_id not in data_store.obs_table["OBS_ID"]:
            raise ValueError("OBS_ID = {} not in obs index table.".format(obs_id))
        if obs_id not in data_store.hdu_table["OBS_ID"]:
            raise ValueError("OBS_ID = {} not in HDU index table.".format(obs_id))

        self.obs_id = obs_id
        self.data_store = data_store

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
        """Load `gammapy.data.EventList` object."""
        return self.load(hdu_type="events")

    @property
    def gti(self):
        """Load `gammapy.data.GTI` object."""
        return self.load(hdu_type="gti")

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

    @lazyproperty
    def obs_info(self):
        """Observation information (`~collections.OrderedDict`)."""
        row = self.data_store.obs_table.select_obs_id(obs_id=self.obs_id)[0]
        return table_row_to_dict(row)

    @lazyproperty
    def tstart(self):
        """Observation start time (`~astropy.time.Time`)."""
        met_ref = time_ref_from_dict(self.data_store.obs_table.meta)
        met = Quantity(self.obs_info["TSTART"].astype("float64"), "second")
        time = met_ref + met
        return time

    @lazyproperty
    def tstop(self):
        """Observation stop time (`~astropy.time.Time`)."""
        met_ref = time_ref_from_dict(self.data_store.obs_table.meta)
        met = Quantity(self.obs_info["TSTOP"].astype("float64"), "second")
        time = met_ref + met
        return time

    @lazyproperty
    def observation_time_duration(self):
        """Observation time duration in seconds (`~astropy.units.Quantity`).

        The wall time, including dead-time.
        """
        return Quantity(self.obs_info["ONTIME"], "second")

    @lazyproperty
    def observation_live_time_duration(self):
        """Live-time duration in seconds (`~astropy.units.Quantity`).

        The dead-time-corrected observation time.

        Computed as ``t_live = t_observation * (1 - f_dead)``
        where ``f_dead`` is the dead-time fraction.
        """
        return Quantity(self.obs_info["LIVETIME"], "second")

    @lazyproperty
    def observation_dead_time_fraction(self):
        """Dead-time fraction (float).

        Defined as dead-time over observation time.

        Dead-time is defined as the time during the observation
        where the detector didn't record events:
        https://en.wikipedia.org/wiki/Dead_time
        https://adsabs.harvard.edu/abs/2004APh....22..285F

        The dead-time fraction is used in the live-time computation,
        which in turn is used in the exposure and flux computation.
        """
        return 1 - self.obs_info["DEADC"]

    @lazyproperty
    def pointing_radec(self):
        """Pointing RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        lon, lat = self.obs_info["RA_PNT"], self.obs_info["DEC_PNT"]
        return SkyCoord(lon, lat, unit="deg", frame="icrs")

    @lazyproperty
    def pointing_altaz(self):
        """Pointing ALT / AZ sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        alt, az = self.obs_info["ALT_PNT"], self.obs_info["AZ_PNT"]
        return SkyCoord(az, alt, unit="deg", frame="altaz")

    @lazyproperty
    def pointing_zen(self):
        """Pointing zenith angle sky (`~astropy.units.Quantity`)."""
        return Quantity(self.obs_info["ZEN_PNT"], unit="deg")

    @lazyproperty
    def target_radec(self):
        """Target RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        lon, lat = self.obs_info["RA_OBJ"], self.obs_info["DEC_OBJ"]
        return SkyCoord(lon, lat, unit="deg", frame="icrs")

    @lazyproperty
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)."""
        return earth_location_from_dict(self.obs_info)

    @lazyproperty
    def muoneff(self):
        """Observation muon efficiency."""
        return self.obs_info["MUONEFF"]

    def peek(self):
        """Quick-look plots in a few panels."""
        raise NotImplementedError

    def make_psf(self, position, energy=None, rad=None):
        """Make energy-dependent PSF for a given source position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position at which to compute the PSF
        energy : `~astropy.units.Quantity`
            1-dim energy array for the output PSF.
            If none is given, the energy array of the PSF from the observation is used.
        rad : `~astropy.coordinates.Angle`
            1-dim offset wrt source position array for the output PSF.
            If none is given, the offset array of the PSF from the observation is used.

        Returns
        -------
        psf : `~gammapy.irf.EnergyDependentTablePSF`
            Energy dependent psf table
        """
        offset = position.separation(self.pointing_radec)

        if energy is None:
            energy = self.psf.to_energy_dependent_table_psf(theta=offset).energy

        if rad is None:
            rad = self.psf.to_energy_dependent_table_psf(theta=offset).rad

        if isinstance(self.psf, PSF3D):
            # PSF3D is a table PSF, so we use the native RAD binning by default
            # TODO: should handle this via a uniform caller API
            psf_value = self.psf.to_energy_dependent_table_psf(theta=offset).evaluate(
                energy
            )
        else:
            psf_value = self.psf.to_energy_dependent_table_psf(
                theta=offset, rad=rad
            ).evaluate(energy)

        arf = self.aeff.data.evaluate(offset=offset, energy=energy)
        exposure = arf * self.observation_live_time_duration

        psf = EnergyDependentTablePSF(
            energy=energy, rad=rad, exposure=exposure, psf_value=psf_value
        )
        return psf

    def to_observation_cta(self):
        """Convert to `~gammapy.data.ObservationCTA`.

        This loads all observation-related info from disk
        and stores it in the in-memory ``ObservationCTA``.

        Returns
        -------
        obs : `~gammapy.data.ObservationCTA`
            Observation
        """
        # maps the ObservationCTA class attributes to the DataStoreObservation properties
        props = {
            "obs_id": "obs_id",
            "gti": "gti",
            "events": "events",
            "aeff": "aeff",
            "edisp": "edisp",
            "psf": "psf",
            "bkg": "bkg",
            "pointing_radec": "pointing_radec",
            "observation_live_time_duration": "observation_live_time_duration",
            "observation_dead_time_fraction": "observation_dead_time_fraction",
        }

        for obs_cta_kwarg, ds_obs_prop in props.items():
            try:
                props[obs_cta_kwarg] = getattr(self, ds_obs_prop)
            except Exception as exception:
                log.warning(exception)
                props[obs_cta_kwarg] = None

        return ObservationCTA(**props)

    def check(self, checks="all"):
        """Run checks.

        This is a generator that yields a list of dicts.
        """
        checker = ObservationChecker(self)
        return checker.run(checks=checks)


class ObservationList(UserList):
    """List of `~gammapy.data.DataStoreObservation`.

    Could be extended to hold a more generic class of observations.
    """

    def __str__(self):
        s = self.__class__.__name__ + "\n"
        s += "Number of observations: {}\n".format(len(self))
        for obs in self:
            s += str(obs)
        return s

    def make_mean_psf(self, position, energy=None, rad=None):
        """Compute mean energy-dependent PSF.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position at which to compute the PSF
        energy : `~astropy.units.Quantity`
            1-dim energy array for the output PSF.
            If none is given, the energy array of the PSF from the first
            observation is used.
        rad : `~astropy.coordinates.Angle`
            1-dim offset wrt source position array for the output PSF.
            If none is given, the energy array of the PSF from the first
            observation is used.

        Returns
        -------
        psf : `~gammapy.irf.EnergyDependentTablePSF`
            Mean PSF
        """
        psf = self[0].make_psf(position, energy, rad)

        if rad is None:
            rad = psf.rad
        if energy is None:
            energy = psf.energy

        exposure = psf.exposure
        psf_value = psf.psf_value.T * psf.exposure

        for obs in self[1:]:
            psf = obs.make_psf(position, energy, rad)
            exposure += psf.exposure
            psf_value += psf.psf_value.T * psf.exposure

        psf_value /= exposure
        psf_tot = EnergyDependentTablePSF(
            energy=energy, rad=rad, exposure=exposure, psf_value=psf_value.T
        )
        return psf_tot

    def make_mean_edisp(
        self,
        position,
        e_true,
        e_reco,
        low_reco_threshold=Energy(0.002, "TeV"),
        high_reco_threshold=Energy(150, "TeV"),
    ):
        """Compute mean energy dispersion.

        Compute the mean edisp of a set of observations j at a given position

        The stacking is implemented in :func:`~gammapy.irf.IRFStacker.stack_edisp`

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position at which to compute the mean EDISP
        e_true : `~gammapy.utils.energy.EnergyBounds`
            True energy axis
        e_reco : `~gammapy.utils.energy.EnergyBounds`
            Reconstructed energy axis
        low_reco_threshold : `~gammapy.utils.energy.Energy`
            low energy threshold in reco energy, default 0.002 TeV
        high_reco_threshold : `~gammapy.utils.energy.Energy`
            high energy threshold in reco energy , default 150 TeV

        Returns
        -------
        stacked_edisp : `~gammapy.irf.EnergyDispersion`
            Stacked EDISP for a set of observation
        """
        list_aeff = []
        list_edisp = []
        list_livetime = []
        list_low_threshold = [low_reco_threshold] * len(self)
        list_high_threshold = [high_reco_threshold] * len(self)

        for obs in self:
            offset = position.separation(obs.pointing_radec)
            list_aeff.append(obs.aeff.to_effective_area_table(offset, energy=e_true))
            list_edisp.append(
                obs.edisp.to_energy_dispersion(offset, e_reco=e_reco, e_true=e_true)
            )
            list_livetime.append(obs.observation_live_time_duration)

        irf_stack = IRFStacker(
            list_aeff=list_aeff,
            list_edisp=list_edisp,
            list_livetime=list_livetime,
            list_low_threshold=list_low_threshold,
            list_high_threshold=list_high_threshold,
        )
        irf_stack.stack_edisp()

        return irf_stack.stacked_edisp


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

    def __init__(self, obs):
        self.obs = obs

    def _record(self, level="info", msg=None):
        return {"level": level, "obs_id": self.obs.obs_id, "msg": msg}

    def check_events(self):
        yield self._record(level="debug", msg="Starting events check")

        try:
            events = self.obs.load("events")
        except Exception:
            yield self._record(level="warning", msg="Loading events failed")
            return

        for record in EventListChecker(events).run():
            yield record

    # TODO: split this out into a GTIChecker
    def check_gti(self):
        yield self._record(level="debug", msg="Starting gti check")

        try:
            gti = self.obs.load("gti")
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
            aeff = self.obs.load("aeff")
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
            edisp = self.obs.load("edisp")
        except Exception:
            yield self._record(level="warning", msg="Loading edisp failed")
            return

        # Check that data isn't all null
        if np.max(edisp.data.data) <= 0:
            yield self._record(level="error", msg="maximum entry of edisp is <= 0")

    def check_psf(self):
        yield self._record(level="debug", msg="Starting psf check")

        try:
            psf = self.obs.load("psf")
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
