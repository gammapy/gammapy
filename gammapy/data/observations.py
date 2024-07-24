# Licensed under a 3-clause BSD style license - see LICENSE.rst
import collections.abc
import copy
import html
import inspect
import itertools
import logging
import warnings
from itertools import zip_longest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.units import Quantity
from astropy.utils import lazyproperty
import matplotlib.pyplot as plt
from gammapy.utils.deprecation import GammapyDeprecationWarning, deprecated
from gammapy.utils.fits import LazyFitsData, earth_location_to_dict
from gammapy.utils.metadata import CreatorMetaData, TargetMetaData, TimeInfoMetaData
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import Checker
from gammapy.utils.time import time_ref_to_dict, time_relative_to_ref
from .event_list import EventList, EventListChecker
from .filters import ObservationFilter
from .gti import GTI
from .metadata import ObservationMetaData
from .pointing import FixedPointingInfo

__all__ = ["Observation", "Observations"]

log = logging.getLogger(__name__)


class Observation:
    """In-memory observation.

    Parameters
    ----------
    obs_id : int, optional
        Observation id. Default is None
    obs_info : dict, optional
        Observation info dict. Default is None.
    aeff : `~gammapy.irf.EffectiveAreaTable2D`, optional
        Effective area. Default is None.
    edisp : `~gammapy.irf.EnergyDispersion2D`, optional
        Energy dispersion. Default is None.
    psf : `~gammapy.irf.PSF3D`, optional
        Point spread function. Default is None.
    bkg : `~gammapy.irf.Background3D`, optional
        Background rate model. Default is None.
    rad_max : `~gammapy.irf.RadMax2D`, optional
        Only for point-like IRFs: RAD_MAX table (energy dependent RAD_MAX)
        For a fixed RAD_MAX, create a RadMax2D with a single bin. Default is None.
    gti : `~gammapy.data.GTI`, optional
        Table with GTI start and stop time. Default is None.
    events : `~gammapy.data.EventList`, optional
        Event list. Default is None.
    obs_filter : `ObservationFilter`, optional
        Observation filter. Default is None.
    pointing : `~gammapy.data.FixedPointingInfo`, optional
        Pointing information. Default is None.
    location : `~astropy.coordinates.EarthLocation`, optional
        Earth location of the observatory. Default is None.
    """

    aeff = LazyFitsData(cache=False)
    edisp = LazyFitsData(cache=False)
    psf = LazyFitsData(cache=False)
    bkg = LazyFitsData(cache=False)
    _rad_max = LazyFitsData(cache=True)
    _events = LazyFitsData(cache=False)
    _gti = LazyFitsData(cache=True)
    _pointing = LazyFitsData(cache=True)

    def __init__(
        self,
        obs_id=None,
        meta=None,
        obs_info=None,
        gti=None,
        aeff=None,
        edisp=None,
        psf=None,
        bkg=None,
        rad_max=None,
        events=None,
        obs_filter=None,
        pointing=None,
        location=None,
    ):
        if obs_info is not None:
            warnings.warn(
                "obs_info argument is deprecated since v1.2. Use meta instead.",
                GammapyDeprecationWarning,
            )
            if meta is None:
                meta = ObservationMetaData.from_header(obs_info)

        self.obs_id = obs_id
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.bkg = bkg
        self._rad_max = rad_max
        self._gti = gti
        self._events = events
        self._pointing = pointing
        self._location = location  # this is part of the meta or is it data?
        self.obs_filter = obs_filter or ObservationFilter()
        self._meta = meta

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    @property
    def meta(self):
        """Return metadata container."""
        if self._meta is None and self.events:
            self._meta = ObservationMetaData.from_header(self.events.table.meta)
        return self._meta

    @property
    def rad_max(self):
        """Rad max IRF. None if not available."""
        # prevent circular import
        from gammapy.irf import RadMax2D

        if self._rad_max is not None:
            return self._rad_max

        # load once to avoid trigger lazy loading it three times
        aeff = self.aeff
        if aeff is not None and aeff.is_pointlike:
            self._rad_max = RadMax2D.from_irf(aeff)
            return self._rad_max

        edisp = self.edisp
        if edisp is not None and edisp.is_pointlike:
            self._rad_max = RadMax2D.from_irf(self.edisp)

        return self._rad_max

    @property
    def available_hdus(self):
        """Which HDUs are available."""
        available_hdus = []
        keys = ["_events", "_gti", "aeff", "edisp", "psf", "bkg", "_rad_max"]
        hdus = ["events", "gti", "aeff", "edisp", "psf", "bkg", "rad_max"]
        for key, hdu in zip(keys, hdus):
            available = self.__dict__.get(key, False)
            available_hdu = self.__dict__.get(f"_{hdu}_hdu", False)
            available_hdu_ = self.__dict__.get(f"_{key}_hdu", False)
            if available or available_hdu or available_hdu_:
                available_hdus.append(hdu)
        return available_hdus

    @property
    def available_irfs(self):
        """Which IRFs are available."""
        return [_ for _ in self.available_hdus if _ not in ["events", "gti"]]

    @property
    def events(self):
        """Event list of the observation as an `~gammapy.data.EventList`."""
        events = self.obs_filter.filter_events(self._events)
        return events

    @events.setter
    def events(self, value):
        if not isinstance(value, EventList):
            raise TypeError(f"events must be an EventList instance, got: {type(value)}")
        self._events = value

    @property
    def gti(self):
        """GTI of the observation as a `~gammapy.data.GTI`."""
        gti = self.obs_filter.filter_gti(self._gti)
        return gti

    @staticmethod
    def _get_obs_info(
        pointing, deadtime_fraction, time_start, time_stop, reference_time, location
    ):
        """Create observation information dictionary from in memory data."""
        obs_info = {
            "DEADC": 1 - deadtime_fraction,
        }
        if isinstance(pointing, SkyCoord):
            obs_info["RA_PNT"] = pointing.icrs.ra.deg
            obs_info["DEC_PNT"] = pointing.icrs.dec.deg

        obs_info.update(time_ref_to_dict(reference_time))
        obs_info["TSTART"] = time_relative_to_ref(time_start, obs_info).to_value(u.s)
        obs_info["TSTOP"] = time_relative_to_ref(time_stop, obs_info).to_value(u.s)

        if location is not None:
            obs_info.update(earth_location_to_dict(location))

        return obs_info

    @classmethod
    def create(
        cls,
        pointing,
        location=None,
        obs_id=0,
        livetime=None,
        tstart=None,
        tstop=None,
        irfs=None,
        deadtime_fraction=0.0,
        reference_time=Time("2000-01-01 00:00:00"),
    ):
        """Create an observation.

        User must either provide the livetime, or the start and stop times.

        Parameters
        ----------
        pointing : `~gammapy.data.FixedPointingInfo` or `~astropy.coordinates.SkyCoord`
            Pointing information.
        location : `~astropy.coordinates.EarthLocation`, optional
            Earth location of the observatory. Default is None.
        obs_id : int, optional
            Observation ID as identifier. Default is 0.
        livetime : ~astropy.units.Quantity`, optional
            Livetime exposure of the simulated observation. Default is None.
        tstart : `~astropy.time.Time` or `~astropy.units.Quantity`, optional
            Start time of observation as `~astropy.time.Time` or duration
            relative to `reference_time`. Default is None.
        tstop : `astropy.time.Time` or `~astropy.units.Quantity`, optional
            Stop time of observation as `~astropy.time.Time` or duration
            relative to `reference_time`. Default is None.
        irfs : dict, optional
            IRFs used for simulating the observation: `bkg`, `aeff`, `psf`, `edisp`, `rad_max`. Default is None.
        deadtime_fraction : float, optional
            Deadtime fraction. Default is 0.
        reference_time : `~astropy.time.Time`, optional
            the reference time to use in GTI definition. Default is `~astropy.time.Time("2000-01-01 00:00:00")`.

        Returns
        -------
        obs : `gammapy.data.MemoryObservation`
            Observation.
        """
        if tstart is None:
            tstart = reference_time.copy()

        if tstop is None:
            tstop = tstart + Quantity(livetime)

        gti = GTI.create(tstart, tstop, reference_time=reference_time)
        obs_info = cls._get_obs_info(
            pointing=pointing,
            deadtime_fraction=deadtime_fraction,
            time_start=gti.time_start[0],
            time_stop=gti.time_stop[0],
            reference_time=reference_time,
            location=location,
        )

        time_info = TimeInfoMetaData(
            time_start=gti.time_start[0],
            time_stop=gti.time_stop[-1],
            reference_time=reference_time,
        )

        meta = ObservationMetaData(
            deadtime_fraction=deadtime_fraction,
            location=location,
            time_info=time_info,
            creation=CreatorMetaData(),
            target=TargetMetaData(),
        )

        if not isinstance(pointing, FixedPointingInfo):
            warnings.warn(
                "Pointing will be required to be provided as FixedPointingInfo",
                GammapyDeprecationWarning,
            )
            pointing = FixedPointingInfo.from_fits_header(obs_info)

        return cls(
            obs_id=obs_id,
            meta=meta,
            gti=gti,
            aeff=irfs.get("aeff"),
            bkg=irfs.get("bkg"),
            edisp=irfs.get("edisp"),
            psf=irfs.get("psf"),
            rad_max=irfs.get("rad_max"),
            pointing=pointing,
            location=location,
        )

    @property
    def tstart(self):
        """Observation start time as a `~astropy.time.Time` object."""
        return self.gti.time_start[0]

    @property
    def tstop(self):
        """Observation stop time as a `~astropy.time.Time` object."""
        return self.gti.time_stop[0]

    @property
    def tmid(self):
        """Midpoint between start and stop time as a `~astropy.time.Time` object."""
        return self.tstart + 0.5 * (self.tstop - self.tstart)

    @property
    def observation_time_duration(self):
        """Observation time duration in seconds as a `~astropy.units.Quantity`.

        The wall time, including dead-time.
        """
        return self.gti.time_sum

    @property
    def observation_live_time_duration(self):
        """Live-time duration in seconds as a `~astropy.units.Quantity`.

        The dead-time-corrected observation time.

        Computed as ``t_live = t_observation * (1 - f_dead)``
        where ``f_dead`` is the dead-time fraction.
        """
        return (
            self.observation_time_duration
            * (1 - self.observation_dead_time_fraction)
            * self.obs_filter.livetime_fraction
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
        return self.meta.deadtime_fraction

    @property
    def obs_info(self):
        """Observation information dictionary."""
        warnings.warn(
            "obs_info property is deprecated since v1.2. Use meta instead.",
            GammapyDeprecationWarning,
        )
        return self.meta.to_header()

    @property
    def pointing(self):
        """Get the pointing for the observation as a `~gammapy.data.FixedPointingInfo` object."""
        if self._pointing is None:
            self._pointing = FixedPointingInfo.from_fits_header(self.events.table.meta)
        return self._pointing

    def get_pointing_altaz(self, time):
        """Get the pointing in alt-az for given time."""
        return self.pointing.get_altaz(time, self.observatory_earth_location)

    def get_pointing_icrs(self, time):
        """Get the pointing in ICRS for given time."""
        return self.pointing.get_icrs(time, self.observatory_earth_location)

    @property
    def observatory_earth_location(self):
        """Observatory location as an `~astropy.coordinates.EarthLocation` object."""
        if self._location is None:
            return self.meta.location
        return self._location

    @lazyproperty
    def target_radec(self):
        """Target RA / DEC sky coordinates as a `~astropy.coordinates.SkyCoord` object."""
        return self.meta.target.position

    @property
    @deprecated(
        "v1.2",
        message="Access additional metadata directly in obs.meta.opt.",
    )
    def muoneff(self):
        """Observation muon efficiency."""
        if "MUONEFF" in self.meta.optional:
            return self.meta.optional["MUONEFF"]
        else:
            raise KeyError("No muon efficiency information.")

    def __str__(self):
        pointing = self.get_pointing_icrs(self.tmid)
        ra = pointing.ra.deg
        dec = pointing.dec.deg

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

        This is a generator that yields a list of dictionary.
        """
        checker = ObservationChecker(self)
        return checker.run(checks=checks)

    def peek(self, figsize=(15, 10)):
        """Quick-look plots in a few panels.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (15, 10).
        """
        plottable_hds = ["events", "aeff", "psf", "edisp", "bkg", "rad_max"]

        plot_hdus = list(set(plottable_hds) & set(self.available_hdus))
        plot_hdus.sort()

        n_irfs = len(plot_hdus)
        nrows = n_irfs // 2
        ncols = 2 + n_irfs % 2

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            gridspec_kw={"wspace": 0.3, "hspace": 0.3},
        )

        for idx, (ax, name) in enumerate(zip_longest(axes.flat, plot_hdus)):
            if name == "aeff":
                self.aeff.plot(ax=ax)
                ax.set_title("Effective area")

            if name == "bkg":
                bkg = self.bkg
                if not bkg.has_offset_axis:
                    bkg = bkg.to_2d()
                bkg.plot(ax=ax)
                ax.set_title("Background rate")

            if name == "psf":
                self.psf.plot_containment_radius_vs_energy(ax=ax)
                ax.set_title("Point spread function")

            if name == "edisp":
                self.edisp.plot_bias(ax=ax, add_cbar=True)
                ax.set_title("Energy dispersion")

            if name == "rad_max":
                self.rad_max.plot_rad_max_vs_energy(ax=ax)
                ax.set_title("Rad max")

            if name == "events":
                m = self.events._counts_image(allsky=False)
                ax.remove()
                ax = fig.add_subplot(nrows, ncols, idx + 1, projection=m.geom.wcs)
                m.plot(ax=ax, stretch="sqrt", vmin=0, add_cbar=True)
                ax.set_title("Events")

            if name is None:
                ax.set_visible(False)

    def select_time(self, time_interval):
        """Select a time interval of the observation.

        Parameters
        ----------
        time_interval : `astropy.time.Time`
            Start and stop time of the selected time interval.
            For now, we only support a single time interval.

        Returns
        -------
        new_obs : `~gammapy.data.Observation`
            A new observation instance of the specified time interval.
        """
        new_obs_filter = self.obs_filter.copy()
        new_obs_filter.time_filter = time_interval
        obs = copy.deepcopy(self)
        obs.obs_filter = new_obs_filter
        return obs

    @classmethod
    def read(cls, event_file, irf_file=None, checksum=False):
        """Create an Observation from a Event List and an (optional) IRF file.

        Parameters
        ----------
        event_file : str or `~pathlib.Path`
            Path to the FITS file containing the event list and the GTI.
        irf_file : str or `~pathlib.Path`, optional
            Path to the FITS file containing the IRF components. Default is None.
            If None, the IRFs will be read from the event file.
        checksum : bool
            If True checks both DATASUM and CHECKSUM cards in the file headers. Default is False.

        Returns
        -------
        observation : `~gammapy.data.Observation`
            Observation with the events and the IRF read from the file.
        """
        from gammapy.irf.io import load_irf_dict_from_file

        events = EventList.read(event_file, checksum=checksum)

        gti = GTI.read(event_file, checksum=checksum)

        irf_file = irf_file if irf_file is not None else event_file
        irf_dict = load_irf_dict_from_file(irf_file)

        obs_info = events.table.meta

        meta = ObservationMetaData.from_header(obs_info)
        return cls(
            events=events,
            gti=gti,
            obs_id=meta.obs_info.obs_id,
            pointing=FixedPointingInfo.from_fits_header(obs_info),
            meta=meta,
            location=meta.location,
            **irf_dict,
        )

    def write(
        self, path, overwrite=False, format="gadf", include_irfs=True, checksum=False
    ):
        """
        Write this observation into `~pathlib.Path` using the specified format.

        Parameters
        ----------
        path : str or `~pathlib.Path`
            Path for the output file.
        overwrite : bool, optional
            Overwrite existing file. Default is False.
        format : {"gadf"}
            Output format, currently only "gadf" is supported. Default is "gadf".
        include_irfs : bool, optional
            Whether to include irf components in the output file. Default is True.
        checksum : bool, optional
            When True adds both DATASUM and CHECKSUM cards to the headers written to the file.
            Default is False.
        """
        if format != "gadf":
            raise ValueError(f'Only the "gadf" format is supported, got {format}')

        path = make_path(path)

        primary = fits.PrimaryHDU()

        primary.header.update(self.meta.creation.to_header(format))

        hdul = fits.HDUList([primary])

        events = self.events
        if events is not None:
            events_hdu = events.to_table_hdu(format=format)
            events_hdu.header.update(
                self.pointing.to_fits_header(time_ref=events.time_ref)
            )
            hdul.append(events_hdu)

        gti = self.gti
        if gti is not None:
            hdul.append(gti.to_table_hdu(format=format))

        if include_irfs:
            for irf_name in self.available_irfs:
                irf = getattr(self, irf_name)
                if irf is not None:
                    hdul.append(irf.to_table_hdu(format="gadf-dl3"))

        hdul.writeto(path, overwrite=overwrite, checksum=checksum)

    def copy(self, in_memory=False, **kwargs):
        """Copy observation.

        Overwriting `Observation` arguments requires the ``in_memory`` argument to be true.

        Parameters
        ----------
        in_memory : bool, optional
            Copy observation in memory. Default is False.
        **kwargs : dict, optional
            Keyword arguments passed to `Observation`.

        Examples
        --------
        >>> from gammapy.data import Observation
        >>> obs = Observation.read("$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz")
        >>> obs_copy = obs.copy(in_memory=True, obs_id=1234)
        >>> print(obs_copy) # doctest: +SKIP

        Returns
        -------
        obs : `Observation`
            Copied observation.
        """
        if in_memory:
            argnames = inspect.getfullargspec(self.__init__).args
            # TODO: remove once obs_info is removed from the list of arguments in __init__
            argnames.remove("obs_info")
            argnames.remove("self")

            for name in argnames:
                if name == "location":
                    attr = "observatory_earth_location"
                else:
                    attr = name
                value = getattr(self, attr)
                kwargs.setdefault(name, copy.deepcopy(value))
            return self.__class__(**kwargs)

        if kwargs:
            raise ValueError("Overwriting arguments requires to set 'in_memory=True'")

        return copy.deepcopy(self)


class Observations(collections.abc.MutableSequence):
    """Container class that holds a list of observations.

    Parameters
    ----------
    observations : list
        A list of `~gammapy.data.Observation`.
    """

    def __init__(self, observations=None):
        self._observations = []

        if observations is None:
            observations = []

        for obs in observations:
            self.append(obs)

    def __getitem__(self, item):
        if isinstance(item, (list, np.ndarray)) and all(
            isinstance(x, str) for x in item
        ):
            return self.__class__([self._observations[self.index(_)] for _ in item])
        elif isinstance(item, (slice, list, np.ndarray)):
            return self.__class__(list(np.array(self._observations)[item]))
        else:
            return self._observations[self.index(item)]

    def __delitem__(self, key):
        del self._observations[self.index(key)]

    def __setitem__(self, key, obs):
        if isinstance(obs, Observation):
            if obs in self:
                log.warning(
                    f"Observation with obs_id {obs.obs_id} already belongs to Observations."
                )
            self._observations[self.index(key)] = obs
        else:
            raise TypeError(f"Invalid type: {type(obs)!r}")

    def insert(self, idx, obs):
        if isinstance(obs, Observation):
            if obs in self:
                log.warning(
                    f"Observation with obs_id {obs.obs_id} already belongs to Observations."
                )
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
        """List of observation IDs (`list`)."""
        return [str(obs.obs_id) for obs in self]

    def select_time(self, time_intervals):
        """Select a time interval of the observations.

        Parameters
        ----------
        time_intervals : `astropy.time.Time` or list of `astropy.time.Time`
            List of start and stop time of the time intervals or one time interval.

        Returns
        -------
        new_observations : `~gammapy.data.Observations`
            A new Observations instance of the specified time intervals.
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

    def group_by_label(self, labels):
        """Split observations in multiple groups of observations.

        Parameters
        ----------
        labels : list or `numpy.ndarray`
            Array of group labels.

        Returns
        -------
        obs_clusters : dict of `~gammapy.data.Observations`
            Dictionary of Observations instance, one instance for each group.
        """
        obs_groups = {}
        for label in np.unique(labels):
            observations = self.__class__(
                [obs for k, obs in enumerate(self) if labels[k] == label]
            )
            obs_groups[f"group_{label}"] = observations
        return obs_groups

    @classmethod
    def from_stack(cls, observations_list):
        # TODO : Do more check when stacking observations when we have metadata.
        """Create a new `Observations` instance by concatenating a list of `Observations` objects.

        Parameters
        ----------
        observations_list : list of `~gammapy.data.Observations`
            The list of `Observations` to stack.

        Returns
        -------
        observations : `~gammapy.data.Observations`
            The `Observations` object resulting from stacking all the `Observations` in `observation_list`.
        """
        obs = itertools.chain(*observations_list)
        return cls(list(obs))

    def in_memory_generator(self):
        """A generator that iterates over observation. Yield an in memory copy of the observation."""
        for obs in self:
            obs_copy = obs.copy(in_memory=True)
            yield obs_copy


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
