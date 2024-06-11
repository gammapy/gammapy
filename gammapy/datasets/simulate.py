# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate observations."""
import html
import logging
from copy import deepcopy
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.table import Table
from astropy.time import Time
from gammapy import __version__
from gammapy.data import EventList, observatory_locations
from gammapy.maps import MapAxis, MapCoord, RegionNDMap, TimeMapAxis
from gammapy.modeling.models import (
    ConstantSpectralModel,
    ConstantTemporalModel,
    PointSpatialModel,
)
from gammapy.utils.fits import earth_location_to_dict
from gammapy.utils.random import get_random_state
from .map import create_map_dataset_from_observation

__all__ = ["MapDatasetEventSampler", "ObservationEventSampler"]

log = logging.getLogger(__name__)


class MapDatasetEventSampler:
    """Sample events from a map dataset.

    Parameters
    ----------
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}, optional
        Defines random number generator initialisation via the `~gammapy.utils.random.get_random_state` function.
    oversample_energy_factor : int, optional
        Defines an oversampling factor for the energies; it is used only when sampling
        an energy-dependent time-varying source.
    t_delta : `~astropy.units.Quantity`, optional
        Time interval used to sample the time-dependent source.
    keep_mc_id : bool, optional
        Flag to tag sampled events from a given model with a Montecarlo identifier.
        Default is True. If set to False, no identifier will be assigned.
    n_event_bunch : int
        Size of events bunches to sample. If None, sample all events in memory.
        Default is 10000.
    """

    def __init__(
        self,
        random_state="random-seed",
        oversample_energy_factor=10,
        t_delta=0.5 * u.s,
        keep_mc_id=True,
        n_event_bunch=10000,
    ):
        self.random_state = get_random_state(random_state)
        self.oversample_energy_factor = oversample_energy_factor
        self.t_delta = t_delta
        self.keep_mc_id = keep_mc_id
        self.n_event_bunch = n_event_bunch

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def _make_table(self, coords, time_ref):
        """Create a table for sampled events.

        Parameters
        ----------
        coords : `~gammapy.maps.MapCoord`
            Coordinates of the sampled events.
        time_ref : `~astropy.time.Time`
            Reference time of the event list.

        Returns
        -------
        table : `~astropy.table.Table`
            Table of the sampled events.
        """
        table = Table()
        try:
            energy = coords["energy_true"]
        except KeyError:
            energy = coords["energy"]

        table["TIME"] = (coords["time"] - time_ref).to("s")
        table["ENERGY_TRUE"] = energy

        table["RA_TRUE"] = coords.skycoord.icrs.ra.to("deg")
        table["DEC_TRUE"] = coords.skycoord.icrs.dec.to("deg")

        return table

    def _evaluate_timevar_source(
        self,
        dataset,
        model,
    ):
        """Calculate Npred for a given `dataset.model` by evaluating
        it on a region geometry.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset.
        model : `~gammapy.modeling.models.SkyModel`
            Sky model instance.

        Returns
        -------
        npred : `~gammapy.maps.RegionNDMap`
            Npred map.
        """
        energy_true = dataset.edisp.edisp_map.geom.axes["energy_true"]
        energy_new = energy_true.upsample(self.oversample_energy_factor)

        position = model.spatial_model.position
        region_exposure = dataset.exposure.to_region_nd_map(position)

        time_axis_eval = TimeMapAxis.from_gti_bounds(
            gti=dataset.gti, t_delta=self.t_delta
        )

        time_min, time_max = time_axis_eval.time_bounds
        time_axis = MapAxis.from_bounds(
            time_min.mjd * u.d,
            time_max.mjd * u.d,
            nbin=time_axis_eval.nbin,
            name="time",
        )

        temp_eval = model.temporal_model.evaluate(
            time_axis_eval.time_mid, energy=energy_new.center
        )

        norm = model.spectral_model(energy=energy_new.center)

        if temp_eval.unit.is_equivalent(norm.unit):
            flux_diff = temp_eval.to(norm.unit) * norm.value[:, None]
        else:
            flux_diff = temp_eval * norm[:, None]

        flux_inte = flux_diff * energy_new.bin_width[:, None]

        flux_pred = RegionNDMap.create(
            region=position,
            axes=[time_axis, energy_new],
            data=np.array(flux_inte),
            unit=flux_inte.unit,
        )

        mapcoord = flux_pred.geom.get_coord(sparse=True)
        mapcoord["energy_true"] = energy_true.center[:, None, None, None]

        flux_values = flux_pred.interp_by_coord(mapcoord) * flux_pred.unit
        data = flux_values * region_exposure.quantity[:, None, :, :]
        data /= time_axis.nbin / self.oversample_energy_factor

        npred = RegionNDMap.create(
            region=position,
            axes=[time_axis, energy_true],
            data=data.to_value(""),
        )

        return npred

    def _sample_coord_time_energy(self, dataset, model):
        """Sample model components of a source with time-dependent spectrum.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset.
        model : `~gammapy.modeling.models.SkyModel`
            Sky model instance.

        Returns
        -------
        table : `~astropy.table.Table`
            Table of sampled events.
        """
        if not isinstance(model.spatial_model, PointSpatialModel):
            raise TypeError(
                f"Event sampler expects PointSpatialModel for a time varying source. Got {model.spatial_model} instead."
            )

        if not isinstance(model.spectral_model, ConstantSpectralModel):
            raise TypeError(
                f"Event sampler expects ConstantSpectralModel for a time varying source. Got {model.spectral_model} instead."
            )

        npred = self._evaluate_timevar_source(dataset, model=model)
        data = npred.data[np.isfinite(npred.data)]
        data = np.clip(data, 0, None)

        try:
            n_events = self.random_state.poisson(np.sum(data))
        except ValueError:
            raise ValueError(
                f"The number of predicted events for the model {model.name} is too large. No event sampling will be performed for this model!"
            )

        coords = npred.sample_coord(n_events=n_events, random_state=self.random_state)

        coords["time"] = Time(coords["time"], format="mjd", scale="tt")

        table = self._make_table(coords, dataset.gti.time_ref)

        return table

    def _sample_coord_time(self, npred, temporal_model, gti):
        """Sample model components of a time-varying source.

        Parameters
        ----------
        npred : `~gammapy.dataset.MapDataset`
            Npred map.
        temporal_model : `~gammapy.modeling.models\
            Temporal model of the source.
        gti : `~gammapy.data.GTI`
             GTI of the dataset.

        Returns
        -------
        table : `~astropy.table.Table`
            Table of sampled events.
        """
        data = npred.data[np.isfinite(npred.data)]
        data = np.clip(data, 0, None)
        n_events = self.random_state.poisson(np.sum(data))

        coords = npred.sample_coord(n_events=n_events, random_state=self.random_state)

        time_start, time_stop, time_ref = (gti.time_start, gti.time_stop, gti.time_ref)
        coords["time"] = temporal_model.sample_time(
            n_events=n_events,
            t_min=time_start,
            t_max=time_stop,
            random_state=self.random_state,
            t_delta=self.t_delta,
        )

        table = self._make_table(coords, time_ref)

        return table

    def sample_sources(self, dataset):
        """Sample source model components.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset.

        Returns
        -------
        events : `~gammapy.data.EventList`
            Event list.
        """

        events_all = EventList(Table())
        for idx, evaluator in enumerate(dataset.evaluators.values()):
            log.info(f"Evaluating model: {evaluator.model.name}")
            if evaluator.needs_update:
                evaluator.update(
                    dataset.exposure,
                    dataset.psf,
                    dataset.edisp,
                    dataset._geom,
                    dataset.mask,
                )
            if not evaluator.contributes:
                continue

            if evaluator.model.temporal_model is None:
                temporal_model = ConstantTemporalModel()
            else:
                temporal_model = evaluator.model.temporal_model

            if temporal_model.is_energy_dependent:
                table = self._sample_coord_time_energy(dataset, evaluator.model)
            else:
                flux = evaluator.compute_flux()
                npred = evaluator.apply_exposure(flux)
                table = self._sample_coord_time(npred, temporal_model, dataset.gti)

            if self.keep_mc_id:
                if len(table) == 0:
                    mcid = table.Column(name="MC_ID", length=0, dtype=int)
                    table.add_column(mcid)

                table["MC_ID"] = idx + 1
                table.meta["MID{:05d}".format(idx + 1)] = idx + 1
                table.meta["MMN{:05d}".format(idx + 1)] = evaluator.model.name

            events_all.stack(EventList(table))

        return events_all

    def sample_background(self, dataset):
        """Sample background.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset.

        Returns
        -------
        events : `gammapy.data.EventList`
            Background events.
        """

        table = Table()
        if dataset.background:
            log.info("Evaluating background...")
            background = dataset.npred_background()

            temporal_model = ConstantTemporalModel()

            table = self._sample_coord_time(background, temporal_model, dataset.gti)

            table["ENERGY"] = table["ENERGY_TRUE"]
            table["RA"] = table["RA_TRUE"]
            table["DEC"] = table["DEC_TRUE"]

            if self.keep_mc_id:
                table["MC_ID"] = 0
                table.meta["MID{:05d}".format(0)] = 0
                table.meta["MMN{:05d}".format(0)] = dataset.background_model.name

        return EventList(table)

    def sample_edisp(self, edisp_map, events):
        """Sample energy dispersion map.

        Parameters
        ----------
        edisp_map : `~gammapy.irf.EDispMap`
            Energy dispersion map.
        events : `~gammapy.data.EventList`
            Event list with the true energies.

        Returns
        -------
        events : `~gammapy.data.EventList`
            Event list with reconstructed energy column.
        """
        coord = MapCoord(
            {
                "lon": events.table["RA_TRUE"].quantity,
                "lat": events.table["DEC_TRUE"].quantity,
                "energy_true": events.table["ENERGY_TRUE"].quantity,
            },
            frame="icrs",
        )

        coords_reco = edisp_map.sample_coord(
            coord, self.random_state, self.n_event_bunch
        )
        events.table["ENERGY"] = coords_reco["energy"]
        return events

    def sample_psf(self, psf_map, events):
        """Sample PSF map.

        Parameters
        ----------
        psf_map : `~gammapy.irf.PSFMap`
            PSF map.
        events : `~gammapy.data.EventList`
            Event list.

        Returns
        -------
        events : `~gammapy.data.EventList`
            Event list with reconstructed position columns.
        """
        coord = MapCoord(
            {
                "lon": events.table["RA_TRUE"].quantity,
                "lat": events.table["DEC_TRUE"].quantity,
                "energy_true": events.table["ENERGY_TRUE"].quantity,
            },
            frame="icrs",
        )

        coords_reco = psf_map.sample_coord(coord, self.random_state, self.n_event_bunch)

        events.table["RA"] = coords_reco["lon"] * u.deg
        events.table["DEC"] = coords_reco["lat"] * u.deg
        return events

    @staticmethod
    def event_det_coords(observation, events):
        """Add columns of detector coordinates (DETX-DETY) to the event list.

        Parameters
        ----------
        observation : `~gammapy.data.Observation`
            In memory observation.
        events : `~gammapy.data.EventList`
            Event list.

        Returns
        -------
        events : `~gammapy.data.EventList`
            Event list with columns of event detector coordinates.
        """
        sky_coord = SkyCoord(events.table["RA"], events.table["DEC"], frame="icrs")
        frame = SkyOffsetFrame(origin=observation.get_pointing_icrs(observation.tmid))
        pseudo_fov_coord = sky_coord.transform_to(frame)

        events.table["DETX"] = pseudo_fov_coord.lon
        events.table["DETY"] = pseudo_fov_coord.lat
        return events

    @staticmethod
    def event_list_meta(dataset, observation, keep_mc_id=True):
        """Event list meta info.
        Please, note that this function will be updated in the future.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset.
        observation : `~gammapy.data.Observation`
            In memory observation.
        keep_mc_id : bool
            Flag to tag sampled events from a given model with a Montecarlo identifier.
            Default is True. If set to False, no identifier will be assigned.

        Returns
        -------
        meta : dict
            Meta dictionary.
        """
        # See: https://gamma-astro-data-formats.readthedocs.io/en/latest/events/events.html#mandatory-header-keywords  # noqa: E501
        meta = {}

        meta["HDUCLAS1"] = "EVENTS"
        meta["EXTNAME"] = "EVENTS"
        meta["HDUDOC"] = (
            "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
        )
        meta["HDUVERS"] = "0.2"
        meta["HDUCLASS"] = "GADF"

        meta["OBS_ID"] = observation.obs_id

        meta["TSTART"] = (observation.tstart - dataset.gti.time_ref).to_value("s")
        meta["TSTOP"] = (observation.tstop - dataset.gti.time_ref).to_value("s")

        meta["ONTIME"] = observation.observation_time_duration.to("s").value
        meta["LIVETIME"] = observation.observation_live_time_duration.to("s").value
        meta["DEADC"] = 1 - observation.observation_dead_time_fraction

        fixed_icrs = observation.pointing.fixed_icrs
        meta["RA_PNT"] = fixed_icrs.ra.deg
        meta["DEC_PNT"] = fixed_icrs.dec.deg

        meta["EQUINOX"] = "J2000"
        meta["RADECSYS"] = "icrs"

        meta["CREATOR"] = "Gammapy {}".format(__version__)
        meta["EUNIT"] = "TeV"
        meta["EVTVER"] = ""

        meta["OBSERVER"] = "Gammapy user"
        meta["DSTYP1"] = "TIME"
        meta["DSUNI1"] = "s"
        meta["DSVAL1"] = "TABLE"
        meta["DSREF1"] = ":GTI"
        meta["DSTYP2"] = "ENERGY"
        meta["DSUNI2"] = "TeV"
        meta["DSVAL2"] = (
            f'{dataset._geom.axes["energy"].edges.min().value}:{dataset._geom.axes["energy"].edges.max().value}'  # noqa: E501
        )
        meta["DSTYP3"] = "POS(RA,DEC)     "

        offset_max = np.max(dataset._geom.width).to_value("deg")
        meta["DSVAL3"] = (
            f"CIRCLE({fixed_icrs.ra.deg},{fixed_icrs.dec.deg},{offset_max})"  # noqa: E501
        )
        meta["DSUNI3"] = "deg             "
        meta["NDSKEYS"] = " 3 "

        # get first non background model component
        for model in dataset.models:
            if model is not dataset.background_model:
                break
        else:
            model = None

        if model:
            meta["OBJECT"] = model.name
            meta["RA_OBJ"] = model.position.icrs.ra.deg
            meta["DEC_OBJ"] = model.position.icrs.dec.deg

        meta["TELAPSE"] = dataset.gti.time_sum.to("s").value
        meta["MJDREFI"] = int(dataset.gti.time_ref.mjd)
        meta["MJDREFF"] = dataset.gti.time_ref.mjd % 1
        meta["TIMEUNIT"] = "s"
        meta["TIMESYS"] = dataset.gti.time_ref.scale
        meta["TIMEREF"] = "LOCAL"
        meta["DATE-OBS"] = dataset.gti.time_start.isot[0][0:10]
        meta["DATE-END"] = dataset.gti.time_stop.isot[0][0:10]
        meta["CONV_DEP"] = 0
        meta["CONV_RA"] = 0
        meta["CONV_DEC"] = 0

        if keep_mc_id:
            meta["NMCIDS"] = len(dataset.models)

        # Necessary for DataStore, but they should be ALT and AZ instead!
        telescope = observation.aeff.meta["TELESCOP"]
        instrument = observation.aeff.meta["INSTRUME"]
        loc = observation.observatory_earth_location
        if loc is None:
            if telescope == "CTA":
                if instrument == "Southern Array":
                    loc = observatory_locations["cta_south"]
                elif instrument == "Northern Array":
                    loc = observatory_locations["cta_north"]
                else:
                    loc = observatory_locations["cta_south"]

            else:
                loc = observatory_locations[telescope.lower()]

        meta.update(earth_location_to_dict(loc))

        # this is not really correct but maybe OK for now
        coord_altaz = observation.pointing.get_altaz(dataset.gti.time_start, loc)

        meta["ALT_PNT"] = coord_altaz.alt.deg[0]
        meta["AZ_PNT"] = coord_altaz.az.deg[0]

        # TO DO: these keywords should be taken from the IRF of the dataset
        meta["ORIGIN"] = "Gammapy"
        meta["TELESCOP"] = observation.aeff.meta["TELESCOP"]
        meta["INSTRUME"] = observation.aeff.meta["INSTRUME"]
        meta["N_TELS"] = ""
        meta["TELLIST"] = ""

        meta["CREATED"] = Time.now().iso
        meta["OBS_MODE"] = "POINTING"
        meta["EV_CLASS"] = ""

        return meta

    def run(self, dataset, observation=None):
        """Run the event sampler, applying IRF corrections.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset.
        observation : `~gammapy.data.Observation`, optional
            In memory observation. Default is None.

        Returns
        -------
        events : `~gammapy.data.EventList`
            Event list.
        """
        events_src = self.sample_sources(dataset)

        if len(events_src.table) > 0:
            if dataset.psf:
                events_src = self.sample_psf(dataset.psf, events_src)
            else:
                events_src.table["RA"] = events_src.table["RA_TRUE"]
                events_src.table["DEC"] = events_src.table["DEC_TRUE"]

            if dataset.edisp:
                events_src = self.sample_edisp(dataset.edisp, events_src)
            else:
                events_src.table["ENERGY"] = events_src.table["ENERGY_TRUE"]

        events_bkg = self.sample_background(dataset)
        events = EventList.from_stack([events_bkg, events_src])

        events.table["EVENT_ID"] = np.arange(len(events.table))
        if observation is not None:
            events = self.event_det_coords(observation, events)
            events.table.meta.update(
                self.event_list_meta(dataset, observation, self.keep_mc_id)
            )

        geom = dataset._geom
        selection = geom.contains(events.map_coord(geom))
        log.info("Event sampling completed.")
        return events.select_row_subset(selection)


class ObservationEventSampler(MapDatasetEventSampler):
    """
    Sample event lists for a given observation and signal models.

    Signal events are sampled from the predicted counts distribution given by the product of the sky models and the
    expected exposure. They are then folded with the instrument response functions.
    To improve performance, IRFs are evaluated on a pre-defined binning,
    not at each individual event energy / coordinate.

    Parameters
    ----------
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}, optional
        Defines random number generator initialisation via the `~gammapy.utils.random.get_random_state` function.
    oversample_energy_factor : int, optional
        Defines an oversampling factor for the energies; it is used only when sampling
        an energy-dependent time-varying source. Default is 10.
    t_delta : `~astropy.units.Quantity`, optional
        Time interval used to sample the time-dependent source. Default is 0.5 s.
    keep_mc_id : bool, optional
        Flag to tag sampled events from a given model with a Montecarlo identifier.
        Default is True. If set to False, no identifier will be assigned.
    n_event_bunch : int
        Size of events bunches to sample. If None, sample all events in memory.
        Default is 10000.
    dataset_kwargs : dict, optional
        Arguments passed to `~gammapy.datasets.create_map_dataset_from_observation()`
    """

    def __init__(
        self,
        random_state="random-seed",
        oversample_energy_factor=10,
        t_delta=0.5 * u.s,
        keep_mc_id=True,
        n_event_bunch=10000,
        dataset_kwargs=None,
    ):
        self.dataset_kwargs = dataset_kwargs
        self.random_state = get_random_state(random_state)
        self.oversample_energy_factor = oversample_energy_factor
        self.t_delta = t_delta
        self.n_event_bunch = n_event_bunch
        self.keep_mc_id = keep_mc_id

    def run(self, observation, models=None, dataset_name=None):
        """Sample events for given observation and signal models.

        The signal distribution is sampled from the given models
        in true coordinates and energy. The true quantities are
        then folded with the IRFs to obtain the observable quantities.

        Parameters
        ----------
        observation : `~gammapy.data.Observation`
            Observation to be simulated.
        models : `~gammapy.modeling.Models`, optional
            Models to simulate.
            Can be None to only sample background events. Default is None.
        dataset_name : str, optional
            If `models` contains one or multiple `FoVBackgroundModel`
            it should match the `dataset_name` of the background model to use.
            Default is None.
        Returns
        -------
        observation : `~gammapy.data.Observation`
            A copy of the input observation with event list filled.
        """

        dataset = create_map_dataset_from_observation(
            observation, models, dataset_name, **self.dataset_kwargs
        )
        events = super().run(dataset, observation)

        observation = deepcopy(observation)
        observation._events = events
        return observation
