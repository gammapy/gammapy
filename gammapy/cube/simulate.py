# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate observations"""
import copy
import numpy as np
import astropy.units as u
from astropy.table import Table
import gammapy
from gammapy.data import EventList
from gammapy.maps import MapCoord
from gammapy.modeling.models import BackgroundModel, ConstantTemporalModel
from gammapy.utils.random import get_random_state

__all__ = ["MapDatasetEventSampler"]


class MapDatasetEventSampler:
    """Sample events from a map dataset

    Parameters
    ----------
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.
    """

    def __init__(self, random_state="random-seed"):
        self.random_state = get_random_state(random_state)

    def _sample_coord_time(self, npred, temporal_model, gti):
        n_events = self.random_state.poisson(np.sum(npred.data))

        coords = npred.sample_coord(n_events=n_events, random_state=self.random_state)

        table = Table()
        try:
            energy = coords["energy_true"]
        except KeyError:
            energy = coords["energy"]

        table["ENERGY_TRUE"] = energy
        table["RA_TRUE"] = coords.skycoord.icrs.ra.to("deg")
        table["DEC_TRUE"] = coords.skycoord.icrs.dec.to("deg")

        time_start, time_stop, time_ref = (gti.time_start, gti.time_stop, gti.time_ref)
        time = temporal_model.sample_time(
            n_events, time_start, time_stop, self.random_state
        )
        table["TIME"] = u.Quantity(((time.mjd - time_ref.mjd) * u.day).to(u.s)).to("s")
        return table

    def sample_sources(self, dataset):
        """Sample source model components.

        Parameters
        ----------
        dataset : `~gammapy.cube.MapDataset`
            Map dataset.

        Returns
        -------
        events : `~gammapy.data.EventList`
            Event list
        """
        events_all = []

        for idx, model in enumerate(dataset.models):
            if isinstance(model, BackgroundModel):
                continue

            evaluator = dataset.evaluators.get(model.name)

            evaluator = copy.deepcopy(evaluator)
            evaluator.model.apply_irf["psf"] = False
            evaluator.model.apply_irf["edisp"] = False
            npred = evaluator.compute_npred()

            temporal_model = ConstantTemporalModel()

            table = self._sample_coord_time(npred, temporal_model, dataset.gti)
            table["MC_ID"] = idx + 1
            events_all.append(EventList(table))

        return EventList.stack(events_all)

    def sample_background(self, dataset):
        """Sample background

        Parameters
        ----------
        dataset : `~gammapy.cube.MapDataset`
            Map dataset

        Returns
        -------
        events : `gammapy.data.EventList`
            Background events
        """
        background = dataset.background_model.evaluate()

        temporal_model = ConstantTemporalModel()

        table = self._sample_coord_time(background, temporal_model, dataset.gti)

        table["MC_ID"] = 0
        table.rename_column("ENERGY_TRUE", "ENERGY")
        table.rename_column("RA_TRUE", "RA")
        table.rename_column("DEC_TRUE", "DEC")

        return EventList(table)

    def sample_edisp(self, edisp_map, events):
        """Sample energy dispersion map.

        Parameters
        ----------
        edisp_map : `~gammapy.cube.EDispMap`
            Energy dispersion map
        events : `~gammapy.data.EventList`
            Event list with the true energies

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

        coords_reco = edisp_map.sample_coord(coord, self.random_state)
        events.table["ENERGY"] = coords_reco["energy"]
        return events

    def sample_psf(self, psf_map, events):
        """Sample psf map.

        Parameters
        ----------
        psf_map : `~gammapy.cube.PSFMap`
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

        coords_reco = psf_map.sample_coord(coord, self.random_state)
        events.table["RA"] = coords_reco["lon"] * u.deg
        events.table["DEC"] = coords_reco["lat"] * u.deg
        return events

    @staticmethod
    def event_list_meta(dataset, observation):
        """Event list meta info.

        Parameters
        ----------
        dataset : `~gammapy.cube.MapDataset`
            Map dataset.
        observation : `~gammapy.data.Observation`
            In memory observation.

        Returns
        -------
        meta : dict
            Meta dictionary.
        """
        # See: https://gamma-astro-data-formats.readthedocs.io/en/latest/events/events.html#mandatory-header-keywords
        meta = {}

        meta["HDUCLAS1"] = "EVENTS"
        meta["EXTNAME"] = "EVENTS"
        meta[
            "HDUDOC"
        ] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
        meta["HDUVER"] = "0.2"
        meta["HDUCLASS"] = "GADF"

        meta["OBS_ID"] = observation.obs_id

        meta["TSTART"] = (
            ((observation.tstart.mjd - dataset.gti.time_ref.mjd) * u.day).to(u.s).value
        )
        meta["TSTOP"] = (
            ((observation.tstop.mjd - dataset.gti.time_ref.mjd) * u.day).to(u.s).value
        )
        meta["ONTIME"] = observation.observation_time_duration.to("s").value
        meta["LIVETIME"] = observation.observation_live_time_duration.to("s").value
        meta["DEADC"] = observation.observation_dead_time_fraction

        meta["RA_PNT"] = observation.pointing_radec.icrs.ra.deg
        meta["DEC_PNT"] = observation.pointing_radec.icrs.dec.deg

        meta["EQUINOX"] = "J2000"
        meta["RADECSYS"] = "icrs"
        # TO DO: these keywords should be taken from the IRF of the dataset
        meta["ORIGIN"] = "Gammapy"
        meta["TELESCOP"] = ""
        meta["INSTRUME"] = ""
        #
        meta["CREATOR"] = "Gammapy {}".format(gammapy.__version__)

        #        TO COMPLETE
        #        meta["OBSERVER"] = ""
        #        meta["CREATED"] = ""
        #        meta["OBJECT"] = ""
        #        meta["RA_OBJ"] = ""
        #        meta["DEC_OBJ"] = ""
        #        meta["OBS_MODE"] = ""
        #        meta["EV_CLASS"] = ""
        #        meta["TELAPSE"] = ""
        #
        #        meta["MJDREFI"] = int(dataset.gti.time_ref.mjd)
        #        meta["MJDREFF"] = dataset.gti.time_ref.mjd % 1
        #        meta["TIMEUNIT"] = "s"
        #        meta["TIMESYS"] = "TT"
        #        meta["TIMEREF"] = "LOCAL"
        #        #         meta["DATE-OBS"] = dataset.gti.time_start.isot[0][0:10]
        #        #         meta["DATE-END"] = dataset.gti.time_stop.isot[0][0:10]
        #        meta["TIME-OBS"] = dataset.gti.time_start.isot[0]
        #        meta["TIME-END"] = dataset.gti.time_stop.isot[0]
        #

        meta["GEOLON"] = 0
        meta["GEOLAT"] = 0
        meta["ALTITUDE"] = 0

        for idx, model in enumerate(dataset.models):
            meta["MID{:05d}".format(idx + 1)] = idx + 1
            meta["MMN{:05d}".format(idx + 1)] = model.name

        return meta

    def run(self, dataset, observation=None):
        """Run the event sampler, applying IRF corrections.

        Parameters
        ----------
        dataset : `~gammapy.cube.MapDataset`
            Map dataset
        observation : `~gammapy.data.Observation`
            In memory observation.
        edisp : Bool
            It allows to include or exclude the Edisp in the simulation.

        Returns
        -------
        events : `~gammapy.data.EventList`
            Event list.
        """
        try:
            dataset.models[1]
            events_src = self.sample_sources(dataset)

            if dataset.psf:
                events_src = self.sample_psf(dataset.psf, events_src)
            else:
                events_src.table["RA"] = events_src.table["RA_TRUE"]
                events_src.table["DEC"] = events_src.table["DEC_TRUE"]

            if dataset.edisp:
                events_src = self.sample_edisp(dataset.edisp, events_src)
            else:
                events_src.table["ENERGY"] = events_src.table["ENERGY_TRUE"]

            if dataset.background_model:
                events_bkg = self.sample_background(dataset)
                events = EventList.stack([events_bkg, events_src])
            else:
                events = events_src

        except:
            if dataset.background_model:
                events_bkg = self.sample_background(dataset)
                events = EventList.stack([events_bkg])

        events.table["EVENT_ID"] = np.arange(len(events.table))
        events.table.meta = self.event_list_meta(dataset, observation)

        return events
