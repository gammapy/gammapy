# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate observations"""
import numpy as np
import astropy.units as u
from astropy.table import Table
import gammapy
from gammapy.cube import (
    MapDataset,
    PSFMap,
    make_map_background_irf,
    make_map_exposure_true_energy,
)
from gammapy.data import EventList
from gammapy.maps import MapCoord, WcsNDMap
from gammapy.modeling.models import BackgroundModel, ConstantTemporalModel
from gammapy.utils.random import get_random_state

__all__ = ["simulate_dataset", "MapDatasetEventSampler"]


def simulate_dataset(
    skymodel,
    geom,
    pointing,
    irfs,
    livetime=1 * u.h,
    offset=1 * u.deg,
    max_radius=0.8 * u.deg,
    random_state="random-seed",
):
    """Simulate a 3D dataset.

    Simulate a source defined with a sky model for a given pointing,
    geometry and irfs for a given exposure time.
    This will return a dataset object which includes the counts cube,
    the exposure cube, the psf cube, the background model and the sky model.

    Parameters
    ----------
    skymodel : `~gammapy.modeling.models.SkyModel`
        Background model map
    geom : `~gammapy.maps.WcsGeom`
        Geometry object for the observation
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing position
    irfs : dict
        Irfs used for simulating the observation
    livetime : `~astropy.units.Quantity`
        Livetime exposure of the simulated observation
    offset : `~astropy.units.Quantity`
        Offset from the center of the pointing position.
        This is used for the PSF and Edisp estimation
    max_radius : `~astropy.coordinates.Angle`
        The maximum radius of the PSF kernel.
    random_state: {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.

    Returns
    -------
    dataset : `~gammapy.cube.MapDataset`
        A dataset of the simulated observation.
    """
    background = make_map_background_irf(
        pointing=pointing, ontime=livetime, bkg=irfs["bkg"], geom=geom
    )

    background_model = BackgroundModel(background)

    psf = irfs["psf"].to_energy_dependent_table_psf(theta=offset)
    psf_map = PSFMap.from_energy_dependent_table_psf(psf)

    exposure = make_map_exposure_true_energy(
        pointing=pointing, livetime=livetime, aeff=irfs["aeff"], geom=geom
    )

    if "edisp" in irfs:
        energy = geom.axes[0].edges
        edisp = irfs["edisp"].to_energy_dispersion(offset, e_reco=energy, e_true=energy)
    else:
        edisp = None

    dataset = MapDataset(
        models=skymodel,
        exposure=exposure,
        background_model=background_model,
        psf=psf_map,
        edisp=edisp,
    )

    npred_map = dataset.npred()
    rng = get_random_state(random_state)
    counts = rng.poisson(npred_map.data)
    dataset.counts = WcsNDMap(geom, counts)

    return dataset


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
        table["ENERGY_TRUE"] = coords["energy"]
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
        for idx, evaluator in enumerate(dataset._evaluators):
            evaluator.edisp = None
            evaluator.psf = None
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
                "energy": events.table["ENERGY_TRUE"].quantity,
            },
            coordsys="icrs",
        )

        coords_reco = edisp_map.sample_coord(coord, self.random_state)
        events.table["ENERGY"] = coords_reco["energy"] * u.TeV
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
                "energy": events.table["ENERGY_TRUE"].quantity,
            },
            coordsys="icrs",
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
        # TODO: currently additional meta data is missing

        meta["MJDREFI"] = int(dataset.gti.time_ref.mjd)
        meta["MJDREFF"] = dataset.gti.time_ref.mjd % 1
        meta["TIMEUNIT"] = "s"
        meta["TIMESYS"] = "TT"
        meta["TIMEREF"] = "LOCAL"
        #         meta["DATE-OBS"] = dataset.gti.time_start.isot[0][0:10]
        #         meta["DATE-END"] = dataset.gti.time_stop.isot[0][0:10]
        meta["TIME-OBS"] = dataset.gti.time_start.isot[0]
        meta["TIME-END"] = dataset.gti.time_stop.isot[0]

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
        meta["ORIGIN"] = ""
        meta["TELESCOP"] = ""
        meta["INSTRUME"] = ""
        #

        meta["CREATOR"] = "Gammapy {}".format(gammapy.__version__)

        meta["OBSERVER"] = ""
        meta["CREATED"] = ""
        meta["OBJECT"] = ""
        meta["RA_OBJ"] = ""
        meta["DEC_OBJ"] = ""
        meta["OBS_MODE"] = ""
        meta["EV_CLASS"] = ""
        meta["TELAPSE"] = ""

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
        events_bkg = self.sample_background(dataset)
        events_src = self.sample_sources(dataset)
        events_src = self.sample_psf(dataset.psf, events_src)

        events_src = self.sample_edisp(dataset.edisp, events_src)

        events = EventList.stack([events_bkg, events_src])
        events.table["EVENT_ID"] = np.arange(len(events.table))
        events.table.meta = self.event_list_meta(dataset, observation)

        return events
