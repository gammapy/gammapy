# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate observations"""
import numpy as np
import astropy.units as u
from astropy.table import Table
from gammapy.cube import (
    MapDataset,
    PSFMap,
    make_map_background_irf,
    make_map_exposure_true_energy,
)
from gammapy.data import EventList
from gammapy.maps import WcsNDMap
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
        """Sample source model components.

        Parameters
        ----------
        npred : `~gammapy.maps.Map`
            Maps of the predicted counts.
        temporal_model : `~gammapy.modeling.models`
            Temporal model.
        gti : `MapDataset` object
            Good time intervals of the given MapDataset object.

        Returns
        -------
        events : `EventList`
            Event list
        """
        table = Table()
        n_events = self.random_state.poisson(np.sum(npred.data))

        # sample position
        coords = npred.sample_coord(n_events=n_events, random_state=self.random_state)
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
        dataset : `MapDataset`
            Map dataset.

        Returns
        -------
        events : `EventList`
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
        dataset : `MapDataset`
            Map dataset.

        Returns
        -------
        events : `EventList`
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
