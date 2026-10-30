# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""On-Off background estimation for dedicated On Off Pair runs"""

import numpy as np
import logging

from astropy.coordinates import SkyCoord
from astropy.table import QTable

from gammapy.data import EventList
from gammapy.datasets import (
    MapDataset,
    MapDatasetOnOff,
    SpectrumDataset,
    SpectrumDatasetOnOff,
)
from ..core import Maker
from ..utils import check_run_pair_validity
from gammapy.maps import Map
from gammapy.utils.coordinates import fov_to_sky, sky_to_fov

log = logging.getLogger(__name__)

__all__ = ["OnOffBackgroundMaker"]


class OnOffBackgroundMaker(Maker):
    """Estimate OFF background from a paired OFF observation.

    Turns a reduced ``MapDataset`` or ``SpectrumDataset``
    into its OnOff counterpart by mirroring the OFF run's events into
    the ON pointing FoV frame and setting acceptances from livetime.

    ON-OFF pairs should be created beforehand by the user.

    ..note..
        Currently, supports only one OFF observation for each ON observation.
        This works only for pointing observations, and drift mode is not supported.

    Parameters
    ----------
    acceptance_method : str, optional
        Allowed options are {"livetime_ratio", "counts_ratio"}
        Strategy for the acceptance calculation, can be either the ratio of livetimes or counts
            "livetime_ratio" takes the ratio of livetimes
            "counts_ratio" takes the ratio of the counts outside an exclusion mask, passed through kwargs
        Strategy for the acceptance calculation, can be either the ratio of livetimes or counts.
        Available options are:

            * "livetime_ratio": takes the ratio of livetimes.
            * "counts_ratio": takes the ratio of the counts outside an exclusion mask, 
              passed through kwargs.
              
        Default is "livetime_ratio".
    kwargs : dict, optional
        Additional arguments passed to the maker.
        Should contain `exclusion_mask` if acceptance_method is `counts_ratio`, 
        otherwise assume no mask.

    Returns
    -------
    dataset : `~gammapy.datasets.MapDatasetOnOff` or `~gammapy.datasets.SpectrumDatasetOnOff`
            Either a spectrum dataset or a map on off dataset.
    """

    tag = "OnOffBackgroundMaker"

    available_acceptance_methods = ["livetime_ratio", "counts_ratio"]

    def __init__(self, acceptance_method="livetime_ratio", **kwargs):
        if acceptance_method not in self.available_acceptance_methods:
            raise ValueError(
                f"Unknown acceptance_method {acceptance_method!r}, "
                f"choose from {self.available_acceptance_methods}"
            )

        self.acceptance_method = acceptance_method
        self.mask = kwargs.get("exclusion_mask", None)

    def _mirror_off_events(self, on_observation, off_observation):
        """Reproject OFF events into the ON pointing FoV frame (aligned with RA/DEC)."""
        events = off_observation.events
        off_pointing = off_observation.get_pointing_icrs(off_observation.tmid)
        on_pointing = on_observation.get_pointing_icrs(on_observation.tmid)

        fov_lon, fov_lat = sky_to_fov(
            events.radec.ra, events.radec.dec, off_pointing.ra, off_pointing.dec
        )
        off_ra, off_dec = fov_to_sky(fov_lon, fov_lat, on_pointing.ra, on_pointing.dec)
        table = events.table.copy()
        off_coord = SkyCoord(off_ra, off_dec, frame="icrs")
        table["RA"] = off_coord.ra
        table["DEC"] = off_coord.dec
        return EventList(table)

    def _make_acceptance(self, dataset, on_observation, off_observation, counts_off):
        geom = dataset.counts.geom

        if self.acceptance_method == "livetime_ratio":
            val_on = on_observation.observation_live_time_duration.to_value("s")
            val_off = off_observation.observation_live_time_duration.to_value("s")

        elif self.acceptance_method == "counts-ratio":
            if self.mask is None:
                mask = Map.from_geom(geom=geom, data=1, dtype=bool)
            val_on = (dataset.counts * mask).data.sum()
            val_off = (counts_off * mask).data.sum()

        acceptance = Map.from_geom(geom, unit="", data=val_on)
        acceptance_off = Map.from_geom(geom, unit="", data=val_off)
        return acceptance, acceptance_off

    def run(self, dataset, on_observation, off_observation):
        """Create background using dedicated off pointing.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Reduced dataset for the ON observation
        on_observation : `~gammapy.data.Observation`
            The ON observation.
        off_observation : `~gammapy.data.Observation`
            The paired OFF observation.

        Returns
        -------
        dataset : `~gammapy.datasets.MapDatasetOnOff` or `~gammapy.datasets.SpectrumDatasetOnOff`
            Either a spectrum dataset or a map on off dataset.
        """

        if (
            on_observation.pointing.mode == "DRIFT"
            or off_observation.pointing.mode == "DRIFT"
        ):
            log.warning(
                f"{on_observation.obs_id}/{off_observation.obs_id} pair is in DRIFT mode. IGNORING."
            )
            return

        if not isinstance(dataset, MapDataset):
            raise TypeError(
                f"Expected MapDataset or SpectrumDataset, got {type(dataset).__name__}"
            )

        geom = dataset.counts.geom
        counts_off = Map.from_geom(geom, unit="")
        counts_off.fill_events(self._mirror_off_events(on_observation, off_observation))

        acceptance, acceptance_off = self._make_acceptance(
            dataset, on_observation, off_observation, counts_off
        )

        if isinstance(dataset, SpectrumDataset):
            return SpectrumDatasetOnOff.from_spectrum_dataset(
                dataset=dataset,
                acceptance=acceptance,
                acceptance_off=acceptance_off,
                counts_off=counts_off,
                name=dataset.name,
            )
        elif isinstance(dataset, MapDataset):
            return MapDatasetOnOff.from_map_dataset(
                dataset=dataset,
                acceptance=acceptance,
                acceptance_off=acceptance_off,
                counts_off=counts_off,
                name=dataset.name,
            )
        return
