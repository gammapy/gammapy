# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""On-Off background estimation for dedicated On Off Pair runs"""

import numpy as np
import astropy.units as u
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
from gammapy.maps import Map
from gammapy.utils.coordinates import fov_to_sky, sky_to_fov

__all__ = ["OnOffBackgroundMaker", "check_run_pair_validity"]


def check_run_pair_validity(
    on_observations,
    off_observations,
    energy_range,
    zenith_tol=10 * u.deg,
    aeff_tol=0.1,
):
    """Validity check over a list of ON/OFF pairs.

    Parameters
    ----------
    on_observations : list of `~gammapy.data.Observation` or `~gammapy.data.Observations`
        The ON observations list
    off_observations : list of `~gammapy.data.Observation` or `~gammapy.data.Observations`
        The OFF observation list
    energy_range : `~gammapy.maps.MapAxis`
        The energy axis on which to check the effective area
    zenith_tol : `~astropy.units.Quantity`, optional
        Zenith tolerance, default 10 deg.
    aeff_tol : float, optional
        Effective-area fractional-deviation tolerance, default 0.1.

    Returns
    -------
    table : `~astropy.table.QTable`
        One row per pair, with columns ``obs_id``, ``off_obs_id``,
        ``zenith_on``, ``zenith_off``, ``delta_zenith``, ``zenith_ok``,
        ``aeff_max_frac_deviation`` and ``aeff_ok``.
    """
    if len(on_observations) != len(off_observations):
        raise ValueError(
            "on_observations and off_observations must have equal length; "
            f"got {len(on_observations)} and {len(off_observations)}"
        )

    results = []
    for on_obs, off_obs in zip(on_observations, off_observations):
        zenith_on = 90.0 * u.deg - on_obs.get_pointing_altaz(on_obs.tmid).alt
        zenith_off = 90.0 * u.deg - off_obs.get_pointing_altaz(off_obs.tmid).alt
        delta_zenith = abs(zenith_on - zenith_off)
        aeff_dev = _aeff_max_frac_deviation(on_obs, off_obs, energy_range)
        aeff_ok = aeff_dev is not None and aeff_dev < aeff_tol
        result = {
            "obs_id": on_obs.obs_id,
            "off_obs_id": off_obs.obs_id,
            "zenith_on": zenith_on,
            "zenith_off": zenith_off,
            "delta_zenith": delta_zenith,
            "zenith_ok": bool(delta_zenith < zenith_tol),
            "aeff_max_frac_deviation": aeff_dev,
            "aeff_ok": bool(aeff_ok),
        }
        results.append(result)

    return QTable(results)


def _aeff_max_frac_deviation(on_observation, off_observation, energy_axis):
    """Max fractional deviation between the two effective areas."""
    aeff_on = on_observation.aeff
    aeff_off = off_observation.aeff
    if aeff_on is None or aeff_off is None:
        return None

    energy = energy_axis.center
    offset = aeff_on.axes["offset"].center
    energy_grid, offset_grid = np.meshgrid(energy, offset, indexing="ij")

    a_on = aeff_on.evaluate(energy_true=energy_grid, offset=offset_grid).to_value("m2")
    a_off = aeff_off.evaluate(energy_true=energy_grid, offset=offset_grid).to_value(
        "m2"
    )

    mask = a_on > 0
    if not np.any(mask):
        return None
    return float(np.max(np.abs(a_off[mask] - a_on[mask]) / a_on[mask]))


class OnOffBackgroundMaker(Maker):
    """Estimate OFF background from a paired OFF observation.

    Turns a reduced ``MapDataset`` / ``SpectrumDataset``
    into its OnOff counterpart by mirroring the OFF run's events into
    the ON pointing FoV frame and setting acceptances from livetime.

    On-off pairs should be created beforehand by the user.

    Parameters
    ----------
    acceptance_method : str, optional
        Allowed option are {"livetime_ratio"}
        Default is "livetime"
        Strategy for the acceptance calculation
        "livetime_ratio" takes the ratio of livetimes, flat acceptance
    """

    tag = "OnOffBackgroundMaker"

    available_acceptance_methods = ["livetime_ratio"]

    def __init__(self, acceptance_method="livetime_ratio"):
        if acceptance_method not in self.available_acceptance_methods:
            raise ValueError(
                f"Unknown acceptance_method {acceptance_method!r}, "
                f"choose from {self.available_acceptance_methods}"
            )
        self.acceptance_method = acceptance_method

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

    def _make_acceptance(self, dataset, on_observation, off_observation):
        geom = dataset.counts.geom
        t_on = on_observation.observation_live_time_duration.to_value("s")
        t_off = off_observation.observation_live_time_duration.to_value("s")

        if self.acceptance_method == "livetime_ratio":
            acceptance = Map.from_geom(geom, unit="", data=t_on)
            acceptance_off = Map.from_geom(geom, unit="", data=t_off)
            return acceptance, acceptance_off

        raise ValueError(f"Unhandled acceptance_method {self.acceptance_method!r}")

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
        """
        geom = dataset.counts.geom
        counts_off = Map.from_geom(geom, unit="")
        counts_off.fill_events(self._mirror_off_events(on_observation, off_observation))

        acceptance, acceptance_off = self._make_acceptance(
            dataset, off_observation, on_observation
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
        raise TypeError(
            f"Expected MapDataset or SpectrumDataset, got {type(dataset).__name__}"
        )
