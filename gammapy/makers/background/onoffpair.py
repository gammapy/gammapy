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
    """Validity check over a list of ON/OFF pairs. Currently supports only one OFF observation
    for each ON observation

    Checks for difference in zenith angle and effective area. Returns a summary table along
    with a flag if the match is sufficient within tolerance.

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


def _aeff_max_frac_deviation(
    on_observation, off_observation, energy_axis=None, offset_axis=None
):
    """Max fractional deviation between the two effective areas."""
    aeff_on = on_observation.aeff
    aeff_off = off_observation.aeff
    if aeff_on is None or aeff_off is None:
        return None

    if energy_axis is None:
        energy = aeff_on.axes["energy"].center
    else:
        energy = energy_axis.center
    if offset_axis is None:
        offset = aeff_on.axes["offset"].center
    else:
        offset = offset_axis.center
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

    ON-OFF pairs should be created beforehand by the user.

    Currently, supports only one OFF observation for each ON observation.

    This works only for pointing observations, and drift mode is not supported.

    Parameters
    ----------
    acceptance_method : str, optional
        Allowed option are {"livetime_ratio", "counts-ratio"}
        Default is "livetime"
        Strategy for the acceptance calculation
        "livetime_ratio" takes the ratio of livetimes
        "counts-ratio" takes the ratio of the counts outside an exclusion mask, passed through kwargs
    kwargs: Additional arguments passed to the maker
            Should contain `exclusion_mask` is acceptance_mathod is `counts_ratio`, otherwise assume no mask
    """

    tag = "OnOffBackgroundMaker"

    available_acceptance_methods = ["livetime_ratio", "counts-ratio"]

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
        """

        if (
            on_observation.pointing.mode == "DRIFT"
            or off_observation.pointing.mode == "DRIFT"
        ):
            raise ValueError("DRIFT mode not supported")
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
        raise TypeError(
            f"Expected MapDataset or SpectrumDataset, got {type(dataset).__name__}"
        )
