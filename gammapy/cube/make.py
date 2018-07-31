# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from astropy.nddata.utils import PartialOverlapError
from ..maps import WcsNDMap
from .counts import make_map_counts
from .exposure import make_map_exposure_true_energy
from .background import make_map_background_irf, make_map_background_fov

__all__ = [
    'MapMaker',
]

log = logging.getLogger(__name__)


class MapMaker(object):
    """Make all basic maps from observations.

    Parameters
    ----------
    ref_geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    cutout_mode : {'trim', 'strict'}, optional
        Options for making cutouts, see :func: `~gammapy.maps.WcsNDMap.make_cutout`
        Should be left to the default value 'trim'
        unless you want only fully contained observations to be added to the map
    """

    def __init__(self, ref_geom, offset_max, cutout_mode="trim"):
        self.offset_max = offset_max
        self.ref_geom = ref_geom

        # We instantiate the end products of the MakeMaps class
        self.counts_map = WcsNDMap(self.ref_geom)

        self.exposure_map = WcsNDMap(self.ref_geom, unit="m2 s")

        self.background_map = WcsNDMap(self.ref_geom)

        # We will need this general exclusion mask for the analysis
        self.exclusion_map = WcsNDMap(self.ref_geom)
        self.exclusion_map.data += 1

        self.cutout_mode = cutout_mode
        self.maps={}

    def process_obs(self, obs):
        """Process one observation and add it to the cutout image

        Parameters
        ----------
        obs : `~gammapy.data.DataStoreObservation`
            Observation
        """
        # First make cutout of the global image
        try:
            exclusion_mask_cutout, cutout_slices = self.exclusion_map.make_cutout(
                obs.pointing_radec, 2 * self.offset_max, mode=self.cutout_mode
            )
        except PartialOverlapError:
            # TODO: can we silently do the right thing here? Discuss
            log.info("Observation {} not fully contained in target image. Skipping it.".format(obs.obs_id))
            return

        cutout_geom = exclusion_mask_cutout.geom

        counts_obs_map = make_map_counts(
            obs.events, cutout_geom, obs.pointing_radec, self.offset_max,
        )

        expo_obs_map = make_map_exposure_true_energy(
            obs.pointing_radec, obs.observation_live_time_duration,
            obs.aeff, cutout_geom, self.offset_max,
        )

        acceptance_obs_map = make_map_background_irf(
            obs.pointing_radec, obs.observation_live_time_duration,
            obs.bkg, cutout_geom, self.offset_max,
        )

        background_obs_map = make_map_background_fov(
            acceptance_obs_map, counts_obs_map, exclusion_mask_cutout,
        )

        self._add_cutouts(cutout_slices, counts_obs_map, expo_obs_map, background_obs_map)


    def _add_cutouts(self, cutout_slices, counts_obs_map, expo_obs_map, acceptance_obs_map):
        """Add current cutout to global maps."""
        self.counts_map.data[cutout_slices] += counts_obs_map.data
        self.exposure_map.data[cutout_slices] += expo_obs_map.quantity.to(self.exposure_map.unit).value
        self.background_map.data[cutout_slices] += acceptance_obs_map.data

    def run(self, obs_list):
        """
        Run MapMaker for a list of observations to create
        stacked counts, exposure and background maps

        Parameters
        --------------
        obs_list: `~gammapy.data.ObservationList`
            List of observations

        Returns
        -----------
        maps: dict of stacked counts, background and exposure maps.
        """

        from astropy.utils.console import ProgressBar

        for obs in ProgressBar(obs_list):
            self.process_obs(obs)
        self.maps = {
            'counts_map': self.counts_map,
            'background_map': self.background_map,
            'exposure_map': self.exposure_map
                }
        return self.maps

