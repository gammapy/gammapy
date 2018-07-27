from __future__ import absolute_import, division, print_function, unicode_literals
from ..maps import WcsNDMap
from astropy.nddata.utils import PartialOverlapError
from .new import *

__all__ =['MapMaker']
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

    def __init__(self, ref_geom_etrue=None, ref_geom_ereco=None, offset_max, cutout_mode="trim"):
        self.offset_max = offset_max
        self.ref_geom_ereco = ref_geom_ereco
        self.ref_geom_etrue = ref_geom_etrue
        if ref_geom_etrue == None:
            self.ref_geom_etrue = ref_geom_ereco

        # We instantiate the end products of the MakeMaps class
        self.counts_map = WcsNDMap(self.ref_geom_ereco)

        self.exposure_map = WcsNDMap(self.ref_geom_etrue, unit="m2 s")

        self.background_map = WcsNDMap(self.ref_geom_ereco)

        # We will need this general exclusion mask for the analysis - for counts and bkg
        self.exclusion_map_etrue = WcsNDMap(self.ref_geom_ereco)
        self.exclusion_map_etrue.data += 1

        # For the exposure
        self.exclusion_map_ereco = WcsNDMap(self.ref_geom_ereco)
        self.exclusion_map_ereco.data += 1

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
            exclusion_mask_cutout_ereco, cutout_slices_ereco = self.exclusion_map_ereco.make_cutout(
                obs.pointing_radec, 2 * self.offset_max, mode=self.cutout_mode
            )
        except PartialOverlapError:
            # TODO: can we silently do the right thing here? Discuss
            log.info("Observation {} not fully contained in target image. Skipping it.".format(obs.obs_id))
            return

        cutout_geom_ereco = exclusion_mask_cutout_ereco.geom

        try:
            exclusion_mask_cutout_etrue, cutout_slices_etrue = self.exclusion_map_etrue.make_cutout(
                obs.pointing_radec, 2 * self.offset_max, mode=self.cutout_mode
            )
        except PartialOverlapError:
            # TODO: can we silently do the right thing here? Discuss
            log.info("Observation {} not fully contained in target image. Skipping it.".format(obs.obs_id))
            return

        cutout_geom_etrue = exclusion_mask_cutout_etrue.geom

        counts_obs_map = make_map_counts(
            obs.events, cutout_geom_ereco, obs.pointing_radec, self.offset_max,
        )

        expo_obs_map = make_map_exposure_true_energy(
            obs.pointing_radec, obs.observation_live_time_duration,
            obs.aeff, cutout_geom_etrue, self.offset_max,
        )

        acceptance_obs_map = make_map_background_irf(
            obs.pointing_radec, obs.observation_live_time_duration,
            obs.bkg, cutout_geom_ereco, self.offset_max,
        )

        background_obs_map = make_map_background_fov(
            acceptance_obs_map, counts_obs_map, exclusion_mask_cutout_ereco,
        )

        self._add_cutouts(cutout_slices_ereco, cutout_slices_etrue, counts_obs_map, expo_obs_map, background_obs_map)


    def _add_cutouts(self, cutout_slices_ereco, cutout_slices_etrue, counts_obs_map, expo_obs_map, acceptance_obs_map):
        """Add current cutout to global maps."""
        self.counts_map.data[cutout_slices_ereco] += counts_obs_map.data
        self.exposure_map.data[cutout_slices_etrue] += expo_obs_map.quantity.to(self.exposure_map.unit).value
        self.background_map.data[cutout_slices_ereco] += acceptance_obs_map.data

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

