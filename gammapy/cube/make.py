# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from astropy.nddata.utils import NoOverlapError
from astropy.coordinates import Angle
from ..maps import Map, WcsGeom
from .counts import fill_map_counts
from .exposure import make_map_exposure_true_energy, _map_spectrum_weight
from .background import make_map_background_irf

__all__ = ["MapMaker", "MapMakerObs"]

log = logging.getLogger(__name__)


class MapMaker(object):
    """Make maps from IACT observations.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask
    """

    def __init__(self, geom, offset_max, exclusion_mask=None):
        if not isinstance(geom, WcsGeom):
            raise ValueError("MapMaker only works with WcsGeom")

        if geom.is_image:
            raise ValueError("MapMaker only works with geom with an energy axis")

        self.geom = geom
        self.offset_max = Angle(offset_max)
        self.maps = {}

        # Some background estimation methods need an exclusion mask.
        if exclusion_mask is not None:
            self.maps["exclusion"] = exclusion_mask

    def run(self, obs_list, selection=None):
        """
        Run MapMaker for a list of observations to create
        stacked counts, exposure and background maps

        Parameters
        --------------
        obs_list : `~gammapy.data.ObservationList`
            List of observations
        selection : list
            List of str, selecting which maps to make.
            Available: 'counts', 'exposure', 'background'
            By default, all maps are made.

        Returns
        -----------
        maps: dict of stacked counts, background and exposure maps.
        """
        selection = _check_selection(selection)

        # Initialise zero-filled maps
        for name in selection:
            unit = "m2 s" if name == "exposure" else ""
            self.maps[name] = Map.from_geom(self.geom, unit=unit)

        for obs in obs_list:
            try:
                self._process_obs(obs, selection)
            except NoOverlapError:
                log.info(
                    "Skipping observation {}, no overlap with map.".format(obs.obs_id)
                )
                continue

        return self.maps

    def _process_obs(self, obs, selection):
        # Compute cutout geometry and slices to stack results back later
        cutout_map = Map.from_geom(self.geom).cutout(
            position=obs.pointing_radec, width=2 * self.offset_max, mode="trim"
        )

        log.info("Processing observation: OBS_ID = {}".format(obs.obs_id))

        # Compute field of view mask on the cutout
        coords = cutout_map.geom.get_coord()
        offset = coords.skycoord.separation(obs.pointing_radec)
        fov_mask = offset >= self.offset_max

        # Only if there is an exclusion mask, make a cutout
        exclusion_mask = self.maps.get("exclusion", None)
        if exclusion_mask is not None:
            exclusion_mask = exclusion_mask.cutout(
                position=obs.pointing_radec, width=2 * self.offset_max, mode="trim"
            )

        # Make maps for this observation
        maps_obs = MapMakerObs(
            obs=obs,
            geom=cutout_map.geom,
            fov_mask=fov_mask,
            exclusion_mask=exclusion_mask,
        ).run(selection)

        # Stack observation maps to total
        for name in selection:
            data = maps_obs[name].quantity.to(self.maps[name].unit).value
            self.maps[name].fill_by_coord(coords, data)

    def make_images(self, spectrum=None):
        """Create 2D images by summing over the energy axis.

        Exposure is weighted with an assumed spectrum,
        resulting in a weighted mean exposure image.

        Parameters
        ----------
        spectrum : `~gammapy.spectrum.models.SpectralModel`
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.

        Returns
        -------
        images : dict of `~gammapy.maps.Map`
            2D images
        """
        images = {}
        for name, map in self.maps.items():
            if name == "exposure":
                map = _map_spectrum_weight(map, spectrum)

            images[name] = map.sum_over_axes()

        return images


class MapMakerObs(object):
    """Make maps for a single IACT observation.

    Parameters
    ----------
    obs : `~gammapy.data.DataStoreObservation`
        Observation
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    fov_mask : `~numpy.ndarray`
        Mask to select pixels in field of view
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask (used by some background estimators)
    """

    def __init__(self, obs, geom, fov_mask=None, exclusion_mask=None):
        self.obs = obs
        self.geom = geom
        self.fov_mask = fov_mask
        self.exclusion_mask = exclusion_mask
        self.maps = {}

    def run(self, selection=None):
        """Make maps.

        Returns dict with keys "counts", "exposure" and "background".

        Parameters
        ----------
        selection : list
            List of str, selecting which maps to make.
            Available: 'counts', 'exposure', 'background'
            By default, all maps are made.
        """
        selection = _check_selection(selection)

        for name in selection:
            getattr(self, "_make_" + name)()

        return self.maps

    def _make_counts(self):
        counts = Map.from_geom(self.geom)
        fill_map_counts(counts, self.obs.events)
        if self.fov_mask is not None:
            counts.data[..., self.fov_mask] = 0
        self.maps["counts"] = counts

    def _make_exposure(self):
        exposure = make_map_exposure_true_energy(
            pointing=self.obs.pointing_radec,
            livetime=self.obs.observation_live_time_duration,
            aeff=self.obs.aeff,
            geom=self.geom,
        )
        if self.fov_mask is not None:
            exposure.data[..., self.fov_mask] = 0
        self.maps["exposure"] = exposure

    def _make_background(self):
        background = make_map_background_irf(
            pointing=self.obs.pointing_radec,
            livetime=self.obs.observation_live_time_duration,
            bkg=self.obs.bkg,
            geom=self.geom,
        )
        if self.fov_mask is not None:
            background.data[..., self.fov_mask] = 0

        # TODO: decide what background modeling options to support
        # Extra things like FOV norm scale or ring would go here.

        self.maps["background"] = background


def _check_selection(selection):
    """Handle default and validation of selection"""
    available = ["counts", "exposure", "background"]

    if selection is None:
        selection = available

    if not isinstance(selection, list):
        raise TypeError("Selection must be a list of str")

    for name in selection:
        if name not in available:
            raise ValueError("Selection not available: {!r}".format(name))

    return selection
