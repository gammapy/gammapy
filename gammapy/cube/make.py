# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.nddata.utils import NoOverlapError
from astropy.coordinates import Angle
from ..maps import Map, WcsGeom
from .counts import fill_map_counts
from .exposure import make_map_exposure_true_energy, _map_spectrum_weight
from .background import make_map_background_irf
from ..stats import significance_on_off
from scipy.stats import norm

__all__ = ["MapMaker", "MapMakerObs", "ImageMaker"]

log = logging.getLogger(__name__)


class MapMaker:
    """Make maps from IACT observations.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry in reco energy
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    geom_true : `~gammapy.maps.WcsGeom`
        Reference image geometry in true energy, used for exposure maps and PSF.
        If none, the same as geom is assumed
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask
    """

    def __init__(self, geom, offset_max, geom_true=None, exclusion_mask=None):
        if not isinstance(geom, WcsGeom):
            raise ValueError("MapMaker only works with WcsGeom")

        if geom.is_image:
            raise ValueError("MapMaker only works with geom with an energy axis")

        self.geom = geom
        self.geom_true = geom_true if geom_true else geom
        self.offset_max = Angle(offset_max)
        self.maps = {}

        # Some background estimation methods need an exclusion mask.
        if exclusion_mask is not None:
            self.maps["exclusion"] = exclusion_mask

    def run(self, observations, selection=None):
        """
        Run MapMaker for a list of observations to create
        stacked counts, exposure and background maps

        Parameters
        --------------
        observations : `~gammapy.data.Observations`
            Observations to process
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
            if name == "exposure":
                self.maps[name] = Map.from_geom(self.geom_true, unit="m2 s")
            else:
                self.maps[name] = Map.from_geom(self.geom, unit="")

        for obs in observations:
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
        cutout_geom = self.geom.cutout(position=obs.pointing_radec, width=2 * self.offset_max, mode="trim")
        cutout_geom_etrue = self.geom_true.cutout(position=obs.pointing_radec, width=2 * self.offset_max, mode="trim")
        log.info("Processing observation: OBS_ID = {}".format(obs.obs_id))

        # Compute field of view mask on the cutout
        coords = cutout_geom.get_coord()
        offset = coords.skycoord.separation(obs.pointing_radec)
        fov_mask = offset >= self.offset_max

        # Compute field of view mask on the cutout in true energy
        coords_etrue = cutout_geom_etrue.get_coord()
        offset_etrue = coords_etrue.skycoord.separation(obs.pointing_radec)
        fov_mask_etrue = offset_etrue >= self.offset_max

        # Only if there is an exclusion mask, make a cutout
        # Exclusion mask only on the background, so only in reco-energy
        exclusion_mask = self.maps.get("exclusion", None)
        if exclusion_mask is not None:
            exclusion_mask = exclusion_mask.cutout(
                position=obs.pointing_radec, width=2 * self.offset_max, mode="trim"
            )

        # Make maps for this observation
        maps_obs = MapMakerObs(
            observation=obs,
            geom=cutout_geom,
            geom_true=cutout_geom_etrue,
            fov_mask=fov_mask,
            fov_mask_etrue=fov_mask_etrue,
            exclusion_mask=exclusion_mask,
        ).run(selection)

        # Stack observation maps to total
        for name in selection:
            data = maps_obs[name].quantity.to_value(self.maps[name].unit)
            if name == "exposure":
                self.maps[name].fill_by_coord(coords_etrue, data)
            else:
                self.maps[name].fill_by_coord(coords, data)

    def make_images(self, spectrum=None, keepdims=False):
        """Create images by summing over the energy axis.

        Exposure is weighted with an assumed spectrum,
        resulting in a weighted mean exposure image.

        Parameters
        ----------
        spectrum : `~gammapy.spectrum.models.SpectralModel`
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.

        keepdims : bool, optional
            If this is set to True, the energy axes is kept with a single bin.
            If False, the energy axes is removed

        Returns
        -------
        images : dict of `~gammapy.maps.Map`
        """
        images = {}
        for name, map in self.maps.items():
            if name == "exposure":
                map = _map_spectrum_weight(map, spectrum)

            images[name] = map.sum_over_axes(keepdims=keepdims)

        return images


class MapMakerObs:
    """Make maps for a single IACT observation.

    Parameters
    ----------
    observation : `~gammapy.data.DataStoreObservation`
        Observation
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    geom_true : `~gammapy.maps.WcsGeom`
        Reference image geometry in true energy, used for exposure maps and PSF.
        If none, the same as geom is assumed
    fov_mask : `~numpy.ndarray`
        Mask to select pixels in field of view
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask (used by some background estimators)
    """

    def __init__(
            self,
            observation,
            geom,
            geom_true=None,
            fov_mask=None,
            fov_mask_etrue=None,
            exclusion_mask=None,
    ):
        self.observation = observation
        self.geom = geom
        self.geom_true = geom_true if geom_true else geom
        self.fov_mask = fov_mask
        self.fov_mask_etrue = fov_mask_etrue
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
        fill_map_counts(counts, self.observation.events)
        if self.fov_mask is not None:
            counts.data[..., self.fov_mask] = 0
        self.maps["counts"] = counts

    def _make_exposure(self):
        exposure = make_map_exposure_true_energy(
            pointing=self.observation.pointing_radec,
            livetime=self.observation.observation_live_time_duration,
            aeff=self.observation.aeff,
            geom=self.geom_true,
        )
        if self.fov_mask_etrue is not None:
            exposure.data[..., self.fov_mask_etrue] = 0
        self.maps["exposure"] = exposure

    def _make_background(self):
        background = make_map_background_irf(
            pointing=self.observation.fixed_pointing_info,
            ontime=self.observation.observation_time_duration,
            bkg=self.observation.bkg,
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


class ImageMaker():
    """Make 2D images.
    The main motivation for this class in addition to the `MapMaker`
    is to have the common 2D image background estimation methods,
    like `~gammapy.background.RingBackgroundEstimator`,
    that work using on and off maps.
    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask
    background_estimator : `~gammapy.background.RingBackgroundEstimator`
        Ring background estimator or something with an equivalend API.
    """

    def __init__(self, geom, offset_max, exclusion_mask=None, background_estimator=None):

        self.geom = geom
        self.offset_max = Angle(offset_max)
        self.background_estimator = background_estimator
        self.exclusion_mask = exclusion_mask
        self.stacked_maps = {}


    def make_maps(self, obs, summed=False):
        """ Returns a dict of on, off, background and alpha maps for each observation

        Parameters
        --------------
        obs : `~gammapy.data.Observations.DataStoreObservation`
            Observation to process

        summed: boolean
            Specifies whether computation should be done on
            the map summed over the energy axes, or
            for each spatial slice.

        Returns
        -----------
        maps: dict of on, off, background and alpha maps
        """


        selection = ["counts", "exposure", "background"]

        maker = MapMaker(self.geom, offset_max=self.offset_max)
        # Initialise zero-filled maps
        for name in selection:
            if name == "exposure":
                maker.maps[name] = Map.from_geom(self.geom, unit="m2 s")
            else:
                maker.maps[name] = Map.from_geom(self.geom, unit="")
        try:
            maker._process_obs(obs, selection)
        except NoOverlapError:
            log.info(
                "Skipping observation {}, no overlap with map.".format(obs.obs_id)
            )

        if summed:
            images = {'counts': maker.maps['counts'].sum_over_axes(),
                      'background': maker.maps['background'].sum_over_axes(),
                      'exclusion': self.exclusion_mask.sum_over_axes()}
        else:
            images = {'counts': maker.maps['counts'],
                    'background': maker.maps['background'],
                    'exclusion': self.exclusion_mask}

        result = self.background_estimator.run(images)
        maps = {'on': maker.maps['counts'],
                'alpha': result["alpha"],
                'off': result["off"],
                'background': result["background"]}
        return maps

    def run(self, observations, summed=False):
        """
        Run ImageMaker for a list of observations to create
        stacked on, off and alpha maps

        Parameters
        --------------
        observations : `~gammapy.data.Observations`
            Observations to process

        summed: boolean
            Specifies whether computation should be done on
            the map summed over the energy axes, or
            for each spatial slice.

        Returns
        -----------
        maps: dict of stacked on, off and alpha maps
        """
        results = []
        for obs in observations:
            results.append(self.make_maps(obs, summed=summed))

        self.stacked_maps["on"] = Map.from_geom(geom=self.geom)
        self.stacked_maps["off"] = Map.from_geom(geom=self.geom)
        self.stacked_maps["alpha"] = Map.from_geom(geom=self.geom)


        for aresult in results:
            self.stacked_maps["on"] += aresult['on']
            self.stacked_maps["off"] += aresult["off"]
            self.stacked_maps["alpha"] += aresult["off"] * aresult["alpha"]
        self.stacked_maps["alpha"] /= self.stacked_maps["off"]

    @property
    def significance_map(self):
        """returns the significance map for all pixels"""
        data = significance_on_off(n_on=self.stacked_maps["on"].data,
                                     n_off=self.stacked_maps["off"].data,
                                     alpha=self.stacked_maps["alpha"].data,
                                     method='lima')
        return self.stacked_maps["on"].copy(data=data)

    @property
    def excess_map(self):
        """returns the excess map for all pixels"""
        return self.stacked_maps["on"] -  self.stacked_maps["alpha"] * self.stacked_maps["off"]


    def plot(self, idx=[0]):
        """Makes some useful plots: the significance and excess maps,
        and the histograms of the significance.

        This can be done for each slice of the map by specifying idx.
        In case the maps are aleady in 2D, no idx is necessary

        Parameters
        ----------
        idx : tuple, optional
        Tuple of scalar indices for each non spatial dimension of the map.
        Tuple should be ordered as (I_0, ..., I_n).
        """
        if self.excess_map.geom.is_image:
            significance_map = self.significance_map
            excess_map = self.excess_map
            exclusion_mask = self.exclusion_mask
        else:
            significance_map = self.significance_map.get_image_by_idx(idx)
            excess_map = self.excess_map.get_image_by_idx(idx)
            exclusion_mask = self.exclusion_mask.get_image_by_idx(idx)


        import matplotlib.pyplot as plt


        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(221, projection=significance_map.geom.wcs)
        ax2 = plt.subplot(222, projection=excess_map.geom.wcs)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)

        ax1.set_title("Significance map")
        significance_map.plot(ax=ax1, add_cbar=True, stretch="sqrt")

        ax2.set_title("Excess map")
        excess_map.plot(ax=ax2, add_cbar=True, stretch="sqrt")

        significance_all = significance_map.data.ravel()
        significance_off = (significance_map * exclusion_mask).data.ravel()

        ax3.hist(significance_all, normed=True, alpha=0.5, color="red", label="all bins",
                 range=[np.nanmin(significance_all), np.nanmax(significance_all)])
        ax3.hist(significance_off, normed=True, alpha=0.5, color="blue", label="off bins",
                 range=[np.nanmin(significance_off), np.nanmax(significance_off)])
        ax3.legend()
        ax3.set_xlabel("significance")
        ax3.set_yscale("log")

        ax4.hist(significance_off, bins=20, normed=True, alpha=0.5, color="blue", label="off bins",
                 range=[np.nanmin(significance_off), np.nanmax(significance_off)])
        mu, std = norm.fit(significance_off[~np.isnan(significance_off)])
        xmin, xmax = ax4.get_xlim()
        x = np.linspace(xmin, xmax, 50)
        p = norm.pdf(x, mu, std)
        title = "Fit results: mu = %.2f, std= %.2f" %(mu, std)
        ax4.legend()
        ax4.set_title(title)
        ax4.plot(x, p, lw=2, color="black")
        ax4.set_xlabel("significance")
        ax4.set_yscale("log")





