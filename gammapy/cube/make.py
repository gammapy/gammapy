# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.coordinates import Angle
from astropy.convolution import Tophat2DKernel
from astropy.utils import lazyproperty
from ..maps import Map, WcsGeom
from .counts import fill_map_counts
from .exposure import make_map_exposure_true_energy, _map_spectrum_weight
from .background import make_map_background_irf
from ..stats import significance_on_off
from scipy.stats import norm

__all__ = ["MapMaker", "MapMakerObs", "MapMakerRing"]

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
        self.exclusion_mask = exclusion_mask

    def _get_empty_maps(self, selection):
        # Initialise zero-filled maps
        maps = {}
        for name in selection:
            if name == "exposure":
                maps[name] = Map.from_geom(self.geom_true, unit="m2 s")
            else:
                maps[name] = Map.from_geom(self.geom, unit="")
        return maps

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
        maps = self._get_empty_maps(selection)

        for obs in observations:
            log.info("Processing observation: OBS_ID = {}".format(obs.obs_id))

            try:
                obs_maker = self._get_obs_maker(obs)
            except NoOverlapError:
                log.info(
                    "Skipping observation {}, no overlap with map.".format(obs.obs_id)
                )
                continue

            maps_obs = obs_maker.run(selection)

            for name in selection:
                data = maps_obs[name].quantity.to_value(maps[name].unit)
                if name == "exposure":
                    maps[name].fill_by_coord(obs_maker.coords_etrue, data)
                else:
                    maps[name].fill_by_coord(obs_maker.coords, data)
        self._maps = maps
        return maps

    def _get_obs_maker(self, obs, mode="trim"):
        # Compute cutout geometry and slices to stack results back later
        cutout_kwargs = {
            "position": obs.pointing_radec,
            "width": 2 * self.offset_max,
            "mode": mode,
        }

        cutout_geom = self.geom.cutout(**cutout_kwargs)
        cutout_geom_etrue = self.geom_true.cutout(**cutout_kwargs)

        if self.exclusion_mask is not None:
            cutout_exclusion = self.exclusion_mask.cutout(**cutout_kwargs)
        else:
            cutout_exclusion = None

        # Make maps for this observation
        return MapMakerObs(
            observation=obs,
            geom=cutout_geom,
            geom_true=cutout_geom_etrue,
            offset_max=self.offset_max,
            exclusion_mask=cutout_exclusion,
        )

    @staticmethod
    def _maps_sum_over_axes(maps, spectrum, keepdims):
        """Compute weighted sum over map axes.

        Parameters
        ----------
        spectrum : `~gammapy.spectrum.models.SpectralModel`
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.

        keepdims : bool, optional
            If this is set to True, the energy axes is kept with a single bin.
            If False, the energy axes is removed

        """
        images = {}
        for name, map in maps.items():
            if name == "exposure":
                map = _map_spectrum_weight(map, spectrum)

            images[name] = map.sum_over_axes(keepdims=keepdims)
        return images

    def run_images(self, observations=None, spectrum=None, keepdims=False):
        """Create images by summing over the energy axis.

        Exposure is weighted with an assumed spectrum,
        resulting in a weighted mean exposure image.

        Parameters
        ----------
        observations: ...
            TODO
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
        if not hasattr(self, "_maps"):
            if observations is None:
                raise ValueError("Requires observations...")
            self.run(observations)

        images = self._maps_sum_over_axes(self._maps, spectrum, keepdims)
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
        self, observation, geom, offset_max, geom_true=None, exclusion_mask=None
    ):
        self.observation = observation
        self.geom = geom
        self.geom_true = geom_true if geom_true else geom
        self.offset_max = offset_max
        self.exclusion_mask = exclusion_mask
        self.maps = {}

    def _fov_mask(self, coords):
        pointing = self.observation.pointing_radec
        offset = coords.skycoord.separation(pointing)
        fov_mask = offset >= self.offset_max
        return fov_mask

    @lazyproperty
    def fov_mask_etrue(self):
        return self._fov_mask(self.coords_etrue)

    @lazyproperty
    def fov_mask(self):
        return self._fov_mask(self.coords)

    @lazyproperty
    def coords(self):
        coords = self.geom.get_coord()
        return coords

    @lazyproperty
    def coords_etrue(self):
        # Compute field of view mask on the cutout in true energy
        coords_etrue = self.geom_true.get_coord()
        return coords_etrue

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


class MapMakerRing(MapMaker):
    """Make maps from IACT observations.

    The main motivation for this class in addition to the `MapMaker`
    is to have the common image background estimation methods,
    like `~gammapy.background.RingBackgroundEstimator`,
    that work using on and off maps.

    To ensure adequate statistics, only observations that are fully
    contained within the reference geometry will be analysed

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask
    background_estimator : `~gammapy.background.RingBackgroundEstimator`
        or `~gammapy.background.AdaptiveRingBackgroundEstimator`
        Ring background estimator or something with an equivalent API.

    """

    def __init__(
        self, geom, offset_max, exclusion_mask=None, background_estimator=None
    ):
        super(MapMakerRing, self).__init__(
            geom=geom,
            offset_max=offset_max,
            exclusion_mask=exclusion_mask,
            geom_true=None,
        )
        self.background_estimator = background_estimator

    def _stack(self, maps_obs, obs_maker):
        selection = ["counts", "exposure", "background"]
        maps = self._get_empty_maps(selection)
        for name in selection:
            data = maps_obs[name].quantity.to_value(maps[name].unit)
            if name == "exposure":
                maps[name].fill_by_coord(obs_maker.coords_etrue, data)
            else:
                maps[name].fill_by_coord(obs_maker.coords, data)
        return maps

    def _run(self, observations, sum_over_axis=False, spectrum=None, keepdims=False):
        """
        Parameters
        --------------
        observations : `~gammapy.data.Observations`
            Observations to process

        Returns
        -----------
        maps: list of dict of maps. Each observation will have the following maps
            counts: The counts map
            background_irf: The template background map from the IRF
            exposure: The uncorrelated on exposure map
            exposure_off: The off exposure map convolved with the ring
            alpha: The alpha map (1/exposure map)
            off: The off map
            background_ring: The ring background map ( = alpha * off)
            exclusion: The exclusion mask

        """

        map_list = []
        for obs in observations:
            try:
                obs_maker = self._get_obs_maker(obs, mode="strict")
            except NoOverlapError:
                log.info(
                    "Skipping observation {}, no overlap with map.".format(obs.obs_id)
                )
            except PartialOverlapError:
                log.info(
                    "Skipping observation {}, partial overlap with map.".format(
                        obs.obs_id
                    )
                )
                continue

            maps_obs = obs_maker.run()

            # Now paste it back on ref geom
            maps_obs = self._stack(maps_obs, obs_maker)

            maps_obs["exclusion"] = self.exclusion_mask

            if sum_over_axis:
                maps_obs = self._maps_sum_over_axes(maps_obs, spectrum, keepdims)
                maps_obs["exclusion"] = self.exclusion_mask.get_image_by_idx([0])

            maps_obs_bkg = self.background_estimator.run(maps_obs)
            maps_obs.update(maps_obs_bkg)
            map_list.append(maps_obs)

        return map_list

    def run_images(self, observations, spectrum=None, keepdims=False):
        """Returns a list of dictionaries of 2D maps.
        The maps are summed over on the energy axis for a classical image analysis

        Parameters
        ---------

        observations: `~gammapy.data.Observations`
            Observations to process
        spectrum : `~gammapy.spectrum.models.SpectralModel`, optional
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        keepdims : bool, optional
            If this is set to True, the energy axes is kept with a single bin.
            If False, the energy axes is removed

        Returns
        ---------
        maps: list of dict of maps. Each observation will have the following maps
            counts: The counts map
            background_irf: The template background map from the IRF
            exposure: The uncorrelated on exposure map
            exposure_off: The off exposure map convolved with the ring
            alpha: The alpha map (1/exposure map)
            off: The off map
            background_ring: The ring background map ( = alpha * off)

        """
        return self._run(
            observations, sum_over_axis=True, spectrum=spectrum, keepdims=keepdims
        )

    def run(self, observations):
        """Returns a list of dictionaries of 3D maps
        Significance and excess can be computed for each slice

        Returns
        ---------
        maps: list of dict of maps. Each observation will have the following maps
            counts: The counts map
            background_irf: The template background map from the IRF
            exposure: The uncorrelated on exposure map
            exposure_off: The off exposure map convolved with the ring
            alpha: The alpha map (1/exposure map)
            off: The off map
            background_ring: The ring background map ( = alpha * off)

        """
        return self._run(observations, sum_over_axis=False)

    def significance_map(self, maplist, convolution_radius=None):
        """
        Parameters
        ---------
        maplist: the list of dictionaries returned by MapMakerRing.run()
                or MapMakerRing.run_images()
        convolution_radius: `~astropy.units.Quantity`
            The convolution radius for Tophat2DKernel

            If not specified, returns the uncorrelated significance map

        Returns
        -------
        significance_map: `~gammapy.maps.Map`
        Stacked significance map for the entire region.

        """
        if convolution_radius:
            scale = counts.geom.pixel_scales[0].to("deg")
            theta = (convolution_radius * scale).value
        else:
            theta = 1
        tophat = Tophat2DKernel(theta)
        tophat.normalize("peak")
        selection = ["on", "off", "exposure_on", "exposure_off"]

        stacked_map = {}
        for name in selection:
            stacked_map[name] = Map.from_geom(maplist[0]["counts"].geom)

        for amap in maplist:
            stacked_map["on"] += amap["counts"]
            stacked_map["off"] += amap["off"]
            stacked_map["exposure_on"] += amap["background"]
            stacked_map["exposure_off"] += amap["exposure_off"]
        stacked_map["exposure_on"] = stacked_map["exposure_on"].convolve(tophat.array)
        stacked_map["exposure_off"] = stacked_map["exposure_off"].convolve(tophat.array)
        stacked_map["on"] = stacked_map["on"].convolve(tophat.array)
        stacked_map["off"] = stacked_map["off"].convolve(tophat.array)
        alpha_map = stacked_map["exposure_on"] / stacked_map["exposure_off"]

        data = significance_on_off(
            n_on=stacked_map["on"].data,
            n_off=stacked_map["off"].data,
            alpha=alpha_map.data,
            method="lima",
        )
        return stacked_map["on"].copy(data=data)

    def significance_map_off(self, maplist, convolution_radius=None):
        """returns the significance map with exclusion region applied

        Parameters
        ---------
        maplist: the list of dictionaries returned by MapMakerRing.run()
                or MapMakerRing.run_images()
        convolution_radius: `~astropy.units.Quantity`
            The convolution radius for Tophat2DKernel

            If not specified, returns the uncorrelated significance map

        Returns
        -------
        significance_map_off: `~gammapy.maps.Map`
        Stacked significance map for the off regions region.

        """

        significance_map = self.significance_map(maplist, convolution_radius)
        if significance_map.geom.is_image:
            return significance_map * self.exclusion_mask.get_image_by_idx([0])
        else:
            return significance_map * self.exclusion_mask

    def excess_map(self, maplist):
        """
        excess = on - alpha * off

        Parameters
        ---------
        maplist: the list of dictionaries returned by MapMakerRing.run()
                or MapMakerRing.run_images()

        Returns
        -------
        excess_map: `~gammapy.maps.Map`
        Stacked excess map for the entire region for each pixel.
        """

        selection = ["on", "alpha", "off"]
        stacked_map = {}
        for name in selection:
            stacked_map[name] = Map.from_geom(maplist[0]["counts"].geom)
        for amap in maplist:
            stacked_map["on"] += amap["counts"]
            stacked_map["off"] += amap["off"]
            stacked_map["alpha"] += amap["off"] * amap["alpha"]
        stacked_map["alpha"] /= stacked_map["off"]
        excess_map = stacked_map["on"] - stacked_map["alpha"] * stacked_map["off"]

        return excess_map

    def excess_map_off(self, maplist):
        """
        returns the excess map with exclusion region applied

        Parameters
        ---------
        maplist: the list of dictionaries returned by MapMakerRing.run()
                or MapMakerRing.run_images()

        Returns
        -------
        excess_map_off: `~gammapy.maps.Map`
        Stacked excess map for the off for each pixel.
        """

        excess_map = self.excess_map(maplist)
        if excess_map.geom.is_image:
            return excess_map * self.exclusion_mask.get_image_by_idx([0])
        else:
            return excess_map * self.exclusion_mask

    def plot(self, maplist, convolution_radius=None, idx=[0]):
        """Makes some useful plots: the significance and excess maps,
        and the histograms of the significance.

        This can be done for each slice of the map by specifying idx.
        In case the maps are aleady in 2D, no idx is necessary

        Parameters
        ----------
        maplist : list of dictionaries of maps

        idx : tuple, optional
            Tuple of scalar indices for each non spatial dimension of the map.
            Tuple should be ordered as (I_0, ..., I_n).

        """
        excess_map = self.excess_map(maplist)
        significance_map = self.significance_map(maplist, convolution_radius)
        significance_map_off = self.significance_map_off(maplist, convolution_radius)
        if excess_map.geom.is_image is False:
            significance_map = significance_map.get_image_by_idx(idx)
            excess_map = excess_map.get_image_by_idx(idx)
            significance_map_off = significance_map_off.get_image_by_idx(idx)

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
        significance_off = significance_map_off.data.ravel()

        ax3.hist(
            significance_all,
            normed=True,
            alpha=0.5,
            color="red",
            label="all bins",
            range=[np.nanmin(significance_all), np.nanmax(significance_all)],
        )
        ax3.hist(
            significance_off,
            normed=True,
            alpha=0.5,
            color="blue",
            label="off bins",
            range=[np.nanmin(significance_off), np.nanmax(significance_off)],
        )
        ax3.legend()
        ax3.set_xlabel("significance")
        ax3.set_yscale("log")

        ax4.hist(
            significance_off,
            bins=20,
            normed=True,
            alpha=0.5,
            color="blue",
            label="off bins",
            range=[np.nanmin(significance_off), np.nanmax(significance_off)],
        )
        mu, std = norm.fit(significance_off[~np.isnan(significance_off)])
        xmin, xmax = ax4.get_xlim()
        x = np.linspace(xmin, xmax, 50)
        p = norm.pdf(x, mu, std)
        title = "Fit results: mu = %.2f, std= %.2f" % (mu, std)
        ax4.legend()
        ax4.set_title(title)
        ax4.plot(x, p, lw=2, color="black")
        ax4.set_xlabel("significance")
        ax4.set_yscale("log")
