# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.coordinates import Angle
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.utils import lazyproperty
from gammapy.irf import EnergyDependentMultiGaussPSF
from gammapy.maps import Map, MapAxis, WcsGeom
from .background import make_map_background_irf
from .counts import fill_map_counts
from .edisp_map import make_edisp_map
from .exposure import _map_spectrum_weight, make_map_exposure_true_energy
from .psf_map import make_psf_map

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
    background_oversampling : int
        Background oversampling factor in energy axis.
    """

    def __init__(
        self,
        geom,
        offset_max,
        geom_true=None,
        exclusion_mask=None,
        background_oversampling=None,
    ):
        if not isinstance(geom, WcsGeom):
            raise ValueError("MapMaker only works with WcsGeom")

        if geom.is_image:
            raise ValueError("MapMaker only works with geom with an energy axis")

        self.geom = geom
        self.geom_true = geom_true if geom_true else geom
        self.offset_max = Angle(offset_max)
        self.exclusion_mask = exclusion_mask
        self.background_oversampling = background_oversampling

    def _get_empty_maps(self, selection):
        # Initialise zero-filled maps
        maps = {}
        for name in selection:
            if name == "exposure":
                maps[name] = Map.from_geom(self.geom_true, unit="m2 s")
            else:
                maps[name] = Map.from_geom(self.geom, unit="")
        return maps

    def run(self, observations, selection=["counts", "exposure", "background"]):
        """Make maps for a list of observations.

        Parameters
        ----------
        observations : `~gammapy.data.Observations`
            Observations to process
        selection : list
            List of str, selecting which maps to make.
            Available: 'counts', 'exposure', 'background'
            By default, all maps are made.

        Returns
        -------
        maps : dict
            Stacked counts, background and exposure maps
        """

        selection = _check_selection(selection)
        maps = self._get_empty_maps(selection)

        for obs in observations:
            log.info(f"Processing observation: OBS_ID = {obs.obs_id}")

            try:
                obs_maker = self._get_obs_maker(obs)
            except NoOverlapError:
                log.info(f"Skipping observation {obs.obs_id} (no map overlap)")
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
            background_oversampling=self.background_oversampling,
        )

    @staticmethod
    def _maps_sum_over_axes(maps, spectrum, keepdims):
        """Compute weighted sum over map axes.

        Parameters
        ----------
        spectrum : `~gammapy.modeling.models.SpectralModel`
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
        # TODO: PSF (and edisp) map sum_over_axis

        return images

    def run_images(self, observations=None, spectrum=None, keepdims=False):
        """Create images by summing over the energy axis.

        Either MapMaker.run() has to be called before calling this function,
        or observations need to be passed.
        If  MapMaker.run() has been called before, then those maps will be
        summed over. Else, new maps will be computed and then summed.

        Exposure is weighted with an assumed spectrum,
        resulting in a weighted mean exposure image.


        Parameters
        ----------
        observations : `~gammapy.data.Observations`
            Observations to process
        spectrum : `~gammapy.modeling.models.SpectralModel`
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

        return self._maps_sum_over_axes(self._maps, spectrum, keepdims)


class MapMakerObs:
    """Make maps for a single IACT observation.

    Parameters
    ----------
    observation : `~gammapy.data.DataStoreObservation`
        Observation
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry in reco energy, used for counts and background maps
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    geom_true : `~gammapy.maps.WcsGeom`
        Reference image geometry in true energy, used for IRF maps. It can have a coarser
        spatial bins than the counts geom.
        If none, the same as geom is assumed
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask (used by some background estimators)
    migra_axis : `~gammapy.maps.MapAxis`
        Migration axis for edisp map
    rad_axis : `~gammapy.maps.MapAxis`
        Radial axis for psf map
    """

    def __init__(
        self,
        observation,
        geom,
        offset_max,
        geom_true=None,
        exclusion_mask=None,
        background_oversampling=None,
        migra_axis=None,
        rad_axis=None,
    ):
        self.observation = observation
        self.geom = geom
        self.geom_true = geom_true if geom_true else geom
        self.offset_max = offset_max
        self.exclusion_mask = exclusion_mask
        self.background_oversampling = background_oversampling
        self.maps = {}
        self.migra_axis = migra_axis
        self.rad_axis = rad_axis

    def _fov_mask(self, coords):
        pointing = self.observation.pointing_radec
        offset = coords.skycoord.separation(pointing)
        return offset >= self.offset_max

    @lazyproperty
    def fov_mask(self):
        return self._fov_mask(self.coords)

    @lazyproperty
    def coords(self):
        return self.geom.get_coord()

    @lazyproperty
    def coords_etrue(self):
        return self.geom_true.get_coord()

    def run(self, selection=None):
        """Make maps.

        Returns dict with keys "counts", "exposure" and "background", "psf" and "edisp".

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

        if "exposure_irf" in self.maps:
            del self.maps["exposure_irf"]
        return self.maps

    def _make_counts(self):
        counts = Map.from_geom(self.geom)
        fill_map_counts(counts, self.observation.events)
        if self.fov_mask is not None:
            counts.data[..., self.fov_mask] = 0
        self.maps["counts"] = counts

    def _make_exposure(self):
        exposure_irf = make_map_exposure_true_energy(
            pointing=self.observation.pointing_radec,
            livetime=self.observation.observation_live_time_duration,
            aeff=self.observation.aeff,
            geom=self.geom_true,
        )
        # the exposure associated with the IRFS
        self.maps["exposure_irf"] = exposure_irf

        # The real exposure map, with FoV cuts
        factor = self.geom.data_shape[-1] / self.geom_true.data_shape[-1]
        exposure = exposure_irf.upsample(factor)
        coords_etrue = exposure.geom.get_coord()
        fov_mask_etrue = self._fov_mask(coords_etrue)
        if fov_mask_etrue is not None:
            exposure.data[..., fov_mask_etrue] = 0
        self.maps["exposure"] = exposure

    def _make_background(self):
        bkg_coordsys = self.observation.bkg.meta.get("FOVALIGN", "ALTAZ")
        if bkg_coordsys == "ALTAZ":
            pnt = self.observation.fixed_pointing_info
        elif bkg_coordsys == "RADEC":
            pnt = self.observation.pointing_radec
        else:
            raise ValueError(
                f"Invalid background coordinate system: {bkg_coordsys!r}\n"
                "Options: ALTAZ, RADEC"
            )
        background = make_map_background_irf(
            pointing=pnt,
            ontime=self.observation.observation_time_duration,
            bkg=self.observation.bkg,
            geom=self.geom,
            oversampling=self.background_oversampling,
        )
        if self.fov_mask is not None:
            background.data[..., self.fov_mask] = 0

        # TODO: decide what background modeling options to support
        # Extra things like FOV norm scale or ring would go here.

        self.maps["background"] = background

    def _make_edisp(self):
        energy_axis = self.geom_true.get_axis_by_name("ENERGY")
        if self.migra_axis is None:
            axes = {
                axis.name.lower(): axis for axis in self.observation.edisp.data.axes
            }
            self.migra_axis = axes["migra"]
        geom_migra = self.geom_true.to_image().to_cube([self.migra_axis, energy_axis])
        edisp_map = make_edisp_map(
            edisp=self.observation.edisp,
            pointing=self.observation.pointing_radec,
            geom=geom_migra,
            max_offset=self.offset_max,
            exposure_map=self.maps["exposure_irf"],
        )
        self.maps["edisp"] = edisp_map

    def _make_psf(self):
        psf = self.observation.psf
        if isinstance(psf, EnergyDependentMultiGaussPSF):
            psf = psf.to_psf3d()
        energy_axis = self.geom_true.get_axis_by_name("ENERGY")
        if self.rad_axis is None:
            rad = psf.rad_lo.value
            rad_irf = np.append(rad, psf.rad_hi.value[-1])
            self.rad_axis = MapAxis.from_edges(rad_irf, name="theta", unit="deg")
        geom_rad = self.geom_true.to_image().to_cube([self.rad_axis, energy_axis])
        psf_map = make_psf_map(
            psf=psf,
            pointing=self.observation.pointing_radec,
            geom=geom_rad,
            max_offset=self.offset_max,
            exposure_map=self.maps["exposure_irf"],
        )
        self.maps["psf"] = psf_map


def _check_selection(selection):
    """Handle default and validation of selection"""
    available = ["counts", "exposure", "background", "psf", "edisp"]
    if selection is None:
        selection = available

    if not isinstance(selection, list):
        raise TypeError("Selection must be a list of str")

    for name in selection:
        if name not in available:
            raise ValueError(f"Selection not available: {name!r}")

    return selection


class MapMakerRing(MapMaker):
    """Make maps from IACT observations.

    The main motivation for this class in addition to the `MapMaker`
    is to have the common image background estimation methods,
    like `~gammapy.cube.RingBackgroundEstimator`,
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
    background_estimator : `~gammapy.cube.RingBackgroundEstimator`
        or `~gammapy.cube.AdaptiveRingBackgroundEstimator`
        Ring background estimator or something with an equivalent API.

    Examples
    --------
    Here is an example how to ise the MapMakerRing with H.E.S.S. DL3 data::

        import numpy as np
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from regions import CircleSkyRegion
        from gammapy.maps import Map, WcsGeom, MapAxis
        from gammapy.cube import MapMakerRing, RingBackgroundEstimator
        from gammapy.data import DataStore

        # Create observation list
        data_store = DataStore.from_file(
            "$GAMMAPY_DATA/hess-dl3-dr1/hess-dl3-dr3-with-background.fits.gz"
        )
        data_sel = data_store.obs_table["TARGET_NAME"] == "MSH 15-52"
        obs_table = data_store.obs_table[data_sel]
        observations = data_store.get_observations(obs_table["OBS_ID"])

        # Define the geom
        pos = SkyCoord(228.32, -59.08, unit="deg")
        energy_axis = MapAxis.from_edges(np.logspace(0, 5.0, 5), unit="TeV", name="energy")
        geom = WcsGeom.create(skydir=pos, binsz=0.02, width=(5, 5), axes=[energy_axis])

        # Make a region mask
        regions = CircleSkyRegion(center=pos, radius=0.3 * u.deg)
        mask = Map.from_geom(geom)
        mask.data = mask.geom.region_mask([regions], inside=False)

        # Run map maker with ring background estimation
        ring_bkg = RingBackgroundEstimator(r_in="0.5 deg", width="0.3 deg")
        maker = MapMakerRing(
            geom=geom, offset_max="2 deg", exclusion_mask=mask, background_estimator=ring_bkg
        )
        images = maker.run_images(observations)
    """

    def __init__(
        self, geom, offset_max, exclusion_mask=None, background_estimator=None
    ):
        super().__init__(
            geom=geom,
            offset_max=offset_max,
            exclusion_mask=exclusion_mask,
            geom_true=None,
        )
        self.background_estimator = background_estimator

    def _run(self, observations, sum_over_axis=False, spectrum=None, keepdims=False):
        selection = ["on", "exposure_on", "off", "exposure_off", "exposure"]
        maps = self._get_empty_maps(selection)
        if sum_over_axis:
            maps = self._maps_sum_over_axes(maps, spectrum, keepdims)

        for obs in observations:
            try:
                obs_maker = self._get_obs_maker(obs, mode="strict")
            except NoOverlapError:
                log.info(f"Skipping obs_id: {obs.obs_id} (no map overlap)")
                continue
            except PartialOverlapError:
                log.info(f"Skipping obs_id: {obs.obs_id} (partial map overlap)")
                continue

            maps_obs = obs_maker.run(selection=["counts", "exposure", "background"])
            maps_obs["exclusion"] = obs_maker.exclusion_mask

            if sum_over_axis:
                maps_obs = self._maps_sum_over_axes(maps_obs, spectrum, keepdims)
                maps_obs["exclusion"] = obs_maker.exclusion_mask.sum_over_axes(
                    keepdims=keepdims
                )
                maps_obs["exclusion"].data = (
                    maps_obs["exclusion"].data / self.geom.axes[0].nbin
                )

            maps_obs_bkg = self.background_estimator.run(maps_obs)
            maps_obs.update(maps_obs_bkg)
            maps_obs["exposure_on"] = maps_obs.pop("background")
            maps_obs["on"] = maps_obs.pop("counts")

            # Now paste the returned maps on the ref geom
            for name in selection:
                data = maps_obs[name].quantity.to_value(maps[name].unit)
                maps[name].fill_by_coord(maps_obs[name].geom.get_coord(), data)

        self._maps = maps
        return maps

    def run_images(self, observations, spectrum=None, keepdims=False):
        """Run image making.

        The maps are summed over on the energy axis for a classical image analysis.

        Parameters
        ----------
        observations : `~gammapy.data.Observations`
            Observations to process
        spectrum : `~gammapy.modeling.models.SpectralModel`, optional
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        keepdims : bool, optional
            If this is set to True, the energy axes is kept with a single bin.
            If False, the energy axes is removed

        Returns
        -------
        maps : dict of `~gammapy.maps.Map`
            Dictionary containing the following maps:

            * ``"on"``: counts map
            * ``"exposure_on"``: on exposure map, which is just the
              template background map from the IRF
            * ``"exposure_off"``: off exposure map convolved with the ring
            * ``"off"``: off map
        """
        return self._run(
            observations, sum_over_axis=True, spectrum=spectrum, keepdims=keepdims
        )

    def run(self, observations):
        """Run map making.

        Parameters
        ----------
        observations : `~gammapy.data.Observations`
            Observations to process

        Returns
        -------
        maps : dict of `~gammapy.maps.Map`
            Dictionary containing the following maps:

            * ``"on"``: counts map
            * ``"exposure_on"``: on exposure map, which is just the
              template background map from the IRF
            * ``"exposure_off"``: off exposure map convolved with the ring
            * ``"off"``: off map
        """
        return self._run(observations, sum_over_axis=False)
