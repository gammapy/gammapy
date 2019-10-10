# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from astropy.coordinates import Angle
from astropy.nddata.utils import NoOverlapError
from astropy.utils import lazyproperty
from gammapy.irf import EnergyDependentMultiGaussPSF
from gammapy.maps import Map, WcsGeom
from gammapy.modeling.models import BackgroundModel
from .background import make_map_background_irf
from .counts import fill_map_counts
from .edisp_map import make_edisp_map
from .exposure import _map_spectrum_weight, make_map_exposure_true_energy
from .fit import MIGRA_AXIS_DEFAULT, RAD_AXIS_DEFAULT, BINSZ_IRF, MapDataset
from .psf_map import make_psf_map

__all__ = ["MapMakerObs", "MapMakerRing"]

log = logging.getLogger(__name__)


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
        background_oversampling=None,
        migra_axis=None,
        rad_axis=None,
        cutout=True,
    ):

        cutout_kwargs = {
            "position": observation.pointing_radec,
            "width": 2 * Angle(offset_max),
            "mode": "trim",
        }

        if cutout:
            geom = geom.cutout(**cutout_kwargs)
            if geom_true is not None:
                geom_true = geom_true.cutout(**cutout_kwargs)

        self.observation = observation
        self.geom = geom
        self.geom_true = geom_true if geom_true else geom.to_binsz(BINSZ_IRF)
        self.offset_max = Angle(offset_max)
        self.background_oversampling = background_oversampling
        self.maps = {}
        self.migra_axis = migra_axis if migra_axis else MIGRA_AXIS_DEFAULT
        self.rad_axis = rad_axis if rad_axis else RAD_AXIS_DEFAULT

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
        """Make map dataset.


        Parameters
        ----------
        selection : list
            List of str, selecting which maps to make.
            Available: 'counts', 'exposure', 'background'
            By default, all maps are made.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.

        """
        selection = _check_selection(selection)

        for name in selection:
            getattr(self, "_make_" + name)()

        bkg = self.maps.get("background")

        if bkg is not None:
            background_model = BackgroundModel(bkg)
        else:
            background_model = None

        dataset = MapDataset(
            counts=self.maps.get("counts"),
            exposure=self.maps.get("exposure"),
            background_model=background_model,
            psf=self.maps.get("psf"),
            edisp=self.maps.get("edisp"),
            gti=self.observation.gti,
            name="obs_{}".format(self.observation.obs_id),
            mask_safe=~self.fov_mask,
        )
        return dataset

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
        mask_irf = self._fov_mask(self.geom_true.to_image().get_coord())
        exposure_irf_masked = exposure_irf.copy()
        exposure_irf_masked.data[..., mask_irf] = 0
        # the exposure associated with the IRFS
        self.maps["exposure_irf"] = exposure_irf_masked

        energy_axis = self.geom_true.get_axis_by_name("energy")
        geom = self.geom.to_image().to_cube([energy_axis])

        exposure = make_map_exposure_true_energy(
            pointing=self.observation.pointing_radec,
            livetime=self.observation.observation_live_time_duration,
            aeff=self.observation.aeff,
            geom=geom,
        )

        fov_mask_etrue = self._fov_mask(geom.to_image().get_coord())
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


class MapMakerRing:
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

        self.geom = geom
        self.offset_max = Angle(offset_max)
        self.exclusion_mask = exclusion_mask
        self.background_estimator = background_estimator

    def _get_empty_maps(self, selection):
        # Initialise zero-filled maps
        maps = {}
        for name in selection:
            if name == "exposure":
                maps[name] = Map.from_geom(self.geom, unit="m2 s")
            else:
                maps[name] = Map.from_geom(self.geom, unit="")
        return maps

    def _get_obs_maker(self, obs):
        # Compute cutout geometry and slices to stack results back later

        # Make maps for this observation
        return MapMakerObs(
            observation=obs,
            geom=self.geom,
            offset_max=self.offset_max,
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


    def _run(self, observations, sum_over_axis=False, spectrum=None, keepdims=False):
        selection = ["on", "exposure_on", "off", "exposure_off", "exposure"]
        maps = self._get_empty_maps(selection)
        if sum_over_axis:
            maps = self._maps_sum_over_axes(maps, spectrum, keepdims)

        for obs in observations:
            try:
                obs_maker = self._get_obs_maker(obs)
            except NoOverlapError:
                log.info(f"Skipping obs_id: {obs.obs_id} (no map overlap)")
                continue

            dataset = obs_maker.run(selection=["counts", "exposure", "background"])
            maps_obs = {}
            maps_obs["counts"] = dataset.counts
            maps_obs["exposure"] = dataset.exposure
            maps_obs["background"] = dataset.background_model.map
            maps_obs["exclusion"] = self.exclusion_mask.cutout(
                position=obs.pointing_radec, width=2 * self.offset_max, mode="trim"
                )

            if sum_over_axis:
                maps_obs = self._maps_sum_over_axes(maps_obs, spectrum, keepdims)
                maps_obs["exclusion"] = maps_obs["exclusion"].sum_over_axes(
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
