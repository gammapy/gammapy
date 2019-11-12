# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from functools import lru_cache
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.utils import lazyproperty
from gammapy.irf import EnergyDependentMultiGaussPSF
from gammapy.maps import Map, WcsGeom
from gammapy.modeling.models import BackgroundModel
from .background import make_map_background_irf
from .edisp_map import make_edisp_map
from .exposure import make_map_exposure_true_energy
from .fit import (
    BINSZ_IRF_DEFAULT,
    MARGIN_IRF_DEFAULT,
    MIGRA_AXIS_DEFAULT,
    RAD_AXIS_DEFAULT,
    MapDataset,
)
from .psf_map import make_psf_map

__all__ = ["MapDatasetMaker"]

log = logging.getLogger(__name__)


class MapDatasetMaker:
    """Make maps for a single IACT observation.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry in reco energy, used for counts and background maps
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    energy_axis_true: `~gammapy.maps.MapAxis`
        True energy axis used for IRF maps
    migra_axis : `~gammapy.maps.MapAxis`
        Migration axis for edisp map
    rad_axis : `~gammapy.maps.MapAxis`
        Radial axis for psf map.
    binsz_irf: float
        IRF Map pixel size in degrees.
    margin_irf: float
        IRF map margin size in degrees
    cutout : bool
         Whether to cutout the observation.
    cutout_mode : {'trim', 'partial', 'strict'}
        Mode option for cutting out the observation,
        for details see `~astropy.nddata.utils.Cutout2D`.
    """

    def __init__(
        self,
        geom,
        offset_max,
        background_oversampling=None,
        energy_axis_true=None,
        migra_axis=None,
        rad_axis=None,
        binsz_irf=None,
        margin_irf=None,
        cutout_mode="trim",
        cutout=True,
    ):
        self.geom = geom
        self.offset_max = Angle(offset_max)
        self.background_oversampling = background_oversampling
        self.migra_axis = migra_axis if migra_axis else MIGRA_AXIS_DEFAULT
        self.rad_axis = rad_axis if rad_axis else RAD_AXIS_DEFAULT
        self.energy_axis_true = energy_axis_true or geom.get_axis_by_name("energy")
        self.binsz_irf = binsz_irf or BINSZ_IRF_DEFAULT

        self.margin_irf = margin_irf or MARGIN_IRF_DEFAULT
        self.margin_irf = self.margin_irf * u.deg

        self.cutout_mode = cutout_mode
        self.cutout_width = 2 * self.offset_max
        self.cutout = cutout

    def _cutout_geom(self, geom, observation):
        if self.cutout:
            return geom.cutout(
                position=observation.pointing_radec,
                width=self.cutout_width,
                mode=self.cutout_mode,
            )
        else:
            return geom

    @lazyproperty
    def geom_image_irf(self):
        """Spatial geometry of IRF Maps (`Geom`)"""
        wcs = self.geom.to_image()
        return WcsGeom.create(
            binsz=self.binsz_irf,
            width=wcs.width + self.margin_irf,
            skydir=wcs.center_skydir,
            proj=wcs.projection,
            coordsys=wcs.coordsys,
        )

    @lazyproperty
    def geom_exposure_irf(self):
        """Geom of Exposure map associated with IRFs (`Geom`)"""
        return self.geom_image_irf.to_cube([self.energy_axis_true])

    @lazyproperty
    def geom_exposure(self):
        """Exposure map geom (`Geom`)"""
        geom_exposure = self.geom.to_image().to_cube([self.energy_axis_true])
        return geom_exposure

    @lazyproperty
    def geom_psf(self):
        """PSFMap geom (`Geom`)"""
        geom_psf = self.geom_image_irf.to_cube([self.rad_axis, self.energy_axis_true])
        return geom_psf

    @lazyproperty
    def geom_edisp(self):
        """EdispMap geom (`Geom`)"""
        return self.geom_image_irf.to_cube([self.migra_axis, self.energy_axis_true])

    def make_counts(self, observation):
        """Make counts map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        counts : `Map`
            Counts map.
        """
        geom = self._cutout_geom(self.geom, observation)
        counts = Map.from_geom(geom)
        counts.fill_events(observation.events)
        return counts

    def make_exposure(self, observation):
        """Make exposure map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        exposure : `Map`
            Exposure map.
        """
        geom = self._cutout_geom(self.geom_exposure, observation)
        return make_map_exposure_true_energy(
            pointing=observation.pointing_radec,
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
        )

    @lru_cache(maxsize=1)
    def make_exposure_irf(self, observation):
        """Make exposure map with irf geometry.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        exposure : `Map`
            Exposure map.
        """
        geom = self._cutout_geom(self.geom_exposure_irf, observation)
        return make_map_exposure_true_energy(
            pointing=observation.pointing_radec,
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
        )

    def make_background(self, observation):
        """Make background map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        background : `Map`
            Background map.
        """
        geom = self._cutout_geom(self.geom, observation)

        bkg_coordsys = observation.bkg.meta.get("FOVALIGN", "ALTAZ")
        if bkg_coordsys == "ALTAZ":
            pointing = observation.fixed_pointing_info
        elif bkg_coordsys == "RADEC":
            pointing = observation.pointing_radec
        else:
            raise ValueError(
                f"Invalid background coordinate system: {bkg_coordsys!r}\n"
                "Options: ALTAZ, RADEC"
            )

        return make_map_background_irf(
            pointing=pointing,
            ontime=observation.observation_time_duration,
            bkg=observation.bkg,
            geom=geom,
            oversampling=self.background_oversampling,
        )

    def make_edisp(self, observation):
        """Make edisp map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        edisp : `EdispMap`
            Edisp map.
        """
        geom = self._cutout_geom(self.geom_edisp, observation)

        exposure = self.make_exposure_irf(observation)

        return make_edisp_map(
            edisp=observation.edisp,
            pointing=observation.pointing_radec,
            geom=geom,
            max_offset=self.offset_max,
            exposure_map=exposure,
        )

    def make_psf(self, observation):
        """Make psf map.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        psf : `PSFMap`
            Psf map.
        """
        psf = observation.psf
        geom = self._cutout_geom(self.geom_psf, observation)

        if isinstance(psf, EnergyDependentMultiGaussPSF):
            psf = psf.to_psf3d(rad=self.rad_axis.center)

        exposure = self.make_exposure_irf(observation)

        return make_psf_map(
            psf=psf,
            pointing=observation.pointing_radec,
            geom=geom,
            max_offset=self.offset_max,
            exposure_map=exposure,
        )

    @lru_cache(maxsize=1)
    def make_mask_safe(self, observation):
        """Make offset mask.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        mask : `Map`
            Mask
        """
        geom = self._cutout_geom(self.geom.to_image(), observation)
        offset = geom.separation(observation.pointing_radec)
        data = offset >= self.offset_max
        return Map.from_geom(geom, data=data)

    @lru_cache(maxsize=1)
    def make_mask_safe_irf(self, observation):
        """Make offset mask with irf geometry.

        Parameters
        ----------
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        mask : `Map`
            Mask
        """
        geom = self._cutout_geom(self.geom_exposure_irf.to_image(), observation)
        offset = geom.separation(observation.pointing_radec)
        data = offset >= self.offset_max
        return Map.from_geom(geom, data=data)

    def run(self, observation, selection=None):
        """Make map dataset.

        Parameters
        ----------
        observation : `~gammapy.data.DataStoreObservation`
            Observation
        selection : list
            List of str, selecting which maps to make.
            Available: 'counts', 'exposure', 'background', 'psf', 'edisp'
            By default, all maps are made.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.
        """
        selection = _check_selection(selection)

        mask_safe = self.make_mask_safe(observation)
        energy_axis = self.geom.get_axis_by_name("energy")
        mask_safe_3d = (
            ~mask_safe.data
            & np.ones(energy_axis.nbin, dtype=bool)[:, np.newaxis, np.newaxis]
        )
        mask_map = Map.from_geom(
            mask_safe.geom.to_cube([energy_axis]), data=mask_safe_3d
        )
        mask_safe_irf = self.make_mask_safe_irf(observation)

        kwargs = {
            "name": f"obs_{observation.obs_id}",
            "gti": observation.gti,
            "mask_safe": mask_map,
        }

        if "counts" in selection:
            counts = self.make_counts(observation)
            # TODO: remove masking out the values here and instead handle the safe mask only when
            #  fitting and / or stacking datasets?
            counts.data[..., mask_safe.data] = 0
            kwargs["counts"] = counts

        if "exposure" in selection:
            exposure = self.make_exposure(observation)
            exposure.data[..., mask_safe.data] = 0
            kwargs["exposure"] = exposure

        if "background" in selection:
            background_map = self.make_background(observation)
            background_map.data[..., mask_safe.data] = 0
            kwargs["background_model"] = BackgroundModel(background_map)

        if "psf" in selection:
            psf = self.make_psf(observation)
            psf.exposure_map.data[..., mask_safe_irf.data] = 0
            kwargs["psf"] = psf

        if "edisp" in selection:
            edisp = self.make_edisp(observation)
            edisp.exposure_map.data[..., mask_safe_irf.data] = 0
            kwargs["edisp"] = edisp

        return MapDataset(**kwargs)


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
