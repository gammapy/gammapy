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
    MapDatasetOnOff
)
from .psf_map import make_psf_map

__all__ = ["MapDatasetMaker", "SafeMaskMaker"]

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

        # define cached methods
        self.make_exposure_irf = lru_cache(maxsize=1)(self.make_exposure_irf)

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

        kwargs = {
            "name": f"obs_{observation.obs_id}",
            "gti": observation.gti,
        }

        if "counts" in selection:
            counts = self.make_counts(observation)
            kwargs["counts"] = counts

        if "exposure" in selection:
            exposure = self.make_exposure(observation)
            kwargs["exposure"] = exposure

        if "background" in selection:
            background_map = self.make_background(observation)
            kwargs["background_model"] = BackgroundModel(background_map)

        if "psf" in selection:
            psf = self.make_psf(observation)
            kwargs["psf"] = psf

        if "edisp" in selection:
            edisp = self.make_edisp(observation)
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


class SafeMaskMaker:
    """Make safe data range mask for a given observation.

    Parameters
    ----------
    methods : {"aeff-default", "aeff-max", "edisp-bias", "offset-max"}
        Method to use for the safe energy range. Can be a
        list with a combination of those. Resulting masks
        are combined with logical `and`. "aeff-default"
        uses the energy ranged specified in the DL3 data
        files, if available.
    aeff_percent : float
        Percentage of the maximal effective area to be used
        as lower energy threshold for method "aeff-max".
    bias_percent : float
        Percentage of the energy bias to be used as lower
        energy threshold for method "edisp-bias"
    """

    def __init__(self, methods="aeff-default", aeff_percent=10, bias_percent=10, offset_max="3 deg"):
        self.methods = list(methods)
        self.aeff_percent = aeff_percent
        self.bias_percent = bias_percent
        self.offset_max = Angle(offset_max)

    def make_mask_offset_max(self, dataset, observation):
        """Make maximum offset mask.

        Parameters
        ----------
        dataset : Dataset`
            Dataset to compute mask for.
        observation: `DataStoreObservation`
            Observation to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Maximum offset mask.

        """
        separation = dataset.counts.geom.separation(observation.pointing_radec)
        return separation < self.offset_max

    @staticmethod
    def make_mask_energy_aeff_default(dataset, observation):
        """Make safe energy mask from aeff default.

        Parameters
        ----------
        dataset : `Dataset`
            Dataset to compute mask for.
        observation: `DataStoreObservation`
            Observation to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        try:
            e_max = observation.aeff.high_threshold
            e_min = observation.aeff.low_threshold
        except KeyError:
            log.warning(f"No thresholds defined for obs {observation}")
            e_min, e_max = None, None

        return dataset.counts.energy_mask(emin=e_min, emax=e_max)

    def make_mask_energy_aeff_max(self, dataset):
        """Make safe energy mask from aeff max.

        Parameters
        ----------
        dataset : `SpectrumDataset` or `SpectrumDatasetOnOff`
            Dataset to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        aeff_thres = self.aeff_percent / 100 * dataset.aeff.max_area
        e_min = dataset.aeff.find_energy(aeff_thres)
        return dataset.counts.energy_mask(emin=e_min)

    def make_mask_energy_edisp_bias(self, dataset):
        """Make safe energy mask from aeff max.

        Parameters
        ----------
        dataset : `SpectrumDataset` or `SpectrumDatasetOnOff`
            Dataset to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        e_min = dataset.edisp.get_bias_energy(self.bias_percent / 100)
        return dataset.counts.energy_mask(emin=e_min)

    def run(self, dataset, observation):
        """Make safe data range mask.

        Parameters
        ----------
        dataset : `Dataset`
            Dataset to compute mask for.
        observation: `DataStoreObservation`
            Observation to compute mask for.

        Returns
        -------
        dataset : `Dataset`
            Dataset with defined safe range mask.
        """
        mask_safe = np.ones(dataset.data_shape, dtype=bool)

        if "offset-max" in self.methods:
            mask_safe &= self.make_mask_offset_max(dataset, observation)

        if "aeff-default" in self.methods:
            mask_safe &= self.make_mask_energy_aeff_default(dataset, observation)

        if "aeff-max" in self.methods:
            mask_safe &= self.make_mask_energy_aeff_max(dataset)

        if "edisp-bias" in self.methods:
            mask_safe &= self.make_mask_energy_edisp_bias(dataset)

        if isinstance(dataset, (MapDataset, MapDatasetOnOff)):
            mask_safe = Map.from_geom(dataset.counts.geom, data=mask_safe)

        dataset.mask_safe = mask_safe
        return dataset
