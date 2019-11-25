# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.coordinates import Angle
from gammapy.irf import EnergyDependentMultiGaussPSF
from gammapy.maps import Map
from gammapy.modeling.models import BackgroundModel
from .background import make_map_background_irf
from .edisp_map import make_edisp_map
from .exposure import make_map_exposure_true_energy
from .fit import MapDataset, MapDatasetOnOff
from .psf_map import make_psf_map

__all__ = ["MapDatasetMaker", "SafeMaskMaker"]

log = logging.getLogger(__name__)


class MapDatasetMaker:
    """Make maps for a single IACT observation.

    Parameters
    ----------
    cutout_width : `~astropy.coordinates.Angle`
        Maximum offset angle
    background_oversampling : int
        Background evaluation oversampling factor in energy.
    cutout_mode : {'trim', 'partial', 'strict'}
        Mode option for cutting out the observation,
        for details see `~astropy.nddata.utils.Cutout2D`.
    """

    def __init__(
        self, cutout_width, background_oversampling=None, cutout_mode="trim", cutout=True
    ):
        self.offset_max = Angle(offset_max)
        self.background_oversampling = background_oversampling
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

    def make_counts(self, dataset, observation):
        """Make counts map.

        Parameters
        ----------
        dataset : `MapDataset`
            Reference dataset.
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        counts : `Map`
            Counts map.
        """
        geom = self._cutout_geom(dataset.counts.geom, observation)
        counts = Map.from_geom(geom)
        counts.fill_events(observation.events)
        return counts

    def make_exposure(self, dataset, observation):
        """Make exposure map.

        Parameters
        ----------
        dataset : `MapDataset`
            Reference dataset.
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        exposure : `Map`
            Exposure map.
        """
        geom = self._cutout_geom(dataset.exposure.geom, observation)
        return make_map_exposure_true_energy(
            pointing=observation.pointing_radec,
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
        )

    def make_exposure_irf(self, geom, observation):
        """Make exposure map with irf geometry.

        Parameters
        ----------
        geom : `Geom`
            Reference geom.
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        exposure : `Map`
            Exposure map.
        """
        geom = self._cutout_geom(geom, observation)
        return make_map_exposure_true_energy(
            pointing=observation.pointing_radec,
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
        )

    def make_background(self, dataset, observation):
        """Make background map.

        Parameters
        ----------
        dataset : `MapDataset`
            Reference dataset.
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        background : `Map`
            Background map.
        """
        geom = self._cutout_geom(dataset.counts.geom, observation)

        bkg_coordsys = observation.bkg.meta.get("FOVALIGN", "RADEC")
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

    def make_edisp(self, dataset, observation):
        """Make edisp map.

        Parameters
        ----------
        dataset : `MapDataset`
            Reference dataset.
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        edisp : `EdispMap`
            Edisp map.
        """
        geom = self._cutout_geom(dataset.edisp.edisp_map.geom, observation)
        exposure = self.make_exposure_irf(dataset.edisp.exposure_map.geom, observation)

        return make_edisp_map(
            edisp=observation.edisp,
            pointing=observation.pointing_radec,
            geom=geom,
            exposure_map=exposure,
        )

    def make_psf(self, dataset, observation):
        """Make psf map.

        Parameters
        ----------
        dataset : `MapDataset`
            Reference dataset.
        observation : `DataStoreObservation`
            Observation container.

        Returns
        -------
        psf : `PSFMap`
            Psf map.
        """
        psf = observation.psf
        geom = self._cutout_geom(dataset.psf.psf_map.geom, observation)

        if isinstance(psf, EnergyDependentMultiGaussPSF):
            rad_axis = dataset.psf.psf_map.geom.get_axis_by_name("theta")
            psf = psf.to_psf3d(rad=rad_axis.center)

        exposure = self.make_exposure_irf(dataset.psf.exposure_map.geom, observation)

        return make_psf_map(
            psf=psf,
            pointing=observation.pointing_radec,
            geom=geom,
            exposure_map=exposure,
        )

    def run(self, dataset, observation, selection=None):
        """Make map dataset.

        Parameters
        ----------
        dataset : `MapDataset`
            Reference dataset.
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

        kwargs = {"name": f"obs_{observation.obs_id}", "gti": observation.gti}

        geom = self._cutout_geom(dataset.counts.geom, observation)
        mask_safe = Map.from_geom(geom, dtype=bool)
        mask_safe.data |= True

        kwargs["mask_safe"] = mask_safe

        if "counts" in selection:
            counts = self.make_counts(dataset, observation)
            kwargs["counts"] = counts

        if "exposure" in selection:
            exposure = self.make_exposure(dataset, observation)
            kwargs["exposure"] = exposure

        if "background" in selection:
            background_map = self.make_background(dataset, observation)
            kwargs["background_model"] = BackgroundModel(background_map)

        if "psf" in selection:
            psf = self.make_psf(dataset, observation)
            kwargs["psf"] = psf

        if "edisp" in selection:
            edisp = self.make_edisp(dataset, observation)
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
    methods : {"aeff-default", "aeff-max", "edisp-bias", "offset-max", "bkg-peak"}
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
    position : `~astropy.coordinates.SkyCoord`
        Position at which the `aeff_percent` or `bias_percent` are computed. By default,
        it uses the position of the center of the map.
    offset_max : str or `~astropy.units.Quantity`
        Maximum offset cut.
    """

    available_methods = {
        "aeff-default",
        "aeff-max",
        "edisp-bias",
        "offset-max",
        "bkg-peak",
    }

    def __init__(
        self,
        methods=("aeff-default",),
        aeff_percent=10,
        bias_percent=10,
        position=None,
        offset_max="3 deg",
    ):
        methods = set(methods)

        if not methods.issubset(self.available_methods):
            difference = methods.difference(self.available_methods)
            raise ValueError(f"{difference} is not a valid method.")

        self.methods = methods
        self.aeff_percent = aeff_percent
        self.bias_percent = bias_percent
        self.position = position
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

        # TODO: introduce RegionNDMap and simplify the code below
        try:
            mask = dataset.counts.energy_mask(emin=e_min, emax=e_max)
        except AttributeError:
            mask = dataset.counts.geom.energy_mask(emin=e_min, emax=e_max)

        return mask

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
        if isinstance(dataset, (MapDataset, MapDatasetOnOff)):
            raise NotImplementedError(
                "'aeff-max' method currently only supported for spectral datasets"
            )

        aeff_thres = self.aeff_percent / 100 * dataset.aeff.max_area
        e_min = dataset.aeff.find_energy(aeff_thres)
        return dataset.counts.energy_mask(emin=e_min)

    def make_mask_energy_edisp_bias(self, dataset):
        """Make safe energy mask from aeff max.

        Parameters
        ----------
        dataset : `MapDataset`, `MapDatasetOnOff`, `SpectrumDataset` or `SpectrumDatasetOnOff`
            Dataset to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        edisp = dataset.edisp

        if isinstance(dataset, (MapDataset, MapDatasetOnOff)):
            position = self.position
            if position is None:
                position = dataset.counts.geom.center_skydir
            e_reco = dataset.counts.geom.get_axis_by_name("energy").edges
            edisp = edisp.get_energy_dispersion(position, e_reco)
            counts = dataset.counts.geom
        else:
            counts = dataset.counts

        e_min = edisp.get_bias_energy(self.bias_percent / 100)
        return counts.energy_mask(emin=e_min)

    @staticmethod
    def make_mask_energy_bkg_peak(dataset):
        """Make safe energy mask based on the binned background.

        The energy threshold is defined as the upper edge of the energy
        bin with the highest predicted background rate. This method is motivated
        by its use in the HESS DL3 validation paper: https://arxiv.org/pdf/1910.08088.pdf

        Parameters
        ----------
        dataset : `MapDataset`, `MapDatasetOnOff`, `SpectrumDataset` or `SpectrumDatasetOnOff`
            Dataset to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """

        if isinstance(dataset, (MapDataset, MapDatasetOnOff)):
            background_spectrum = dataset.background_model.map.get_spectrum()
            counts = dataset.counts.geom
        else:
            background_spectrum = dataset.background
            counts = dataset.counts

        idx = np.argmax(background_spectrum.data)
        e_min = background_spectrum.energy.edges[idx + 1]
        return counts.energy_mask(emin=e_min)

    def run(self, dataset, observation=None):
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

        if "bkg-peak" in self.methods:
            mask_safe &= self.make_mask_energy_bkg_peak(dataset)

        if isinstance(dataset, (MapDataset, MapDatasetOnOff)):
            mask_safe = Map.from_geom(dataset.counts.geom, data=mask_safe)

        dataset.mask_safe = mask_safe
        return dataset
