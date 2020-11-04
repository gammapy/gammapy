# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.coordinates import Angle
from regions import PointSkyRegion
from gammapy.irf import EDispKernelMap, EffectiveAreaTable
from gammapy.maps import Map
from .core import Maker

__all__ = ["SafeMaskMaker"]


log = logging.getLogger(__name__)


class SafeMaskMaker(Maker):
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

    tag = "SafeMaskMaker"
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
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.
        observation: `~gammapy.data.Observation`
            Observation to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Maximum offset mask.
        """
        separation = dataset._geom.separation(observation.pointing_radec)
        return separation < self.offset_max

    @staticmethod
    def make_mask_energy_aeff_default(dataset, observation):
        """Make safe energy mask from aeff default.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.
        observation: `~gammapy.data.Observation`
            Observation to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        try:
            energy_max = observation.aeff.high_threshold
            energy_min = observation.aeff.low_threshold
        except KeyError:
            log.warning(f"No thresholds defined for obs {observation.obs_id}")
            energy_min, energy_max = None, None

        return dataset.counts.geom.energy_mask(
            energy_min=energy_min, energy_max=energy_max
        )

    def make_mask_energy_aeff_max(self, dataset):
        """Make safe energy mask from effective area maximum value.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        geom = dataset._geom

        if self.position is None:
            position = PointSkyRegion(dataset.counts.geom.center_skydir)
        else:
            position = PointSkyRegion(self.position)

        exposure = dataset.exposure.get_spectrum(position)

        energy = exposure.geom.axes["energy_true"]
        aeff = EffectiveAreaTable(
            energy_axis_true=energy,
            data=(exposure.quantity / dataset.gti.time_sum).squeeze(),
        )
        aeff_thres = (self.aeff_percent / 100) * aeff.max_area
        energy_min = aeff.find_energy(aeff_thres)
        return geom.energy_mask(energy_min=energy_min)

    def make_mask_energy_edisp_bias(self, dataset):
        """Make safe energy mask from energy dispersion bias.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        edisp, geom = dataset.edisp, dataset._geom

        position = self.position
        if position is None:
            position = dataset.counts.geom.center_skydir
            e_reco = dataset.counts.geom.axes["energy"].edges
        if isinstance(edisp, EDispKernelMap):
            edisp = edisp.get_edisp_kernel(position)
        else:
            edisp = edisp.get_edisp_kernel(position, e_reco)

        energy_min = edisp.get_bias_energy(self.bias_percent / 100)
        return geom.energy_mask(energy_min=energy_min)

    @staticmethod
    def make_mask_energy_bkg_peak(dataset):
        """Make safe energy mask based on the binned background.

        The energy threshold is defined as the upper edge of the energy
        bin with the highest predicted background rate. This method is motivated
        by its use in the HESS DL3 validation paper: https://arxiv.org/pdf/1910.08088.pdf

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        geom = dataset.counts.geom
        background_spectrum = dataset.npred_background().get_spectrum()
        idx = np.argmax(background_spectrum.data, axis=0)
        energy_axis = geom.axes["energy"]
        energy_min = energy_axis.pix_to_coord(idx)
        return geom.energy_mask(energy_min=energy_min)

    def run(self, dataset, observation=None):
        """Make safe data range mask.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.
        observation: `~gammapy.data.Observation`
            Observation to compute mask for.

        Returns
        -------
        dataset : `Dataset`
            Dataset with defined safe range mask.
        """
        mask_safe = np.ones(dataset._geom.data_shape, dtype=bool)

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

        dataset.mask_safe = Map.from_geom(dataset._geom, data=mask_safe)
        return dataset
