# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from gammapy.irf import EDispKernelMap
from gammapy.maps import Map
from gammapy.modeling.models import TemplateSpectralModel
from .core import Maker

__all__ = ["SafeMaskMaker"]


log = logging.getLogger(__name__)


class SafeMaskMaker(Maker):
    """Make safe data range mask for a given observation.


    .. warning::

         Currently some methods computing a safe energy range ("aeff-default",
         "aeff-max" and "edisp-bias") determine a true energy range and apply
         it to reconstructed energy, effectively neglecting the energy dispersion.

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
        Position at which the `aeff_percent` or `bias_percent` are computed.
    fixed_offset : `~astropy.coordinates.Angle`
        offset, calculated from the pointing position, at which
        the `aeff_percent` or `bias_percent` are computed.
        If neither the position nor fixed_offset is specified,
        it uses the position of the center of the map by default.
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
        fixed_offset=None,
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
        self.fixed_offset = fixed_offset
        self.offset_max = Angle(offset_max)
        if self.position and self.fixed_offset:
            raise ValueError(
                "`position` and `fixed_offset` attributes are mutually exclusive"
            )

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
        if observation is None:
            raise ValueError("Method 'offset-max' requires an observation object.")

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
        if observation is None:
            raise ValueError("Method 'offset-max' requires an observation object.")

        energy_max = observation.aeff.meta.get("HI_THRES", None)

        if energy_max:
            energy_max = energy_max * u.TeV
        else:
            log.warning(
                f"No default upper safe energy threshold defined for obs {observation.obs_id}"
            )

        energy_min = observation.aeff.meta.get("LO_THRES", None)

        if energy_min:
            energy_min = energy_min * u.TeV
        else:
            log.warning(
                f"No default lower safe energy threshold defined for obs {observation.obs_id}"
            )

        return dataset._geom.energy_mask(energy_min=energy_min, energy_max=energy_max)

    def make_mask_energy_aeff_max(self, dataset, observation=None):
        """Make safe energy mask from effective area maximum value.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.
        observation: `~gammapy.data.Observation`
            Observation to compute mask for. It is a mandatory argument when fixed_offset is set.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        geom, exposure = dataset._geom, dataset.exposure

        if self.fixed_offset:
            if observation:
                position = observation.pointing_radec.directional_offset_by(
                    position_angle=0.0 * u.deg, separation=self.fixed_offset
                )
            else:
                raise ValueError(
                    f"observation argument is mandatory with {self.fixed_offset}"
                )

        elif self.position:
            position = self.position
        else:
            position = geom.center_skydir

        aeff = exposure.get_spectrum(position) / exposure.meta["livetime"]
        if not np.any(aeff.data > 0.0):
            log.warning(
                f"Effective area is all zero at [{position.to_string('dms')}]. "
                f"No safe energy band can be defined for the dataset '{dataset.name}': "
                "setting `mask_safe` to all False."
            )
            return Map.from_geom(geom, data=False, dtype="bool")

        model = TemplateSpectralModel.from_region_map(aeff)

        energy_true = model.energy
        energy_min = energy_true[np.where(model.values > 0)[0][0]]
        energy_max = energy_true[-1]

        aeff_thres = (self.aeff_percent / 100) * aeff.quantity.max()
        inversion = model.inverse(
            aeff_thres, energy_min=energy_min, energy_max=energy_max
        )

        if not np.isnan(inversion[0]):
            energy_min = inversion[0]

        return geom.energy_mask(energy_min=energy_min)

    def make_mask_energy_edisp_bias(self, dataset, observation=None):
        """Make safe energy mask from energy dispersion bias.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.
        observation: `~gammapy.data.Observation`
            Observation to compute mask for. It is a mandatory argument when fixed_offset is set.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        edisp, geom = dataset.edisp, dataset._geom
        position = None

        if self.fixed_offset:
            if observation:
                position = observation.pointing_radec.directional_offset_by(
                    position_angle=0 * u.deg, separation=self.fixed_offset
                )
            else:
                raise ValueError(
                    f"{observation} argument is mandatory with {self.fixed_offset}"
                )

        if isinstance(edisp, EDispKernelMap):
            if position:
                edisp = edisp.get_edisp_kernel(position=position)
            else:
                edisp = edisp.get_edisp_kernel(position=self.position)
        else:
            if position:
                e_reco = dataset._geom.axes["energy"].edges
                edisp = edisp.get_edisp_kernel(position=position, energy_axis=e_reco)
            else:
                e_reco = dataset._geom.axes["energy"].edges
                edisp = edisp.get_edisp_kernel(
                    position=self.position, energy_axis=e_reco
                )

        energy_min = edisp.get_bias_energy(self.bias_percent / 100)
        return geom.energy_mask(energy_min=energy_min[0])

    @staticmethod
    def make_mask_energy_bkg_peak(dataset):
        """Make safe energy mask based on the binned background.

        The energy threshold is defined as the lower edge of the energy
        bin with the highest predicted background rate. This is to ensure analysis in
        a region where a  Powerlaw approximation to the background spectrum is valid.
        The is motivated by its use in the HESS DL3
        validation paper: https://arxiv.org/pdf/1910.08088.pdf

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
        background_spectrum = dataset.npred_background().get_spectrum()
        idx = np.argmax(background_spectrum.data, axis=0)
        energy_axis = geom.axes["energy"]
        energy_min = energy_axis.edges[idx]
        return geom.energy_mask(energy_min=energy_min)

    @staticmethod
    def make_mask_bkg_invalid(dataset):
        """Mask non-finite values and zeros values in background maps.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        bkg = dataset.background.data
        mask = np.isfinite(bkg)

        if not dataset.stat_type == "wstat":
            mask &= bkg > 0.0

        return mask

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

        if dataset.mask_safe:
            mask_safe = dataset.mask_safe.data
        else:
            mask_safe = np.ones(dataset._geom.data_shape, dtype=bool)

        if dataset.background is not None:
            # apply it first so only clipped values are removed for "bkg-peak"
            mask_safe &= self.make_mask_bkg_invalid(dataset)

        if "offset-max" in self.methods:
            mask_safe &= self.make_mask_offset_max(dataset, observation)

        if "aeff-default" in self.methods:
            mask_safe &= self.make_mask_energy_aeff_default(dataset, observation)

        if "aeff-max" in self.methods:
            mask_safe &= self.make_mask_energy_aeff_max(dataset, observation)

        if "edisp-bias" in self.methods:
            mask_safe &= self.make_mask_energy_edisp_bias(dataset, observation)

        if "bkg-peak" in self.methods:
            mask_safe &= self.make_mask_energy_bkg_peak(dataset)

        dataset.mask_safe = Map.from_geom(dataset._geom, data=mask_safe, dtype=bool)
        return dataset
