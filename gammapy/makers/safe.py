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
        energy threshold for method "edisp-bias".
    position : `~astropy.coordinates.SkyCoord`
        Position at which the `aeff_percent` or `bias_percent` are computed.
    fixed_offset : `~astropy.coordinates.Angle`
        Offset, calculated from the pointing position, at which
        the `aeff_percent` or `bias_percent` are computed.
        If neither the position nor fixed_offset is specified,
        it uses the position of the center of the map by default.
    offset_max : str or `~astropy.units.Quantity`
        Maximum offset cut.
    irfs : {"DL4", "DL3"}
        Whether to use reprojected ("DL4") or raw ("DL3") irfs.
        Default is "DL4".
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
        methods=["aeff-default"],
        aeff_percent=10,
        bias_percent=10,
        position=None,
        fixed_offset=None,
        offset_max="3 deg",
        irfs="DL4",
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

        if irfs not in ["DL3", "DL4"]:
            ValueError(
                "Invalid option for irfs: expected 'DL3' or 'DL4', got {irfs} instead."
            )
        self.irfs = irfs

    def make_mask_offset_max(self, dataset, observation):
        """Make maximum offset mask.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.
        observation : `~gammapy.data.Observation`
            Observation to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Maximum offset mask.
        """
        if observation is None:
            raise ValueError("Method 'offset-max' requires an observation object.")

        separation = dataset._geom.separation(
            observation.get_pointing_icrs(observation.tmid)
        )
        return separation < self.offset_max

    @staticmethod
    def make_mask_energy_aeff_default(dataset, observation):
        """Make safe energy mask from aeff default.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.
        observation : `~gammapy.data.Observation`
            Observation to compute mask for.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        if observation is None:
            raise ValueError("Method 'aeff-default' requires an observation object.")

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

    def _get_offset(self, observation):
        offset = self.fixed_offset
        if offset is None:
            if self.position:
                offset = observation.get_pointing_icrs(observation.tmid).separation(
                    self.position
                )
            else:
                offset = 0.0 * u.deg
        return offset

    def _get_position(self, observation, geom):
        if self.fixed_offset is not None and observation is not None:
            pointing = observation.get_pointing_icrs(observation.tmid)
            return pointing.directional_offset_by(
                position_angle=0 * u.deg, separation=self.fixed_offset
            )
        elif self.position is not None:
            return self.position
        else:
            return geom.center_skydir

    def make_mask_energy_aeff_max(self, dataset, observation=None):
        """Make safe energy mask from effective area maximum value.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Dataset to compute mask for.
        observation : `~gammapy.data.Observation`
            Observation to compute mask for. It is a mandatory argument when fixed_offset is set.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """

        if self.fixed_offset is not None and observation is None:
            raise ValueError(
                f"{observation} argument is mandatory with {self.fixed_offset}"
            )

        geom, exposure = dataset._geom, dataset.exposure

        if self.irfs == "DL3":
            offset = self._get_offset(observation)

            values = observation.aeff.evaluate(
                offset=offset, energy_true=observation.aeff.axes["energy_true"].edges
            )
            valid = observation.aeff.axes["energy_true"].edges[
                values > self.aeff_percent * np.max(values) / 100
            ]
            energy_min = np.min(valid)

        else:
            position = self._get_position(observation, geom)

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
        observation : `~gammapy.data.Observation`
            Observation to compute mask for. It is a mandatory argument when fixed_offset is set.

        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """

        if self.fixed_offset is not None and observation is None:
            raise ValueError(
                f"{observation} argument is mandatory with {self.fixed_offset}"
            )

        edisp, geom = dataset.edisp, dataset._geom

        if self.irfs == "DL3":
            offset = self._get_offset(observation)
            edisp = observation.edisp.to_edisp_kernel(offset)
        else:
            kwargs = dict()
            kwargs["position"] = self._get_position(observation, geom)
            if not isinstance(edisp, EDispKernelMap):
                kwargs["energy_axis"] = dataset._geom.axes["energy"]
            edisp = edisp.get_edisp_kernel(**kwargs)
        energy_min = edisp.get_bias_energy(self.bias_percent / 100)[0]
        return geom.energy_mask(energy_min=energy_min)

    def make_mask_energy_bkg_peak(self, dataset, observation=None):
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
        observation: `~gammapy.data.Observation`
            Observation to compute mask for. It is a mandatory argument when DL3 irfs are used.


        Returns
        -------
        mask_safe : `~numpy.ndarray`
            Safe data range mask.
        """
        geom = dataset._geom
        if self.irfs == "DL3":
            bkg = observation.bkg.to_2d()
            background_spectrum = np.ravel(
                bkg.integral(axis_name="offset", offset=bkg.axes["offset"].bounds[1])
            )
            energy_axis = bkg.axes["energy"]
        else:
            background_spectrum = dataset.npred_background().get_spectrum()
            energy_axis = geom.axes["energy"]

        idx = np.argmax(background_spectrum.data, axis=0)
        return geom.energy_mask(energy_min=energy_axis.edges[idx])

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
        observation : `~gammapy.data.Observation`
            Observation to compute mask for.

        Returns
        -------
        dataset : `Dataset`
            Dataset with defined safe range mask.
        """

        if self.irfs == "DL3":
            if observation is None:
                raise ValueError("observation argument is mandatory with DL3 irfs")

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
            mask_safe &= self.make_mask_energy_bkg_peak(dataset, observation)

        dataset.mask_safe = Map.from_geom(dataset._geom, data=mask_safe, dtype=bool)
        return dataset
