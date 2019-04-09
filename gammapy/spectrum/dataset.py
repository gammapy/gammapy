# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from .observation import SpectrumObservation
from .utils import SpectrumEvaluator
from ..utils.fitting import Dataset, Parameters
from ..stats import wstat

__all__ = ["SpectrumDatasetOnOff"]


class SpectrumDatasetOnOff(Dataset):
    """Compute spectral model fit statistic on a ON OFF Spectrum.


    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Fit model
    counts_on : `~gammapy.spectrum.PHACountsSpectrum`
        ON Counts spectrum
    counts_off : `~gammapy.spectrum.PHACountsSpectrum`
        OFF Counts spectrum
    livetime : `~astropy.units.Quantity`
        Livetime
    mask : numpy.array
        Mask to apply to the likelihood.
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    """

    def __init__(
        self,
        model=None,
        counts_on=None,
        counts_off=None,
        livetime=None,
        mask=None,
        aeff=None,
        edisp=None,
    ):
        if mask is not None and mask.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.counts_on = counts_on
        self.counts_off = counts_off
        self.livetime = livetime
        self.mask = mask
        self.aeff = aeff
        self.edisp = edisp

        self.model = model

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions"""
        return self.counts_on.backscal / self.counts_off.backscal

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        if model is not None:
            self._parameters = Parameters(self._model.parameters.parameters)
            if self.edisp is None:
                self._predictor = SpectrumEvaluator(
                    model=self.model,
                    livetime=self.livetime,
                    aeff=self.aeff,
                    e_true=self.counts_on.energy.bins,
                )
            else:
                self._predictor = SpectrumEvaluator(
                    model=self.model,
                    aeff=self.aeff,
                    edisp=self.edisp,
                    livetime=self.livetime,
                )

        else:
            self._parameters = None
            self._predictor = None

    @property
    def parameters(self):
        if self._parameters is None:
            raise AttributeError("No model set for Dataset")
        else:
            return self._parameters

    @property
    def data_shape(self):
        """Shape of the counts data"""
        return self.counts_on.data.data.shape

    def npred(self):
        """Returns npred counts vector """
        if self._predictor is None:
            raise AttributeError("No model set for this Dataset")
        model_npred = self._predictor.compute_npred().data.data
        return model_npred

    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        on_stat_ = wstat(
            n_on=self.counts_on.data.data,
            n_off=self.counts_off.data.data,
            alpha=self.alpha,
            mu_sig=self.npred(),
        )
        return np.nan_to_num(on_stat_)

    def likelihood(self, parameters, mask=None):
        """Total likelihood given the current model parameters.

        Parameters
        ----------
        mask : `~numpy.ndarray`
            Mask to be combined with the dataset mask.
        """
        if self.mask is None and mask is None:
            stat = self.likelihood_per_bin()
        elif self.mask is None:
            stat = self.likelihood_per_bin()[mask]
        elif mask is None:
            stat = self.likelihood_per_bin()[self.mask]
        else:
            stat = self.likelihood_per_bin()[mask & self.mask]

        return np.sum(stat, dtype=np.float64)

    @classmethod
    def read(cls, filename):
        """Read from file

        For now, filename is assumed to the name of a PHA file where BKG file, ARF, and RMF names
        must be set in the PHA header and be present in the same folder

        Parameters
        ----------
        filename : str
            OGIP PHA file to read
        """
        observation = SpectrumObservation.read(filename)
        return SpectrumDatasetOnOff._from_spectrum_observation(observation)

    # TODO: check if SpectrumObservation is needed in the long run
    @classmethod
    def _from_spectrum_observation(cls, observation):
        """Creates a SpectrumDatasetOnOff from a SpectrumObservation object"""

        # Build mask from quality vector
        quality = observation.on_vector.quality
        mask = quality == 0

        return cls(
            counts_on=observation.on_vector,
            aeff=observation.aeff,
            counts_off=observation.off_vector,
            edisp=observation.edisp,
            livetime=observation.livetime,
            mask=mask,
        )
