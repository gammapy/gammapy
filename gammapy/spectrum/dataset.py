# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from .observation import SpectrumObservation
from .utils import SpectrumEvaluator
from ..utils.fitting import Dataset, Parameters
from ..stats import wstat, cash
from ..utils.random import get_random_state
from .core import CountsSpectrum


__all__ = ["SpectrumDatasetOnOff", "SpectrumDataset"]


class SpectrumDataset(Dataset):
    """Compute spectral model fit statistic on a CountsSpectrum.

    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Fit model
    counts : `~gammapy.spectrum.CountsSpectrum`
        Counts spectrum
    livetime : float
        Livetime
    mask : `~numpy.ndarray`
        Mask to apply to the likelihood.
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    background : `~gammapy.spectrum.CountsSpectrum`
        Background to use for the fit.
    """

    def __init__(
        self,
        model=None,
        counts=None,
        livetime=None,
        mask=None,
        aeff=None,
        edisp=None,
        background=None,
    ):
        if mask is not None and mask.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.counts = counts
        self.livetime = livetime
        self.mask = mask
        self.aeff = aeff
        self.edisp = edisp
        self.background = background
        self.model = model

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
                    e_true=self.counts.energy.edges
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
        return self.counts.data.shape

    def npred(self):
        """Returns npred map (model + background)"""
        npred = self._predictor.compute_npred()
        if self.background:
            npred.data.data += self.background.data.data
        return npred

    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        return cash(n_on=self.counts.data.data, mu_on=self.npred().data.data)

    def likelihood(self, parameters=None, mask=None):
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

    def fake(self, random_state="random-seed"):
        """Simulate a fake `~gammapy.spectrum.CountsSpectrum`.

        Parameters
        ----------
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
                Defines random number generator initialisation.
                Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        spectrum : `~gammapy.spectrum.CountsSpectrum`
            the fake count spectrum
        """
        random_state = get_random_state(random_state)
        data = random_state.poisson(self.npred().data.data)
        energy = self.counts.energy.edges
        return CountsSpectrum(energy[:-1], energy[1:], data)

    @property
    def energy_range(self):
        """Energy range defined by the mask"""
        energy = self.counts.energy.edges
        e_lo = energy[:-1][self.mask]
        e_hi = energy[1:][self.mask]
        return u.Quantity([e_lo.min(), e_hi.max()])


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
                    e_true=self.counts_on.energy.edges
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
        npred = self._predictor.compute_npred()
        return npred

    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        npred = self.npred()
        on_stat_ = wstat(
            n_on=self.counts_on.data.data,
            n_off=self.counts_off.data.data,
            alpha=self.alpha,
            mu_sig=npred.data.data,
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
        return observation.to_spectrum_dataset()

    @property
    def energy_range(self):
        """Energy range defined by the mask"""
        energy = self.counts_on.energy.edges
        e_lo = energy[:-1][self.mask]
        e_hi = energy[1:][self.mask]
        return u.Quantity([e_lo.min(), e_hi.max()])

    def _as_counts_spectrum(self, data):
        energy = self.counts_on.energy.edges
        return CountsSpectrum(data=data, energy_lo=energy[:-1], energy_hi=energy[1:])

    def excess(self):
        """Excess (counts_on - alpha * counts_off)"""
        excess = self.counts_on.data.data - self.alpha * self.counts_off.data.data
        return self._as_counts_spectrum(excess)

    def residuals(self):
        """Residuals (npred - excess). """
        residuals = self.npred().data.data - self.excess().data.data
        return self._as_counts_spectrum(residuals)

    def peek(self):
        """Plot counts and residuals in two panels.

        Calls ``plot_counts`` and ``plot_residuals``.
        """
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt

        gs = GridSpec(7, 1)

        ax_spectrum = plt.subplot(gs[:5, :])
        self.plot_counts(ax=ax_spectrum)

        ax_spectrum.set_xticks([])

        ax_residuals = plt.subplot(gs[5:, :])
        self.plot_residuals(ax=ax_residuals)
        return ax_spectrum, ax_residuals

    @property
    def _e_unit(self):
        return self.counts_on.energy.unit

    def plot_counts(self, ax=None):
        """Plot predicted and detected counts.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        self.npred().plot(ax=ax, label="mu_src", energy_unit=self._e_unit)
        self.excess().plot(ax=ax, label="Excess", fmt=".", energy_unit=self._e_unit)

        e_min, e_max = self.energy_range
        kwargs = {"color": "black", "linestyle": "dashed"}
        ax.axvline(e_min.to_value(self._e_unit), label="fit range", **kwargs)
        ax.axvline(e_max.to_value(self._e_unit), **kwargs)

        ax.legend(numpoints=1)
        ax.set_title("")
        return ax

    def plot_residuals(self, ax=None):
        """Plot residuals.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        residuals = self.residuals()

        residuals.plot(ax=ax, ecolor="black", fmt="none", energy_unit=self._e_unit)
        ax.axhline(0, color="black", lw=0.5)

        ymax = 1.2 * max(residuals.data.data.value)
        ax.set_ylim(-ymax, ymax)

        ax.set_xlabel("Energy [{}]".format(self._e_unit))
        ax.set_ylabel("Residuals")
        return ax
