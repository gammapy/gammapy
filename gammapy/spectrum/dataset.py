# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from pathlib import Path
from astropy import units as u
from .utils import SpectrumEvaluator
from ..utils.scripts import make_path
from ..utils.fitting import Dataset, Parameters
from ..stats import wstat, cash
from ..utils.random import get_random_state
from .core import CountsSpectrum, PHACountsSpectrum
from ..data import SpectrumStats
from ..irf import EffectiveAreaTable, EnergyDispersion, IRFStacker

__all__ = ["SpectrumDatasetOnOff", "SpectrumDataset", "SpectrumDatasetOnOffStacker"]


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
    mask_fit : `~numpy.ndarray`
        Mask to apply to the likelihood for fitting.
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    background : `~gammapy.spectrum.CountsSpectrum`
        Background to use for the fit.
    mask_safe : `~numpy.ndarray`
        Mask defining the safe data range.
    """

    def __init__(
        self,
        model=None,
        counts=None,
        livetime=None,
        mask_fit=None,
        aeff=None,
        edisp=None,
        background=None,
        mask_safe=None,
    ):
        if mask_fit is not None and mask_fit.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.counts = counts
        self.livetime = livetime
        self.mask_fit = mask_fit
        self.aeff = aeff
        self.edisp = edisp
        self.background = background
        self.model = model
        self.mask_safe = mask_safe

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
                    e_true=self.counts.energy.edges,
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

    def likelihood(self):
        """Total likelihood given the current model parameters.
        """
        if self.mask_fit is None and self.mask_safe is None:
            stat = self.likelihood_per_bin()
        elif self.mask_fit is None:
            stat = self.likelihood_per_bin()[self.mask_safe]
        elif self.mask_safe is None:
            stat = self.likelihood_per_bin()[self.mask_fit]
        else:
            stat = self.likelihood_per_bin()[self.mask_safe & self.mask_fit]

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
        """Energy range defined by the safe mask"""
        energy = self.counts.energy.edges
        e_lo = energy[:-1][self.mask_safe]
        e_hi = energy[1:][self.mask_safe]
        return u.Quantity([e_lo.min(), e_hi.max()])


class SpectrumDatasetOnOff(Dataset):
    """Compute spectral model fit statistic on a ON OFF Spectrum.


    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Fit model
    counts : `~gammapy.spectrum.PHACountsSpectrum`
        ON Counts spectrum
    counts_off : `~gammapy.spectrum.PHACountsSpectrum`
        OFF Counts spectrum
    livetime : `~astropy.units.Quantity`
        Livetime
    mask_fit : `~numpy.array`
        Mask to apply to the likelihood for fitting.
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    mask_safe : `~numpy.array`
        Mask defining the safe data range.
    """

    def __init__(
        self,
        model=None,
        counts=None,
        counts_off=None,
        livetime=None,
        mask_fit=None,
        aeff=None,
        edisp=None,
        mask_safe=None,
    ):

        self.counts = counts
        self.counts_off = counts_off
        self.livetime = livetime
        self.mask_fit = mask_fit
        self.aeff = aeff
        self.edisp = edisp
        self.model = model

        if mask_safe is None:
            mask_safe = np.logical_not(counts.quality)

        self.mask_safe = mask_safe

    @property
    def livetime(self):
        return self._livetime

    @livetime.setter
    def livetime(self, livetime):
        self._livetime = livetime
        if self.counts is not None:
            self.counts.livetime = livetime
        # TODO : check if off might have different exposure
        if self.counts_off is not None:
            self.counts_off.livetime = livetime

    @property
    def lo_threshold(self):
        """Low energy threshold"""
        return self.counts.lo_threshold

    @lo_threshold.setter
    def lo_threshold(self, threshold):
        self.counts.lo_threshold = threshold
        if self.counts_off is not None:
            self.counts_off.lo_threshold = threshold

    @property
    def hi_threshold(self):
        """High energy threshold"""
        return self.counts.hi_threshold

    @hi_threshold.setter
    def hi_threshold(self, threshold):
        self.counts.hi_threshold = threshold
        if self.counts_off is not None:
            self.counts_off.hi_threshold = threshold

    def reset_thresholds(self):
        """Reset energy thresholds (i.e. declare all energy bins valid)"""
        self.counts.reset_thresholds()
        if self.counts_off is not None:
            self.counts_off.reset_thresholds()

    @property
    def mask_safe(self):
        """The mask defined by the counts PHACountsSpectrum"""
        return np.logical_not(self.counts.quality)

    @mask_safe.setter
    def mask_safe(self, mask):
        if mask is None:
            mask = np.ones_like(self.counts.quality, dtype="bool")
        if mask.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")
        else:
            self.counts.quality = np.logical_not(mask)

    def set_fit_energy_range(self, emin=None, emax=None):
        """Set the energy range for the fit.

        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Minimum energy, default is None (i.e. set to minimal energy)
        emax : `~astropy.units.Quantity`
            Maximum energy, default is None (i.e. set to maximal energy)
        """
        energy = self.counts.energy.edges

        if emin is None:
            mask_lo = np.ones_like(energy, dtype="bool")
        else:
            mask_lo = energy[:-1] >= emin

        if emax is None:
            mask_hi = np.ones_like(energy, dtype="bool")
        else:
            mask_hi = energy[1:] <= emax

        self.mask_fit = mask_lo & mask_hi

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions"""
        return self.counts.backscal / self.counts_off.backscal

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
                    e_true=self.counts.energy.edges,
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
        return self.counts.data.data.shape

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
            n_on=self.counts.data.data,
            n_off=self.counts_off.data.data,
            alpha=self.alpha,
            mu_sig=npred.data.data,
        )
        return np.nan_to_num(on_stat_)

    def likelihood(self):
        """Total likelihood given the current model parameters.
        """
        if self.mask_fit is None and self.mask_safe is None:
            stat = self.likelihood_per_bin()
        elif self.mask_fit is None:
            stat = self.likelihood_per_bin()[self.mask_safe]
        elif self.mask_safe is None:
            stat = self.likelihood_per_bin()[self.mask_fit]
        else:
            stat = self.likelihood_per_bin()[self.mask_safe & self.mask_fit]

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
        raise NotImplementedError(
            "To read from an OGIP fits file use SpectrumDatasetOnOff.from_ogip_files."
        )

    @property
    def energy_range(self):
        """Energy range defined by the safe mask."""
        energy = self.counts.energy.edges
        e_lo = energy[:-1][self.mask_safe]
        e_hi = energy[1:][self.mask_safe]
        return u.Quantity([e_lo.min(), e_hi.max()])

    def _as_counts_spectrum(self, data):
        energy = self.counts.energy.edges
        return CountsSpectrum(data=data, energy_lo=energy[:-1], energy_hi=energy[1:])

    def excess(self):
        """Excess (counts - alpha * counts_off)"""
        excess = self.counts.data.data - self.alpha * self.counts_off.data.data
        return self._as_counts_spectrum(excess)

    def residuals(self):
        """Residuals (npred - excess). """
        residuals = self.npred().data.data - self.excess().data.data
        return self._as_counts_spectrum(residuals)

    def peek(self, figsize=(10, 10)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=figsize)

        ax1.set_title("Counts")
        energy_unit = "TeV"
        if self.counts_off is not None:
            energy = self.counts_off.energy.edges
            data = self.counts_off.data.data * self.alpha
            background_vector = CountsSpectrum(
                data=data, energy_lo=energy[:-1], energy_hi=energy[1:]
            )
            background_vector.plot_hist(
                ax=ax1, label="alpha * n_off", color="darkblue", energy_unit=energy_unit
            )
        self.counts.plot_hist(
            ax=ax1,
            label="n_on",
            color="darkred",
            energy_unit=energy_unit,
            show_energy=(self.hi_threshold, self.lo_threshold),
        )
        ax1.set_xlim(
            0.7 * self.lo_threshold.to_value(energy_unit),
            1.3 * self.hi_threshold.to_value(energy_unit),
        )
        ax1.legend(numpoints=1)

        ax2.set_title("Effective Area")
        e_unit = self.aeff.energy.unit
        self.aeff.plot(ax=ax2, show_energy=(self.hi_threshold, self.lo_threshold))
        ax2.set_xlim(
            0.7 * self.lo_threshold.to_value(e_unit),
            1.3 * self.hi_threshold.to_value(e_unit),
        )

        ax3.axis("off")
        if self.counts_off is not None:
            ax3.text(0, 0.2, "{}".format(self.total_stats_safe_range), fontsize=12)

        ax4.set_title("Energy Dispersion")
        if self.edisp is not None:
            self.edisp.plot_matrix(ax=ax4)

        # TODO: optimize layout
        plt.subplots_adjust(wspace=0.3)

    def plot_fit(self):
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
        return self.counts.energy.unit

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

    def to_ogip_files(self, outdir=None, use_sherpa=False, overwrite=False):
        """Write OGIP files.

        If you want to use the written files with Sherpa you have to set the
        ``use_sherpa`` flag. Then all files will be written in units 'keV' and
        'cm2'.

        Parameters
        ----------
        outdir : `pathlib.Path`
            output directory, default: pwd
        use_sherpa : bool, optional
            Write Sherpa compliant files, default: False
        overwrite : bool
            Overwrite existing files?
        """
        outdir = Path.cwd() if outdir is None else Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        phafile = self.counts.phafile
        bkgfile = self.counts.bkgfile
        arffile = self.counts.arffile
        rmffile = self.counts.rmffile

        self.counts.write(outdir / phafile, overwrite=overwrite, use_sherpa=use_sherpa)
        self.aeff.write(outdir / arffile, overwrite=overwrite, use_sherpa=use_sherpa)

        if self.counts_off is not None:
            self.counts_off.write(
                outdir / bkgfile, overwrite=overwrite, use_sherpa=use_sherpa
            )
        if self.edisp is not None:
            self.edisp.write(
                str(outdir / rmffile), overwrite=overwrite, use_sherpa=use_sherpa
            )

    @classmethod
    def from_ogip_files(cls, filename):
        """Read `~gammapy.spectrum.SpectrumDatasetOnOff` from OGIP files.

        BKG file, ARF, and RMF must be set in the PHA header and be present in
        the same folder.

        Parameters
        ----------
        filename : str
            OGIP PHA file to read
        """
        filename = make_path(filename)
        dirname = filename.parent
        on_vector = PHACountsSpectrum.read(filename)
        rmf, arf, bkg = on_vector.rmffile, on_vector.arffile, on_vector.bkgfile

        try:
            energy_dispersion = EnergyDispersion.read(str(dirname / rmf))
        except IOError:
            # TODO : Add logger and echo warning
            energy_dispersion = None

        try:
            off_vector = PHACountsSpectrum.read(str(dirname / bkg))
        except IOError:
            # TODO : Add logger and echo warning
            off_vector = None

        effective_area = EffectiveAreaTable.read(str(dirname / arf))

        mask = on_vector.quality == 0

        return cls(
            counts=on_vector,
            aeff=effective_area,
            counts_off=off_vector,
            edisp=energy_dispersion,
            livetime=on_vector.livetime,
            mask_safe=mask,
        )

    # TODO : do we keep this or should this become the Dataset name
    # This was imported and adapted from the SpectrumObservation class
    @property
    def obs_id(self):
        """The observation ID of the dataset"""
        return self.counts.obs_id

    # TODO : do we keep SpectrumStats or do we adapt this part of code?
    # This was imported and adapted from the SpectrumObservation class
    @property
    def total_stats(self):
        """Return total `~gammapy.spectrum.SpectrumStats`
        """
        return self.stats_in_range(0, self.counts.energy.nbin - 1)

    @property
    def total_stats_safe_range(self):
        """Return total `~gammapy.spectrum.SpectrumStats` within the tresholds
        """
        safe_bins = self.counts.bins_in_safe_range
        return self.stats_in_range(safe_bins[0], safe_bins[-1])

    def stats_in_range(self, bin_min, bin_max):
        """Compute stats for a range of energy bins.

        Parameters
        ----------
        bin_min, bin_max: int
            Bins to include

        Returns
        -------
        stats : `~gammapy.spectrum.SpectrumStats`
            Stacked stats
        """
        idx = np.arange(bin_min, bin_max + 1)
        stats_list = []

        for ii in idx:
            if self.counts_off is not None:
                n_off = int(self.counts_off.data.data.value[ii])
                a_off = self.counts_off._backscal_array[ii]
            else:
                n_off = 0
                a_off = 1  # avoid zero division error

            stat = SpectrumStats(
                energy_min=self.counts.energy.edges[ii],
                energy_max=self.counts.energy.edges[ii + 1],
                n_on=int(self.counts.data.data.value[ii]),
                n_off=n_off,
                a_on=self.counts._backscal_array[ii],
                a_off=a_off,
                obs_id=self.obs_id,
                livetime=self.livetime,
            )
            stats_list.append(stat)

        stacked_stats = SpectrumStats.stack(stats_list)
        stacked_stats.livetime = self.livetime
        stacked_stats.gamma_rate = stacked_stats.excess / stacked_stats.livetime
        stacked_stats.obs_id = self.counts.obs_id
        stacked_stats.energy_min = self.counts.energy.edges[bin_min]
        stacked_stats.energy_max = self.counts.energy.edges[bin_max + 1]
        return stacked_stats


class SpectrumDatasetOnOffStacker:
    r"""Stack a list of homogeneous datasets.

    The stacking of :math:`j` datasets is implemented as follows.
    :math:`k` and :math:`l` denote a bin in reconstructed and true energy,
    respectively.

    .. math::
        \epsilon_{jk} =\left\{\begin{array}{cl} 1, & \mbox{if
            bin k is inside the energy thresholds}\\ 0, & \mbox{otherwise} \end{array}\right.

        \overline{\mathrm{n_{on}}}_k = \sum_{j} \mathrm{n_{on}}_{jk} \cdot
            \epsilon_{jk}

        \overline{\mathrm{n_{off}}}_k = \sum_{j} \mathrm{n_{off}}_{jk} \cdot
            \epsilon_{jk}

        \overline{\alpha}_k =
        \frac{\overline{{b_{on}}}_k}{\overline{{b_{off}}}_k}

        \overline{{b}_{on}}_k = 1

        \overline{{b}_{off}}_k = \frac{1}{\sum_{j}\alpha_{jk} \cdot
            \mathrm{n_{off}}_{jk} \cdot \epsilon_{jk}} \cdot \overline{\mathrm {n_{off}}}

    Please refer to the `~gammapy.irf.IRFStacker` for the description
    of how the IRFs are stacked.

    Parameters
    ----------
    obs_list : list of `~gammapy.spectrum.SpectrumDatasetOnOff`
        Observations to stack

    Examples
    --------
    >>> from gammapy.spectrum import SpectrumDatasetOnOff, SpectrumDatasetOnOffStacker
    >>> obs_ids = [23523, 23526, 23559, 23592]
    >>> datasets = []
    >>> for obs in obs_ids:
    >>>     filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{}.fits"
    >>>     ds = SpectrumDatasetOnOff.from_ogip_files(filename.format(obs))
    >>>     datasets.append(ds)
    >>> obs_stacker = SpectrumDatasetOnOffStacker(datasets)
    >>> stacked = obs_stacker.run()
    >>> print(stacked.livetime)
    6313.8116406202325 s
   """

    def __init__(self, obs_list):
        self.obs_list = obs_list
        self.stacked_on_vector = None
        self.stacked_off_vector = None
        self.stacked_aeff = None
        self.stacked_edisp = None
        self.stacked_bkscal_on = None
        self.stacked_bkscal_off = None
        self.stacked_obs = None

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n{}".format(self.obs_list)
        return ss

    def run(self):
        """Run all steps in the correct order."""
        self.stack_counts_vectors()
        self.stack_aeff()
        self.stack_edisp()
        self.stack_obs()
        return self.stacked_obs

    def stack_counts_vectors(self):
        """Stack on and off vectors."""
        self.stack_on_vector()
        self.stack_off_vector()
        self.stack_backscal()
        self.setup_counts_vectors()

    def stack_on_vector(self):
        """Stack the on count vector."""
        on_vector_list = [o.counts for o in self.obs_list]
        self.stacked_on_vector = self.stack_counts_spectrum(on_vector_list)

    def stack_off_vector(self):
        """Stack the off count vector."""
        off_vector_list = [o.counts_off for o in self.obs_list]
        self.stacked_off_vector = self.stack_counts_spectrum(off_vector_list)

    @staticmethod
    def stack_counts_spectrum(counts_spectrum_list):
        """Stack `~gammapy.spectrum.PHACountsSpectrum`.

        * Bins outside the safe energy range are set to 0
        * Attributes are set to None.
        * The quality vector of the observations are combined with a logical or,
          such that the low (high) threshold of the stacked obs is the minimum
          low (maximum high) threshold of the observation list to be stacked.
        """
        template = counts_spectrum_list[0].copy()
        energy = template.energy
        stacked_data = np.zeros(energy.nbin)
        stacked_quality = np.ones(energy.nbin)
        for spec in counts_spectrum_list:
            stacked_data += spec.counts_in_safe_range.value
            temp = np.logical_and(stacked_quality, spec.quality)
            stacked_quality = np.array(temp, dtype=int)

        return PHACountsSpectrum(
            data=stacked_data,
            energy_lo=energy.edges[:-1],
            energy_hi=energy.edges[1:],
            quality=stacked_quality,
        )

    def stack_backscal(self):
        """Stack ``backscal`` for on and off vector."""
        nbins = self.obs_list[0].counts.energy.nbin
        bkscal_on = np.ones(nbins)
        bkscal_off = np.zeros(nbins)

        alpha_sum = 0.0

        for obs in self.obs_list:
            bkscal_on_data = obs.counts._backscal_array.copy()
            bkscal_off_data = obs.counts_off._backscal_array.copy()
            bkscal_off += (
                bkscal_on_data / bkscal_off_data
            ) * obs.counts_off.counts_in_safe_range.value
            alpha_sum += (obs.alpha * obs.counts_off.counts_in_safe_range).sum()

        with np.errstate(divide="ignore", invalid="ignore"):
            stacked_bkscal_off = self.stacked_off_vector.data.data.value / bkscal_off
            alpha_average = (
                alpha_sum / self.stacked_off_vector.counts_in_safe_range.sum()
            )

        # there should be no nan values in backscal_on or backscal_off
        # this leads to problems when fitting the data
        # use 1 for backscale of on_vector and 1 / alpha_average for backscale of off_vector
        alpha_correction = 1
        idx = np.where(self.stacked_off_vector.data.data == 0)[0]
        bkscal_on[idx] = alpha_correction
        # For the bins where the stacked OFF counts equal 0, the alpha value is performed by weighting on the total
        # OFF counts of each run
        stacked_bkscal_off[idx] = alpha_correction / alpha_average

        self.stacked_bkscal_on = bkscal_on
        self.stacked_bkscal_off = stacked_bkscal_off

    def setup_counts_vectors(self):
        """Add correct attributes to stacked counts vectors."""
        livetimes = [obs.livetime.to_value("s") for obs in self.obs_list]
        self.total_livetime = u.Quantity(np.sum(livetimes), "s")

        self.stacked_on_vector.livetime = self.total_livetime
        self.stacked_off_vector.livetime = self.total_livetime
        self.stacked_on_vector.backscal = self.stacked_bkscal_on
        self.stacked_off_vector.backscal = self.stacked_bkscal_off
        self.stacked_on_vector.obs_id = [obs.obs_id for obs in self.obs_list]
        self.stacked_off_vector.obs_id = [obs.obs_id for obs in self.obs_list]

    def stack_aeff(self):
        """Stack effective areas (weighted by livetime).

        Calls `gammapy.irf.IRFStacker.stack_aeff`.
        """
        irf_stacker = IRFStacker(
            list_aeff=[obs.aeff for obs in self.obs_list],
            list_livetime=[obs.livetime for obs in self.obs_list],
        )
        irf_stacker.stack_aeff()
        self.stacked_aeff = irf_stacker.stacked_aeff

    def stack_edisp(self):
        """Stack energy dispersion (weighted by exposure).

        Calls `~gammapy.irf.IRFStacker.stack_edisp`
        """
        irf_stacker = IRFStacker(
            list_aeff=[obs.aeff for obs in self.obs_list],
            list_livetime=[obs.livetime for obs in self.obs_list],
            list_edisp=[obs.edisp for obs in self.obs_list],
            list_low_threshold=[obs.lo_threshold for obs in self.obs_list],
            list_high_threshold=[obs.hi_threshold for obs in self.obs_list],
        )
        irf_stacker.stack_edisp()
        self.stacked_edisp = irf_stacker.stacked_edisp

    def stack_obs(self):
        """Create stacked `~gammapy.spectrum.SpectrumDatasetOnOff`"""
        self.stacked_obs = SpectrumDatasetOnOff(
            counts=self.stacked_on_vector,
            counts_off=self.stacked_off_vector,
            aeff=self.stacked_aeff,
            edisp=self.stacked_edisp,
            livetime=self.total_livetime,
        )
