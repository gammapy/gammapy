# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.data import GTI
from gammapy.irf import EDispKernel, EffectiveAreaTable, IRFStacker
from gammapy.modeling import Dataset, Parameters
from gammapy.modeling.models import Models, SkyModel
from gammapy.stats import cash, significance, significance_on_off, wstat
from gammapy.utils.fits import energy_axis_to_ebounds
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_name, make_path
from .core import CountsSpectrum, SpectrumEvaluator

__all__ = [
    "SpectrumDatasetOnOff",
    "SpectrumDataset",
    "plot_spectrum_datasets_off_regions",
]


class SpectrumDataset(Dataset):
    """Spectrum dataset for likelihood fitting.

    The spectrum dataset bundles reduced counts data, with a spectral model,
    background model and instrument response function to compute the fit-statistic
    given the current model and data.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Fit model
    counts : `~gammapy.spectrum.CountsSpectrum`
        Counts spectrum
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective area
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion
    background : `~gammapy.spectrum.CountsSpectrum`
        Background to use for the fit.
    mask_safe : `~numpy.ndarray`
        Mask defining the safe data range.
    mask_fit : `~numpy.ndarray`
        Mask to apply to the likelihood for fitting.
    name : str
        Dataset name.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation

    See Also
    --------
    SpectrumDatasetOnOff, FluxPointsDataset, gammapy.cube.MapDataset
    """

    stat_type = "cash"
    tag = "SpectrumDataset"

    def __init__(
        self,
        models=None,
        counts=None,
        livetime=None,
        aeff=None,
        edisp=None,
        background=None,
        mask_safe=None,
        mask_fit=None,
        name=None,
        gti=None,
    ):

        if mask_fit is not None and mask_fit.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.counts = counts

        if livetime is not None:
            livetime = u.Quantity(livetime)

        self.livetime = livetime
        self.mask_fit = mask_fit
        self.aeff = aeff
        self.edisp = edisp
        self.background = background
        self.mask_safe = mask_safe
        self.gti = gti

        self._name = make_name(name)
        self.models = models

    @property
    def name(self):
        return self._name

    def __str__(self):
        str_ = self.__class__.__name__ + "\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"

        str_ += "\t{:32}: {} \n\n".format("Name", self.name)

        counts = np.nan
        if self.counts is not None:
            counts = np.sum(self.counts.data)
        str_ += "\t{:32}: {:.0f} \n".format("Total counts", counts)

        npred = np.nan
        if self.models is not None:
            npred = np.sum(self.npred().data)
        str_ += "\t{:32}: {:.2f}\n".format("Total predicted counts", npred)

        counts_off = np.nan
        if getattr(self, "counts_off", None) is not None:
            counts_off = np.sum(self.counts_off.data)
            str_ += "\t{:32}: {:.2f}\n\n".format("Total off counts", counts_off)

        background = np.nan
        if getattr(self, "background", None) is not None:
            background = np.sum(self.background.data)
            str_ += "\t{:32}: {:.2f}\n\n".format("Total background counts", background)

        aeff_min, aeff_max, aeff_unit = np.nan, np.nan, ""
        if self.aeff is not None:
            try:
                aeff_min = np.min(
                    self.aeff.data.data.value[self.aeff.data.data.value > 0]
                )
            except ValueError:
                aeff_min = 0
            aeff_max = np.max(self.aeff.data.data.value)
            aeff_unit = self.aeff.data.data.unit

        str_ += "\t{:32}: {:.2e} {}\n".format("Effective area min", aeff_min, aeff_unit)
        str_ += "\t{:32}: {:.2e} {}\n\n".format(
            "Effective area max", aeff_max, aeff_unit
        )

        livetime = np.nan
        if self.livetime is not None:
            livetime = self.livetime
        str_ += "\t{:32}: {:.2e}\n\n".format("Livetime", livetime)

        # data section
        n_bins = 0
        if self.counts is not None:
            n_bins = self.counts.data.size
        str_ += "\t{:32}: {} \n".format("Number of total bins", n_bins)

        n_fit_bins = 0
        if self.mask is not None:
            n_fit_bins = np.sum(self.mask)
        str_ += "\t{:32}: {} \n\n".format("Number of fit bins", n_fit_bins)

        # likelihood section
        str_ += "\t{:32}: {}\n".format("Fit statistic type", self.stat_type)

        stat = np.nan
        if self.models is not None and self.counts is not None:
            stat = self.stat_sum()
        str_ += "\t{:32}: {:.2f}\n\n".format("Fit statistic value (-2 log(L))", stat)

        n_pars, n_free_pars = 0, 0
        if self.models is not None:
            n_pars = len(self.models.parameters)
            n_free_pars = len(self.parameters.free_parameters)

        str_ += "\t{:32}: {}\n".format("Number of parameters", n_pars)
        str_ += "\t{:32}: {}\n\n".format("Number of free parameters", n_free_pars)

        if self.models is not None:
            str_ += "\t" + "\n\t".join(str(self.models).split("\n")[2:])

        return str_.expandtabs(tabsize=2)

    @property
    def models(self):
        """Models (`gammapy.modeling.models.Models`)."""
        return self._models

    @models.setter
    def models(self, models):
        if models is None:
            self._models = None
        else:
            if isinstance(models, SkyModel):
                models = [models]
            elif isinstance(models, (Models, list)):
                models = list(models)
            else:
                raise TypeError("Invalid models")
            models_list = [
                model
                for model in models
                if self.name in model.datasets_names or model.datasets_names == "all"
            ]
            self._models = Models(models_list)

        evaluators = []
        if self.models is not None:
            for model in self.models:
                evaluator = SpectrumEvaluator(
                    model=model,
                    livetime=self.livetime,
                    aeff=self.aeff,
                    edisp=self.edisp,
                )
                evaluators.append(evaluator)

        self._evaluators = evaluators

    @property
    def mask_safe(self):
        if self._mask_safe is None:
            return np.ones(self.data_shape, bool)
        else:
            return self._mask_safe

    @mask_safe.setter
    def mask_safe(self, mask):
        self._mask_safe = mask

    @property
    def parameters(self):
        """List of parameters (`~gammapy.modeling.Parameters`)"""
        parameters = []

        for component in self.models:
            parameters.append(component.spectral_model.parameters)

        return Parameters.from_stack(parameters)

    @property
    def _energy_axis(self):
        if self.counts is not None:
            e_axis = self.counts.energy
        elif self.edisp is not None:
            e_axis = self.edisp.data.axis("e_reco")
        elif self.aeff is not None:
            # assume e_reco = e_true
            e_axis = self.aeff.data.axis("energy")
        return e_axis

    @property
    def data_shape(self):
        """Shape of the counts data"""
        return (self._energy_axis.nbin,)

    def npred_sig(self):
        """Predicted counts from source model (`CountsSpectrum`)."""
        data = np.zeros(self.data_shape)
        npred = self._as_counts_spectrum(data)

        for evaluator in self._evaluators:
            npred += evaluator.compute_npred()

        return npred

    def npred(self):
        """Return npred map (model + background)"""
        npred = self.npred_sig()

        if self.background:
            npred += self.background

        return npred

    def stat_array(self):
        """Likelihood per bin given the current model parameters"""
        return cash(n_on=self.counts.data, mu_on=self.npred().data)

    def _as_counts_spectrum(self, data):
        energy = self._energy_axis.edges
        return CountsSpectrum(data=data, energy_lo=energy[:-1], energy_hi=energy[1:])

    @property
    def excess(self):
        """Excess (counts - alpha * counts_off)"""
        excess = self.counts.data - self.background.data
        return self._as_counts_spectrum(excess)

    def fake(self, random_state="random-seed", name=None):
        """Simulate fake counts for the current model and reduced irfs.

        This method overwrites the counts defined on the dataset object.

        Parameters
        ----------
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
        """
        self._name = make_name(name)
        random_state = get_random_state(random_state)
        npred = self.npred()
        npred.data = random_state.poisson(npred.data)
        self.counts = npred

    @property
    def energy_range(self):
        """Energy range defined by the safe mask"""
        energy = self._energy_axis.edges
        e_min, e_max = energy[:-1], energy[1:]

        if self.mask_safe is not None:
            if self.mask_safe.any():
                e_min = e_min[self.mask_safe]
                e_max = e_max[self.mask_safe]
            else:
                return None, None

        return u.Quantity([e_min.min(), e_max.max()])

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

        self.npred_sig().plot(ax=ax, label="mu_src", energy_unit=self._e_unit)
        self.excess.plot(ax=ax, label="Excess", fmt=".", energy_unit=self._e_unit)

        e_min, e_max = self.energy_range
        kwargs = {"color": "black", "linestyle": "dashed"}
        ax.axvline(e_min.to_value(self._e_unit), label="fit range", **kwargs)
        ax.axvline(e_max.to_value(self._e_unit), **kwargs)

        ax.legend(numpoints=1)
        ax.set_title("")
        return ax

    def residuals(self, method="diff"):
        """Compute the spectral residuals.

        Parameters
        ----------
        method : {"diff", "diff/model", "diff/sqrt(model)"}
            Method used to compute the residuals. Available options are:
                - ``diff`` (default): data - model
                - ``diff/model``: (data - model) / model
                - ``diff/sqrt(model)``: (data - model) / sqrt(model)

        Returns
        -------
        residuals : `CountsSpectrum`
            Residual spectrum
        """
        residuals = self._compute_residuals(self.counts, self.npred(), method)
        return residuals

    def plot_residuals(self, method="diff", ax=None, **kwargs):
        """Plot residuals.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.
        method : {"diff", "diff/model", "diff/sqrt(model)"}
            Normalization used to compute the residuals, see `SpectrumDataset.residuals()`
        **kwargs : dict
            Keywords passed to `CountsSpectrum.plot()`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        residuals = self.residuals(method=method)
        label = self._residuals_labels[method]

        residuals.plot(
            ax=ax, ecolor="black", fmt="none", energy_unit=self._e_unit, **kwargs
        )
        ax.axhline(0, color="black", lw=0.5)

        ymax = 1.2 * np.nanmax(residuals.data)
        ax.set_ylim(-ymax, ymax)

        ax.set_xlabel(f"Energy [{self._e_unit}]")
        ax.set_ylabel(f"Residuals ({label})")
        return ax

    @classmethod
    def create(
        cls, e_reco, e_true=None, region=None, reference_time="2000-01-01", name=None
    ):
        """Creates empty spectrum dataset.

        Empty containers are created with the correct geometry.
        counts, background and aeff are zero and edisp is diagonal.

        The safe_mask is set to False in every bin.

        Parameters
        ----------
        e_reco : `~astropy.units.Quantity`
            edges of counts vector
        e_true : `~astropy.units.Quantity`
            edges of effective area table. If not set use reco energy values. Default : None
        region : `~regions.SkyRegion`
            Region to define the dataset for.
        reference_time : `~astropy.time.Time`
            reference time of the dataset, Default is "2000-01-01"
        """
        if e_true is None:
            e_true = e_reco

        counts = CountsSpectrum(e_reco[:-1], e_reco[1:], region=region)
        background = CountsSpectrum(e_reco[:-1], e_reco[1:], region=region)
        aeff = EffectiveAreaTable(
            e_true[:-1], e_true[1:], np.zeros(e_true[:-1].shape) * u.m ** 2
        )
        edisp = EDispKernel.from_diagonal_response(e_true, e_reco)
        mask_safe = np.zeros_like(counts.data, "bool")
        gti = GTI.create(u.Quantity([], "s"), u.Quantity([], "s"), reference_time)
        livetime = gti.time_sum

        return SpectrumDataset(
            counts=counts,
            aeff=aeff,
            edisp=edisp,
            mask_safe=mask_safe,
            background=background,
            livetime=livetime,
            gti=gti,
            name=name,
        )

    def stack(self, other):
        r"""Stack this dataset with another one.

        Safe mask is applied to compute the stacked counts vector.
        Counts outside each dataset safe mask are lost.

        Stacking is performed in-place.

        The stacking of 2 datasets is implemented as follows.
        Here, :math:`k` denotes a bin in reconstructed energy and :math:`j = {1,2}` is the dataset number

        The ``mask_safe`` of each dataset is defined as:

        .. math::
            \epsilon_{jk} =\left\{\begin{array}{cl} 1, &
            \mbox{if bin k is inside the energy thresholds}\\ 0, &
            \mbox{otherwise} \end{array}\right.

        Then the total ``counts`` and model background ``bkg`` are computed according to:

        .. math::
            \overline{\mathrm{n_{on}}}_k =  \mathrm{n_{on}}_{1k} \cdot \epsilon_{1k} +
             \mathrm{n_{on}}_{2k} \cdot \epsilon_{2k}

            \overline{bkg}_k = bkg_{1k} \cdot \epsilon_{1k} +
             bkg_{2k} \cdot \epsilon_{2k}

        The stacked ``safe_mask`` is then:

        .. math::
            \overline{\epsilon_k} = \epsilon_{1k} OR \epsilon_{2k}

        Please refer to the `~gammapy.irf.IRFStacker` for the description
        of how the IRFs are stacked.

        Parameters
        ----------
        other : `~gammapy.spectrum.SpectrumDataset`
            the dataset to stack to the current one
        """
        if not isinstance(other, SpectrumDataset):
            raise TypeError("Incompatible types for SpectrumDataset stacking")

        if self.counts is not None:
            self.counts.data[~self.mask_safe] = 0
            self.counts.data[other.mask_safe] += other.counts.data[other.mask_safe]

        if self.background is not None:
            self.background.data[~self.mask_safe] = 0
            self.background.data[other.mask_safe] += other.background.data[
                other.mask_safe
            ]

        if self.aeff is not None:
            if self.livetime is None or other.livetime is None:
                raise ValueError("IRF stacking requires livetime for both datasets.")

            irf_stacker = IRFStacker(
                list_aeff=[self.aeff, other.aeff],
                list_livetime=[self.livetime, other.livetime],
                list_edisp=[self.edisp, other.edisp],
                list_low_threshold=[self.energy_range[0], other.energy_range[0]],
                list_high_threshold=[self.energy_range[1], other.energy_range[1]],
            )
            irf_stacker.stack_aeff()
            if self.edisp is not None:
                irf_stacker.stack_edisp()
                self.edisp = irf_stacker.stacked_edisp
            self.aeff = irf_stacker.stacked_aeff

        self.mask_safe = np.logical_or(self.mask_safe, other.mask_safe)

        if self.gti is not None:
            self.gti = self.gti.stack(other.gti).union()

        # TODO: for the moment, since dead time is not accounted for, livetime cannot be the sum of GTIs
        if self.livetime is not None:
            self.livetime += other.livetime

    def peek(self, figsize=(16, 4)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        e_min, e_max = self.energy_range

        _, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        ax1.set_title("Counts")
        energy_unit = "TeV"

        if isinstance(self, SpectrumDatasetOnOff) and self.counts_off is not None:
            self.background.plot_hist(
                ax=ax1, label="alpha * N_off", color="darkblue", energy_unit=energy_unit
            )
        elif self.background is not None:
            self.background.plot_hist(
                ax=ax1, label="background", color="darkblue", energy_unit=energy_unit
            )

        self.counts.plot_hist(
            ax=ax1,
            label="n_on",
            color="darkred",
            energy_unit=energy_unit,
            show_energy=(e_min, e_max),
        )

        ax1.set_xlim(
            0.7 * e_min.to_value(energy_unit), 1.3 * e_max.to_value(energy_unit)
        )
        ax1.legend(numpoints=1)

        ax2.set_title("Effective Area")
        e_unit = self.aeff.energy.unit
        self.aeff.plot(ax=ax2, show_energy=(e_min, e_max))
        ax2.set_xlim(0.7 * e_min.to_value(e_unit), 1.3 * e_max.to_value(e_unit))

        ax3.set_title("Energy Dispersion")
        if self.edisp is not None:
            self.edisp.plot_matrix(ax=ax3)

        # TODO: optimize layout
        plt.subplots_adjust(wspace=0.3)

    def info_dict(self, in_safe_energy_range=True):
        """Info dict with summary statistics, summed over energy

        Parameters
        ----------
        in_safe_energy_range : bool
            Whether to sum only in the safe energy range

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        info = dict()
        mask = self.mask_safe if in_safe_energy_range else slice(None)

        info["name"] = self.name
        info["livetime"] = self.livetime.copy()

        info["n_on"] = self.counts.data[mask].sum()

        info["background"] = self.background.data[mask].sum()
        info["excess"] = self.excess.data[mask].sum()
        info["significance"] = significance(
            self.counts.data[mask].sum(), self.background.data[mask].sum(),
        )

        info["background_rate"] = info["background"] / info["livetime"]
        info["gamma_rate"] = info["excess"] / info["livetime"]
        return info


class SpectrumDatasetOnOff(SpectrumDataset):
    """Spectrum dataset for on-off likelihood fitting.

    The on-off spectrum dataset bundles reduced counts data, off counts data,
    with a spectral model, relative background efficiency and instrument
    response functions to compute the fit-statistic given the current model
    and data.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Fit model
    counts : `~gammapy.spectrum.CountsSpectrum`
        ON Counts spectrum
    counts_off : `~gammapy.spectrum.CountsSpectrum`
        OFF Counts spectrum
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective area
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion
    mask_safe : `~numpy.array`
        Mask defining the safe data range.
    mask_fit : `~numpy.array`
        Mask to apply to the likelihood for fitting.
    acceptance : `~numpy.array` or float
        Relative background efficiency in the on region.
    acceptance_off : `~numpy.array` or float
        Relative background efficiency in the off region.
    name : str
        Name of the dataset.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation

    See Also
    --------
    SpectrumDataset, FluxPointsDataset, MapDataset
    """

    stat_type = "wstat"
    tag = "SpectrumDatasetOnOff"

    def __init__(
        self,
        models=None,
        counts=None,
        counts_off=None,
        livetime=None,
        aeff=None,
        edisp=None,
        mask_safe=None,
        mask_fit=None,
        acceptance=None,
        acceptance_off=None,
        name=None,
        gti=None,
    ):

        self.counts = counts
        self.counts_off = counts_off

        if livetime is not None:
            livetime = u.Quantity(livetime)

        self.livetime = livetime
        self.mask_fit = mask_fit
        self.aeff = aeff
        self.edisp = edisp
        self.mask_safe = mask_safe

        if np.isscalar(acceptance):
            acceptance = np.ones(self.data_shape) * acceptance

        if np.isscalar(acceptance_off):
            acceptance_off = np.ones(self.data_shape) * acceptance_off

        self.acceptance = acceptance
        self.acceptance_off = acceptance_off
        self._name = make_name(name)
        self.gti = gti
        self.models = models

    def __str__(self):
        str_ = super().__str__()

        str_list = str_.split("\n")

        acceptance = np.nan
        if self.acceptance is not None:
            acceptance = np.mean(self.acceptance)

        str_acc = "\t{:32}: {}\n".format("Acceptance mean:", acceptance)
        str_list.insert(16, str_acc)
        str_ = "\n".join(str_list)
        return str_.expandtabs(tabsize=2)

    @property
    def background(self):
        """"""
        background = self.alpha * self.counts_off.data
        return self._as_counts_spectrum(background)

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions"""
        return self.acceptance / self.acceptance_off

    def stat_array(self):
        """Likelihood per bin given the current model parameters"""
        mu_sig = self.npred_sig().data
        on_stat_ = wstat(
            n_on=self.counts.data,
            n_off=self.counts_off.data,
            alpha=self.alpha,
            mu_sig=mu_sig,
        )
        return np.nan_to_num(on_stat_)

    def fake(self, background_model, random_state="random-seed", name=None):
        """Simulate fake counts for the current model and reduced irfs.

        This method overwrites the counts and off counts defined on the dataset object.

        Parameters
        ----------
        background_model : `~gammapy.spectrum.CountsSpectrum`
            BackgroundModel. In the future will be part of the SpectrumDataset Class.
            For the moment, a CountSpectrum.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
        """
        self._name = make_name(name)
        random_state = get_random_state(random_state)

        npred_sig = self.npred_sig()
        npred_sig.data = random_state.poisson(npred_sig.data)

        npred_bkg = background_model.copy()
        npred_bkg.data = random_state.poisson(npred_bkg.data)

        self.counts = npred_sig + npred_bkg

        npred_off = background_model / self.alpha
        npred_off.data = random_state.poisson(npred_off.data)
        self.counts_off = npred_off

    @classmethod
    def create(
        cls, e_reco, e_true=None, region=None, reference_time="2000-01-01", name=None
    ):
        """Create empty SpectrumDatasetOnOff.

        Empty containers are created with the correct geometry.
        counts, counts_off and aeff are zero and edisp is diagonal.

        The safe_mask is set to False in every bin.

        Parameters
        ----------
        e_reco : `~astropy.units.Quantity`
            edges of counts vector
        e_true : `~astropy.units.Quantity`
            edges of effective area table. If not set use reco energy values. Default : None
        region : `~regions.SkyRegion`
            Region to define the dataset for.
        reference_time : `~astropy.time.Time`
            reference time of the dataset, Default is "2000-01-01"
        """
        if e_true is None:
            e_true = e_reco

        counts = CountsSpectrum(e_reco[:-1], e_reco[1:], region=region)
        counts_off = CountsSpectrum(e_reco[:-1], e_reco[1:], region=region)
        aeff = EffectiveAreaTable(
            e_true[:-1], e_true[1:], np.zeros(e_true[:-1].shape) * u.m ** 2
        )
        edisp = EDispKernel.from_diagonal_response(e_true, e_reco)
        mask_safe = np.zeros_like(counts.data, "bool")
        gti = GTI.create(u.Quantity([], "s"), u.Quantity([], "s"), reference_time)
        livetime = gti.time_sum
        acceptance = np.ones_like(counts.data, int)
        acceptance_off = np.ones_like(counts.data, int)

        return SpectrumDatasetOnOff(
            counts=counts,
            counts_off=counts_off,
            aeff=aeff,
            edisp=edisp,
            mask_safe=mask_safe,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            livetime=livetime,
            gti=gti,
            name=name,
        )

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

    def _is_stackable(self):
        """Check if the Dataset contains enough information to be stacked"""
        if (
            self.acceptance_off is None
            or self.acceptance is None
            or self.counts_off is None
        ):
            return False
        else:
            return True

    def stack(self, other):
        r"""Stack this dataset with another one.

        Safe mask is applied to compute the stacked counts vector.
        Counts outside each dataset safe mask are lost.

        Stacking is performed in-place.

        The stacking of 2 datasets is implemented as follows.
        Here, :math:`k`  denotes a bin in reconstructed energy and :math:`j = {1,2}` is the dataset number

        The ``mask_safe`` of each dataset is defined as:

        .. math::
            \epsilon_{jk} =\left\{\begin{array}{cl} 1, &
            \mbox{if k is inside the energy thresholds}\\ 0, &
            \mbox{otherwise} \end{array}\right.

        Then the total ``counts`` and ``counts_off`` are computed according to:

        .. math::
            \overline{\mathrm{n_{on}}}_k =  \mathrm{n_{on}}_{1k} \cdot \epsilon_{1k} +
            \mathrm{n_{on}}_{2k} \cdot \epsilon_{2k}

            \overline{\mathrm{n_{off}}}_k = \mathrm{n_{off}}_{1k} \cdot \epsilon_{1k} +
            \mathrm{n_{off}}_{2k} \cdot \epsilon_{2k}

        The stacked ``safe_mask`` is then:

        .. math::
            \overline{\epsilon_k} = \epsilon_{1k} OR \epsilon_{2k}

        In each energy bin :math:`k`, the count excess is computed taking into account the ON ``acceptance``,
        :math:`a_{on}_k` and the OFF one: ``acceptance_off``, :math:`a_{off}_k`. They define
        the :math:`\alpha_k=a_{on}_k/a_{off}_k` factors such that :math:`n_{ex}_k = n_{on}_k - \alpha_k n_{off}_k`.
        We define the stacked value of :math:`\overline{{a}_{on}}_k = 1` so that:

        .. math::
            \overline{{a}_{off}}_k = \frac{\overline{\mathrm {n_{off}}}}{\alpha_{1k} \cdot
            \mathrm{n_{off}}_{1k} \cdot \epsilon_{1k} + \alpha_{2k} \cdot \mathrm{n_{off}}_{2k} \cdot \epsilon_{2k}}

        Please refer to the `~gammapy.irf.IRFStacker` for the description
        of how the IRFs are stacked.

        Parameters
        ----------
        other : `~gammapy.spectrum.SpectrumDatasetOnOff`
            the dataset to stack to the current one

        Examples
        --------
        >>> from gammapy.spectrum import SpectrumDatasetOnOff
        >>> obs_ids = [23523, 23526, 23559, 23592]
        >>> datasets = []
        >>> for obs in obs_ids:
        >>>     filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{}.fits"
        >>>     ds = SpectrumDatasetOnOff.from_ogip_files(filename.format(obs))
        >>>     datasets.append(ds)
        >>> stacked = datasets[0]
        >>> for ds in datasets[1:]:
        >>>     stacked.stack(ds)
        >>> print(stacked.livetime)
        6313.8116406202325 s
        """
        if not isinstance(other, SpectrumDatasetOnOff):
            raise TypeError("Incompatible types for SpectrumDatasetOnOff stacking")

        # We assume here that counts_off, acceptance and acceptance_off are well defined.
        if not self._is_stackable() or not other._is_stackable():
            raise ValueError("Cannot stack incomplete SpectrumDatsetOnOff.")

        total_off = np.zeros_like(self.counts_off.data, dtype=float)
        total_alpha = np.zeros_like(self.counts_off.data, dtype=float)

        total_off[self.mask_safe] += self.counts_off.data[self.mask_safe]
        total_off[other.mask_safe] += other.counts_off.data[other.mask_safe]
        total_alpha[self.mask_safe] += (self.alpha * self.counts_off)[self.mask_safe]
        total_alpha[other.mask_safe] += (other.alpha * other.counts_off)[
            other.mask_safe
        ]

        with np.errstate(divide="ignore", invalid="ignore"):
            acceptance_off = total_off / total_alpha
            average_alpha = total_alpha.sum() / total_off.sum()

        # For the bins where the stacked OFF counts equal 0, the alpha value is performed by weighting on the total
        # OFF counts of each run
        is_zero = total_off == 0
        acceptance_off[is_zero] = 1 / average_alpha

        self.acceptance = np.ones_like(self.counts_off.data, dtype=float)
        self.acceptance_off = acceptance_off

        if self.counts_off is not None:
            self.counts_off.data[~self.mask_safe] = 0
            self.counts_off.data[other.mask_safe] += other.counts_off.data[
                other.mask_safe
            ]

        super().stack(other)

    def to_ogip_files(self, outdir=None, use_sherpa=False, overwrite=False):
        """Write OGIP files.

        If you want to use the written files with Sherpa you have to set the
        ``use_sherpa`` flag. Then all files will be written in units 'keV' and
        'cm2'.

        The naming scheme is fixed, with {name} the dataset name:

        * PHA file is named pha_obs{name}.fits
        * BKG file is named bkg_obs{name}.fits
        * ARF file is named arf_obs{name}.fits
        * RMF file is named rmf_obs{name}.fits

        Parameters
        ----------
        outdir : `pathlib.Path`
            output directory, default: pwd
        use_sherpa : bool, optional
            Write Sherpa compliant files, default: False
        overwrite : bool
            Overwrite existing files?
        """
        # TODO: refactor and reduce amount of code duplication
        outdir = Path.cwd() if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        phafile = f"pha_obs{self.name}.fits"

        bkgfile = phafile.replace("pha", "bkg")
        arffile = phafile.replace("pha", "arf")
        rmffile = phafile.replace("pha", "rmf")

        counts_table = self.counts.to_table()
        counts_table["QUALITY"] = np.logical_not(self.mask_safe)
        counts_table["BACKSCAL"] = self.acceptance
        counts_table["AREASCAL"] = np.ones(self.acceptance.size)
        meta = self._ogip_meta()

        meta["respfile"] = rmffile
        meta["backfile"] = bkgfile
        meta["ancrfile"] = arffile
        meta["hduclas2"] = "TOTAL"
        counts_table.meta = meta

        name = counts_table.meta["name"]
        hdu = fits.BinTableHDU(counts_table, name=name)
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu, self._ebounds_hdu(use_sherpa)])

        if self.gti is not None:
            hdu = fits.BinTableHDU(self.gti.table, name="GTI")
            hdulist.append(hdu)

        if self.counts.region is not None and self.counts.wcs is not None:
            region_table = self.counts._to_region_table()
            region_hdu = fits.BinTableHDU(region_table, name="REGION")
            hdulist.append(region_hdu)

        hdulist.writeto(outdir / phafile, overwrite=overwrite)

        self.aeff.write(outdir / arffile, overwrite=overwrite, use_sherpa=use_sherpa)

        if self.counts_off is not None:
            counts_off_table = self.counts_off.to_table()
            counts_off_table["QUALITY"] = np.logical_not(self.mask_safe)
            counts_off_table["BACKSCAL"] = self.acceptance_off
            counts_off_table["AREASCAL"] = np.ones(self.acceptance.size)
            meta = self._ogip_meta()
            meta["hduclas2"] = "BKG"

            counts_off_table.meta = meta
            name = counts_off_table.meta["name"]
            hdu = fits.BinTableHDU(counts_off_table, name=name)
            hdulist = fits.HDUList(
                [fits.PrimaryHDU(), hdu, self._ebounds_hdu(use_sherpa)]
            )
            if self.counts_off.region is not None and self.counts_off.wcs is not None:
                region_table = self.counts_off._to_region_table()
                region_hdu = fits.BinTableHDU(region_table, name="REGION")
                hdulist.append(region_hdu)

            hdulist.writeto(outdir / bkgfile, overwrite=overwrite)

        if self.edisp is not None:
            self.edisp.write(
                outdir / rmffile, overwrite=overwrite, use_sherpa=use_sherpa
            )

    def _ebounds_hdu(self, use_sherpa):
        energy = self.counts.energy.edges

        if use_sherpa:
            energy = energy.to("keV")

        return energy_axis_to_ebounds(energy)

    def _ogip_meta(self):
        """Meta info for the OGIP data format"""
        return {
            "name": "SPECTRUM",
            "hduclass": "OGIP",
            "hduclas1": "SPECTRUM",
            "corrscal": "",
            "chantype": "PHA",
            "detchans": self.counts.energy.nbin,
            "filter": "None",
            "corrfile": "",
            "poisserr": True,
            "hduclas3": "COUNT",
            "hduclas4": "TYPE:1",
            "lo_thres": self.energy_range[0].to_value("TeV"),
            "hi_thres": self.energy_range[1].to_value("TeV"),
            "exposure": self.livetime.to_value("s"),
            "obs_id": self.name,
        }

    @classmethod
    def from_ogip_files(cls, filename):
        """Read `~gammapy.spectrum.SpectrumDatasetOnOff` from OGIP files.

        BKG file, ARF, and RMF must be set in the PHA header and be present in
        the same folder.

        The naming scheme is fixed to the following scheme:

        * PHA file is named ``pha_obs{name}.fits``
        * BKG file is named ``bkg_obs{name}.fits``
        * ARF file is named ``arf_obs{name}.fits``
        * RMF file is named ``rmf_obs{name}.fits``
          with ``{name}`` the dataset name.

        Parameters
        ----------
        filename : str
            OGIP PHA file to read
        """
        filename = make_path(filename)
        dirname = filename.parent

        with fits.open(filename, memmap=False) as hdulist:
            data = _read_ogip_hdulist(hdulist)

        counts = CountsSpectrum(
            energy_hi=data["energy_hi"],
            energy_lo=data["energy_lo"],
            data=data["data"],
            region=data["region"],
            wcs=data["wcs"],
        )

        phafile = filename.name

        try:
            rmffile = phafile.replace("pha", "rmf")
            energy_dispersion = EDispKernel.read(dirname / rmffile)
        except OSError:
            # TODO : Add logger and echo warning
            energy_dispersion = None

        try:
            bkgfile = phafile.replace("pha", "bkg")
            with fits.open(dirname / bkgfile, memmap=False) as hdulist:
                data_bkg = _read_ogip_hdulist(hdulist)
                counts_off = CountsSpectrum(
                    energy_hi=data_bkg["energy_hi"],
                    energy_lo=data_bkg["energy_lo"],
                    data=data_bkg["data"],
                    region=data_bkg["region"],
                    wcs=data_bkg["wcs"],
                )

                acceptance_off = data_bkg["backscal"]
        except OSError:
            # TODO : Add logger and echo warning
            counts_off, acceptance_off = None, None

        arffile = phafile.replace("pha", "arf")
        aeff = EffectiveAreaTable.read(dirname / arffile)

        mask_safe = np.logical_not(data["quality"])

        return cls(
            counts=counts,
            aeff=aeff,
            counts_off=counts_off,
            edisp=energy_dispersion,
            livetime=data["livetime"],
            mask_safe=mask_safe,
            acceptance=data["backscal"],
            acceptance_off=acceptance_off,
            name=str(data["obs_id"]),
            gti=data["gti"],
        )

    def info_dict(self, in_safe_energy_range=True):
        """Info dict with summary statistics, summed over energy

        Parameters
        ----------
        in_safe_energy_range : bool
            Whether to sum only in the safe energy range

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        info = super().info_dict(in_safe_energy_range)
        mask = self.mask_safe if in_safe_energy_range else slice(None)

        # TODO: handle energy dependent a_on / a_off
        info["a_on"] = self.acceptance[0].copy()

        if self.counts_off is not None:
            info["n_off"] = self.counts_off.data[mask].sum()
            info["a_off"] = self.acceptance_off[0].copy()
        else:
            info["n_off"] = 0
            info["a_off"] = 1

        info["alpha"] = self.alpha[0].copy()
        info["significance"] = significance_on_off(
            self.counts.data[mask].sum(),
            self.counts_off.data[mask].sum(),
            self.alpha[0],
        )

        return info

    def to_dict(self, filename, *args, **kwargs):
        """Convert to dict for YAML serialization."""
        outdir = Path(filename).parent
        filename = str(outdir / f"pha_obs{self.name}.fits")

        return {
            "name": self.name,
            "type": self.tag,
            "filename": filename,
        }

    def write(self, filename, overwrite):
        """Write spectrum dataset on off to file.

        Currently only the OGIP format is supported

        Parameters
        ----------
        filename : str
            Filename to write to.
        overwrite : bool
            Overwrite existing file.
        """
        outdir = Path(filename).parent
        self.to_ogip_files(outdir=outdir, overwrite=overwrite)

    @classmethod
    def from_dict(cls, data, components, models):
        """Create flux point dataset from dict.

        Parameters
        ----------
        data : dict
            Dict containing data to create dataset from.
        components : list of dict
            Not used.
        models : list of `SkyModel`
            List of model components.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.

        """

        filename = data["filename"]

        dataset = cls.from_ogip_files(filename=filename)
        dataset.mask_fit = None
        dataset.models = models
        return dataset

    @classmethod
    def from_spectrum_dataset(
        cls, dataset, acceptance, acceptance_off, counts_off=None
    ):
        """Create spectrum dataseton off from another dataset.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Spectrum dataset defining counts, edisp, aeff, livetime etc.
        acceptance : `~numpy.array` or float
            Relative background efficiency in the on region.
        acceptance_off : `~numpy.array` or float
            Relative background efficiency in the off region.
        counts_off : `~gammapy.spectrum.CountsSpectrum`
            Off counts spectrum . If the dataset provides a background model,
            and no off counts are defined. The off counts are deferred from
            counts_off / alpha.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.

        """
        if counts_off is None and dataset.background is not None:
            alpha = acceptance / acceptance_off
            counts_off = dataset.background / alpha

        return cls(
            counts=dataset.counts,
            aeff=dataset.aeff,
            counts_off=counts_off,
            edisp=dataset.edisp,
            livetime=dataset.livetime,
            mask_safe=dataset.mask_safe,
            mask_fit=dataset.mask_fit,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            gti=dataset.gti,
            name=dataset.name,
        )


def _read_ogip_hdulist(
    hdulist, hdu1="SPECTRUM", hdu2="EBOUNDS", hdu3="GTI", hdu4="REGION"
):
    """Create from `~astropy.io.fits.HDUList`."""
    counts_table = Table.read(hdulist[hdu1])
    ebounds = Table.read(hdulist[hdu2])
    emin = ebounds["E_MIN"].quantity
    emax = ebounds["E_MAX"].quantity

    if hdu3 in hdulist:
        gti = GTI(Table.read(hdulist[hdu3]))
    else:
        gti = None

    if hdu4 in hdulist:
        region, wcs = CountsSpectrum.read_region_table(hdulist[hdu4])
    else:
        region = None
        wcs = None

    # Check if column are present in the header
    quality = None
    areascal = None
    backscal = None

    if "QUALITY" in counts_table.colnames:
        quality = counts_table["QUALITY"].data
    if "AREASCAL" in counts_table.colnames:
        areascal = counts_table["AREASCAL"].data
    if "BACKSCAL" in counts_table.colnames:
        backscal = counts_table["BACKSCAL"].data

    return dict(
        data=counts_table["COUNTS"],
        backscal=backscal,
        energy_lo=emin,
        energy_hi=emax,
        quality=quality,
        areascal=areascal,
        livetime=counts_table.meta["EXPOSURE"] * u.s,
        obs_id=counts_table.meta["OBS_ID"],
        is_bkg=False,
        gti=gti,
        region=region,
        wcs=wcs,
    )


def plot_spectrum_datasets_off_regions(datasets, ax=None):
    """Plot spectrum datasets of regions.

    Parameters
    ----------
    datasets : list of `SpectrumDatasetOnOff`
        List of spectrum on-off datasets
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ax = plt.gca() or ax

    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = color_cycle.by_key()["color"]
    handles = []

    for color, dataset in zip(colors, datasets):
        kwargs = {"edgecolor": color, "facecolor": "none"}
        dataset.counts_off.plot_region(ax=ax, **kwargs)

        # create proxy artist for the custom legend
        handle = mpatches.Patch(label=dataset.name, **kwargs)
        handles.append(handle)

    plt.legend(handles=handles)
