# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import numpy as np
import logging
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.data import GTI
from gammapy.datasets import Dataset
from gammapy.irf import EDispKernel, EDispKernelMap, EffectiveAreaTable
from gammapy.maps import RegionGeom, RegionNDMap
from gammapy.modeling.models import Models, ProperModels
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic, cash, wstat
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_name, make_path
from gammapy.utils.table import hstack_columns
from gammapy.modeling.models import BackgroundModel
from .map import MapEvaluator
from .utils import get_figure, get_axes

__all__ = ["SpectrumDatasetOnOff", "SpectrumDataset"]

log = logging.getLogger(__name__)


class SpectrumDataset(Dataset):
    """Spectrum dataset for likelihood fitting.

    The spectrum dataset bundles reduced counts data, with a spectral model,
    background model and instrument response function to compute the fit-statistic
    given the current model and data.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Fit model
    counts : `~gammapy.maps.RegionNDMap`
        Counts spectrum
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff : `~gammapy.maps.RegionNDMap`
        Effective area
    edisp : `~gammapy.irf.EDispKernelMap`
        Energy dispersion kernel.
    mask_safe : `~gammapy.maps.RegionNDMap`
        Mask defining the safe data range.
    mask_fit : `~gammapy.maps.RegionNDMap`
        Mask to apply to the likelihood for fitting.
    name : str
        Dataset name.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    meta_table : `~astropy.table.Table`
        Table listing informations on observations used to create the dataset.
        One line per observation for stacked datasets.

    See Also
    --------
    SpectrumDatasetOnOff, FluxPointsDataset, MapDataset
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
        mask_safe=None,
        mask_fit=None,
        name=None,
        gti=None,
        meta_table=None,
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
        self._background_model = None
        self.mask_safe = mask_safe
        self.gti = gti
        self.meta_table = meta_table

        self._name = make_name(name)
        self.models = models

        # TODO: this enforces the exposure on the edisp map, maybe better move
        #  to where the EDispKernelMap is created?
        if edisp is not None:
            self.edisp.exposure_map.data = self.exposure.data

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
        if self.background_model is not None:
            background = np.sum(self.background_model.evaluate().data)
        str_ += "\t{:32}: {:.2f}\n\n".format("Total background counts", background)

        aeff_min, aeff_max, aeff_unit = np.nan, np.nan, ""
        if self.aeff is not None:
            try:
                aeff_min = np.min(self.aeff.data[self.aeff.data > 0])
            except ValueError:
                aeff_min = 0
            aeff_max = np.max(self.aeff.data)
            aeff_unit = self.aeff.unit

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
            n_free_pars = len(self.models.parameters.free_parameters)

        str_ += "\t{:32}: {}\n".format("Number of parameters", n_pars)
        str_ += "\t{:32}: {}\n\n".format("Number of free parameters", n_free_pars)

        if self.models is not None:
            str_ += "\t" + "\n\t".join(str(self.models).split("\n")[2:])

        return str_.expandtabs(tabsize=2)

    @property
    def evaluators(self):
        """Model evaluators"""

        edisp = self._edisp_kernel

        if self.models:
            for model in self.models:
                evaluator = self._evaluators.get(model)

                if evaluator is None:
                    evaluator = MapEvaluator(
                        model=model, exposure=self.exposure, edisp=edisp, gti=self.gti,
                    )
                    self._evaluators[model] = evaluator

        keys = list(self._evaluators.keys())
        for key in keys:
            if key not in self.models:
                del self._evaluators[key]

        return self._evaluators

    @property
    def models(self):
        """Models (`gammapy.modeling.models.Models`)."""
        return ProperModels(self)

    @property
    def background_model(self):
        return self._background_model

    @models.setter
    def models(self, models):
        if models is None:
            self._models = None
        else:
            self._models = Models(models)

        # TODO: clean this up (probably by removing)
        for model in self.models:
            if isinstance(model, BackgroundModel):
                if model.datasets_names is not None:
                    if self.name in model.datasets_names:
                        self._background_model = model
                        break
        else:
            log.warning(f"No background model defined for dataset {self.name}")
        self._evaluators = {}

    @property
    def mask_safe(self):
        if self._mask_safe is None:
            data = np.ones(self._geom.data_shape, dtype=bool)
            return RegionNDMap.from_geom(self._geom, data=data)
        else:
            return self._mask_safe

    @mask_safe.setter
    def mask_safe(self, mask):
        if mask is None or isinstance(mask, RegionNDMap):
            self._mask_safe = mask
        else:
            raise ValueError(f"Must be `RegionNDMap` and not {type(mask)}")

    @property
    def _energy_axis(self):
        if self.counts is not None:
            e_axis = self.counts.geom.get_axis_by_name("energy")
        elif self.edisp is not None:
            e_axis = self.edisp.data.axis("energy")
        elif self.aeff is not None:
            # assume e_reco = e_true
            e_axis = self.aeff.data.axis("energy_true")
        return e_axis

    @property
    def _geom(self):
        """Main analysis geometry"""
        if self.counts is not None:
            return self.counts.geom
        elif self.background_model is not None:
            return self.background_model.map.geom
        elif self.background is not None:
            return self.background.geom
        else:
            raise ValueError("Either 'counts', 'background_model' must be defined.")

    @property
    def data_shape(self):
        """Shape of the counts data"""
        return self._geom.data_shape

    @property
    def _edisp_kernel(self):
        """The edisp kernel stored in the EDispMapKernel"""
        if self.edisp is not None:
            return self.edisp.get_edisp_kernel()

    def npred(self):
        """Predicted counts from source and background model (`RegionNDMap`)."""
        npred_total = RegionNDMap.from_geom(self._geom)

        for evaluator in self.evaluators.values():
            npred = evaluator.compute_npred()
            npred_total.stack(npred)

        return npred_total

    def stat_array(self):
        """Likelihood per bin given the current model parameters"""
        return cash(n_on=self.counts.data, mu_on=self.npred().data)

    @property
    def excess(self):
        return self.counts - self.background_model.evaluate()

    @property
    def exposure(self):
        """Excess (aeff * livetime)"""
        return self.aeff * self.livetime

    def fake(self, random_state="random-seed"):
        """Simulate fake counts for the current model and reduced irfs.

        This method overwrites the counts defined on the dataset object.

        Parameters
        ----------
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
        """
        random_state = get_random_state(random_state)
        npred = self.npred()
        npred.data = random_state.poisson(npred.data)
        self.counts = npred

    @property
    # TODO: make this a method to support different methods?
    def energy_range(self):
        """Energy range defined by the safe mask"""
        energy = self._energy_axis.edges
        e_min, e_max = energy[:-1], energy[1:]

        if self.mask_safe is not None:
            if self.mask_safe.data.any():
                e_min = e_min[self.mask_safe.data[:, 0, 0]]
                e_max = e_max[self.mask_safe.data[:, 0, 0]]
            else:
                return None, None

        return u.Quantity([e_min.min(), e_max.max()])

    def plot_fit(self, fig=None, fig_kwargs=None, ax_kwargs=None, method="diff", **kwargs):
        """Plot spectrum and residuals in two panels.

        Calls `~SpectrumDataset.plot_excess` and `~SpectrumDataset.plot_residuals`.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            Figure to add AxesSubplot on. Overrides ``fig_kwargs``.
        fig_kwargs: dict
            Keyword arguments passed to `~matplotlib.pyplot.figure`.
        ax_kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes`.
        method : {"diff", "diff/model", "diff/sqrt(model)"}
            Normalization used to compute the residuals, see `SpectrumDataset.residuals`.
        **kwargs : dict
            Keyword arguments passed to `gammapy.maps.RegionNDMap.plot`.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object.
        ax_spectrum, ax_residuals : `~matplotlib.axes.AxesSubplot`
            Spectrum and residuals subplots.
        """
        from matplotlib.gridspec import GridSpec

        fig_kwargs = fig_kwargs or {}
        fig_kwargs.setdefault("figsize", (8, 7))
        fig = get_figure(None, fig, fig_kwargs)
        fig.clf()
        ax_kwargs = ax_kwargs or {}

        gs = GridSpec(7, 1)
        ax_spectrum = fig.add_subplot(gs[:5, :], **ax_kwargs)
        self.plot_excess(ax_spectrum)

        ax_spectrum.label_outer()

        ax_residuals = fig.add_subplot(gs[5:, :], sharex=ax_spectrum, **ax_kwargs)
        self.plot_residuals(ax_residuals, method=method, **kwargs)

        label = self._residuals_labels[method]
        ax_residuals.set_ylabel(f"Residuals\n{label}")

        return fig, (ax_spectrum, ax_residuals)

    @property
    def _e_unit(self):
        return self._geom.axes[0].unit

    def _plot_energy_range(self, ax):
        e_min, e_max = self.energy_range
        kwargs = {"color": "black", "linestyle": "dashed"}
        ax.axvline(e_min.to_value(self._e_unit), label="fit range", **kwargs)
        ax.axvline(e_max.to_value(self._e_unit), **kwargs)

    def plot_counts(self, ax=None, fig=None, ax_kwargs=None, fig_kwargs=None):
        """Plot counts and background.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on. Overrides other parameters.
        fig : `~matplotlib.figure.Figure`
            Figure to add AxesSubplot on or get Axes from. Overrides ``fig_kwargs``.
        ax_kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes`.
        fig_kwargs: dict
            Keyword arguments passed to `~matplotlib.pyplot.figure`.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure of the Axes object.
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        ax, fig = get_axes(ax, fig, ax_kwargs, fig_kwargs)

        self.counts.plot_hist(ax, label="n_on")

        if isinstance(self, SpectrumDatasetOnOff) and self.counts_off is not None:
            self.background.plot_hist(ax, label="alpha * n_off")
        elif self.background_model is not None:
            self.background_model.evaluate().plot_hist(ax, label="background")

        self._plot_energy_range(ax)
        e_min, e_max = self.energy_range
        e_unit = self._e_unit
        ax.set_xlim(0.7 * e_min.to_value(e_unit), 1.3 * e_max.to_value(e_unit))

        ax.legend(numpoints=1)
        return fig, ax

    def plot_excess(self, ax=None, fig=None, ax_kwargs=None, fig_kwargs=None):
        """Plot measured and predicted excess.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on. Overrides other parameters.
        fig : `~matplotlib.figure.Figure`
            Figure to add AxesSubplot on or get Axes from. Overrides ``fig_kwargs``.
        ax_kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes`.
        fig_kwargs: dict
            Keyword arguments passed to `~matplotlib.pyplot.figure`.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure of the Axes object.
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        ax, fig = get_axes(ax, fig, ax_kwargs, fig_kwargs)

        self.excess.plot(
            ax=ax,
            label="Measured excess",
            yerr=np.sqrt(np.abs(self.excess.data.flatten())),
        )
        if self.background_model:
            pred_excess = self.npred() - self.background_model.evaluate()
        elif self.background:
            pred_excess = self.npred() - self.background

        pred_excess.plot_hist(ax, label="Predicted excess")
        self._plot_energy_range(ax)

        ax.legend(numpoints=1)
        return fig, ax

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
        residuals : `RegionNDMap`
            Residual spectrum
        """
        residuals = self._compute_residuals(self.counts, self.npred(), method)
        return residuals

    def plot_residuals(
        self, ax=None, fig=None, ax_kwargs=None, fig_kwargs=None, method="diff", **kwargs
    ):
        """Plot residuals.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on. Overrides the next three parameters.
        fig : `~matplotlib.figure.Figure`
            Figure to add AxesSubplot on or get Axes from. Overrides ``fig_kwargs``.
        ax_kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes`.
        fig_kwargs: dict
            Keyword arguments passed to `~matplotlib.pyplot.figure`.
        method : {"diff", "diff/model", "diff/sqrt(model)"}
            Normalization used to compute the residuals, see `SpectrumDataset.residuals`.
        **kwargs : dict
            Keyword arguments passed to `gammapy.maps.RegionNDMap.plot`.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure of the Axes object.
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        ax, fig = get_axes(ax, fig, ax_kwargs, fig_kwargs)

        residuals = self.residuals(method)
        if method == "diff":
            yerr = np.sqrt((self.counts.data + self.npred().data).flatten())
        else:
            yerr = np.ones_like(residuals.data.flatten())

        kwargs.setdefault("color", kwargs.pop("c", "black"))
        residuals.plot(ax, yerr=yerr, **kwargs)
        ax.axhline(0, color=kwargs["color"], lw=0.5)

        label = self._residuals_labels[method]
        ax.set_ylabel(f"Residuals ({label})")
        ax.set_yscale("linear")
        ymin = 1.05 * np.nanmin(residuals.data - yerr)
        ymax = 1.05 * np.nanmax(residuals.data + yerr)
        ax.set_ylim(ymin, ymax)
        return fig, ax

    @classmethod
    def create(
        cls,
        e_reco,
        e_true=None,
        region=None,
        reference_time="2000-01-01",
        name=None,
        meta_table=None,
    ):
        """Creates empty spectrum dataset.

        Empty containers are created with the correct geometry.
        counts, background and aeff are zero and edisp is diagonal.

        The safe_mask is set to False in every bin.

        Parameters
        ----------
        e_reco : `~gammapy.maps.MapAxis`
            counts energy axis. Its name must be "energy".
        e_true : `~gammapy.maps.MapAxis`
            effective area table energy axis. Its name must be "energy-true".
            If not set use reco energy values. Default : None
        region : `~regions.SkyRegion`
            Region to define the dataset for.
        reference_time : `~astropy.time.Time`
            reference time of the dataset, Default is "2000-01-01"
        meta_table : `~astropy.table.Table`
            Table listing informations on observations used to create the dataset.
            One line per observation for stacked datasets.
        """
        if e_true is None:
            e_true = e_reco.copy(name="energy_true")

        if region is None:
            region = "icrs;circle(0, 0, 1)"

        name = make_name(name)
        counts = RegionNDMap.create(region=region, axes=[e_reco])
        background = RegionNDMap.create(region=region, axes=[e_reco])
        models = Models(
            [BackgroundModel(background, name=name + "-bkg", datasets_names=[name])]
        )
        aeff = RegionNDMap.create(region=region, axes=[e_true], unit="cm2")
        edisp = EDispKernelMap.from_diagonal_response(e_reco, e_true, counts.geom)
        mask_safe = RegionNDMap.from_geom(counts.geom, dtype="bool")
        gti = GTI.create(u.Quantity([], "s"), u.Quantity([], "s"), reference_time)
        livetime = gti.time_sum

        return SpectrumDataset(
            counts=counts,
            aeff=aeff,
            edisp=edisp,
            mask_safe=mask_safe,
            livetime=livetime,
            gti=gti,
            models=models,
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

        Parameters
        ----------
        other : `~gammapy.spectrum.SpectrumDataset`
            the dataset to stack to the current one
        """
        if not isinstance(other, SpectrumDataset):
            raise TypeError("Incompatible types for SpectrumDataset stacking")

        if self.counts is not None:
            self.counts *= self.mask_safe
            self.counts.stack(other.counts, weights=other.mask_safe)

        if self.stat_type == "cash":
            if self.background_model and other.background_model:
                self._background_model.map *= self.mask_safe
                self._background_model.stack(other.background_model, other.mask_safe)
                self.models = Models([self.background_model])
        else:
            self.models = None

        if self.livetime is None or other.livetime is None:
            raise ValueError("IRF stacking requires livetime for both datasets.")
        else:
            stacked_exposure = self.exposure
            stacked_exposure.stack(other.exposure)
            self.aeff = stacked_exposure / (self.livetime + other.livetime)

        if self.edisp is not None and other.edisp is not None:
            self.edisp.edisp_map *= self.mask_safe.data
            self.edisp.stack(other.edisp, weights=other.mask_safe)

        if self.mask_safe is not None and other.mask_safe is not None:
            self.mask_safe.stack(other.mask_safe)

        if self.gti is not None:
            self.gti = self.gti.stack(other.gti).union()

        # TODO: for the moment, since dead time is not accounted for, livetime cannot be the sum of GTIs
        if self.livetime is not None:
            self.livetime += other.livetime

        if self.meta_table and other.meta_table:
            self.meta_table = hstack_columns(self.meta_table, other.meta_table)
        elif other.meta_table:
            self.meta_table = other.meta_table.copy()

    def peek(self, fig=None, fig_kwargs=None, ax_kwargs=None):
        """Quick-look summary plots.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            Figure to add AxesSubplot on. Overrides ``fig_kwargs``.
        fig_kwargs: dict
            Keyword arguments passed to `~matplotlib.pyplot.figure`.
        ax_kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes`.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object.
        ax1, ax2, ax3 : `~matplotlib.axes.AxesSubplot`
            Counts, effective area and energy dispersion subplots.
        """
        fig_kwargs = fig_kwargs or {}
        fig_kwargs.setdefault("figsize", (16, 4))
        fig = get_figure(None, fig, fig_kwargs)
        fig.clf()

        ax_kwargs = ax_kwargs or {}
        ax1, ax2, ax3 = fig.subplots(1, 3, **ax_kwargs)

        self.plot_counts(ax1)
        ax1.set_title("Counts")

        self.aeff.plot(ax2)
        self._plot_energy_range(ax2)
        ax2.set_title("Effective Area")
        e_min, e_max = self.energy_range
        e_unit = self.aeff.geom.axes[0].unit
        ax2.set_xlim(0.7 * e_min.to_value(e_unit), 1.3 * e_max.to_value(e_unit))

        if self.edisp is not None:
            self._edisp_kernel.plot_matrix(ax3)
        ax3.set_title("Energy Dispersion")

        # TODO: optimize layout
        fig.subplots_adjust(wspace=0.3)
        return fig, (ax1, ax2, ax3)

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
        mask = self.mask_safe.data if in_safe_energy_range else slice(None)

        info["name"] = self.name
        info["livetime"] = self.livetime.copy()

        info["n_on"] = self.counts.data[mask].sum()

        if self.background_model:
            background = self.background_model.evaluate().data[mask].sum()
        elif self.background:
            background = self.background.data[mask].sum()

        info["background"] = background
        info["excess"] = self.excess.data[mask].sum()
        info["significance"] = CashCountsStatistic(
            self.counts.data[mask].sum(), background
        ).significance

        info["background_rate"] = info["background"] / info["livetime"]
        info["gamma_rate"] = info["excess"] / info["livetime"]
        return info

    def slice_by_idx(self, slices, name=None):
        """Slice sub dataset.

        The slicing only applies to the maps that define the corresponding axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.
        name : str
            Name of the sliced dataset.

        Returns
        -------
        map_out : `Map`
            Sliced map object.
        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name}
        kwargs["livetime"] = self.livetime

        if self.counts is not None:
            kwargs["counts"] = self.counts.slice_by_idx(slices=slices)

        if self.exposure is not None:
            kwargs["aeff"] = self.aeff.slice_by_idx(slices=slices)

        if self.background_model is not None:
            m = self.background_model.evaluate().slice_by_idx(slices=slices)
            bkg_model = BackgroundModel(map=m, datasets_names=[name])
            bkg_model.norm.frozen = True
            kwargs["models"] = bkg_model

        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.slice_by_idx(slices=slices)

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.slice_by_idx(slices=slices)

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.slice_by_idx(slices=slices)

        return self.__class__(**kwargs)


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
    counts : `~gammapy.maps.RegionNDMap`
        ON Counts spectrum
    counts_off : `~gammapy.maps.RegionNDMap`
        OFF Counts spectrum
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff : `~gammapy.maps.RegionNDMap`
        Effective area
    edisp : `~gammapy.irf.EDispKernelMap`
        Energy dispersion kernel
    mask_safe : `~gammapy.maps.RegionNDMap`
        Mask defining the safe data range.
    mask_fit : `~gammapy.maps.RegionNDMap`
        Mask to apply to the likelihood for fitting.
    acceptance : `~gammapy.maps.RegionNDMap` or float
        Relative background efficiency in the on region.
    acceptance_off : `~gammapy.maps.RegionNDMap` or float
        Relative background efficiency in the off region.
    name : str
        Name of the dataset.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    meta_table : `~astropy.table.Table`
        Table listing informations on observations used to create the dataset.
        One line per observation for stacked datasets.

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
        meta_table=None,
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
        self.meta_table = meta_table

        if np.isscalar(acceptance):
            data = np.ones(self._geom.data_shape) * acceptance
            acceptance = RegionNDMap.from_geom(self._geom, data=data)

        self.acceptance = acceptance

        if np.isscalar(acceptance_off):
            data = np.ones(self._geom.data_shape) * acceptance_off
            acceptance_off = RegionNDMap.from_geom(self._geom, data=data)

        self.acceptance_off = acceptance_off

        self._evaluators = {}
        self._name = make_name(name)
        self.gti = gti
        self.models = models
        self._background_model = None

        # TODO: this enforces the exposure on the edisp map, maybe better move
        #  to where the EDispKernelMap is created?
        if edisp is not None:
            self.edisp.exposure_map.data = self.exposure.data

    def __str__(self):
        str_ = super().__str__()

        str_list = str_.split("\n")

        acceptance = np.nan
        if self.acceptance is not None:
            acceptance = np.mean(self.acceptance.data)

        str_acc = "\t{:32}: {}\n".format("Acceptance mean:", acceptance)
        str_list.insert(16, str_acc)
        str_ = "\n".join(str_list)

        acceptance_off = np.nan
        if self.acceptance_off is not None:
            acceptance_off = np.sum(self.acceptance_off.data)
        str_ += "\t{:32}: {:.0f} \n".format("Acceptance off", acceptance_off)

        return str_.expandtabs(tabsize=2)

    @property
    def background(self):
        """"""
        return self.alpha * self.counts_off.data

    @property
    def excess(self):
        """counts - bkg"""
        return self.counts - self.background

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions"""
        return self.acceptance / self.acceptance_off

    @property
    def _geom(self):
        """Main analysis geometry"""
        if self.counts is not None:
            return self.counts.geom
        elif self.counts_off is not None:
            return self.counts_off.geom
        elif self.acceptance is not None:
            return self.acceptance.geom
        elif self.acceptance_off is not None:
            return self.acceptance_off.geom
        else:
            raise ValueError(
                "Either 'counts', 'counts_off', 'acceptance' or 'acceptance_of' must be defined."
            )

    def stat_array(self):
        """Likelihood per bin given the current model parameters"""
        mu_sig = self.npred().data
        on_stat_ = wstat(
            n_on=self.counts.data,
            n_off=self.counts_off.data,
            alpha=self.alpha.data,
            mu_sig=mu_sig,
        )
        return np.nan_to_num(on_stat_)

    def fake(self, background_model, random_state="random-seed"):
        """Simulate fake counts for the current model and reduced irfs.

        This method overwrites the counts and off counts defined on the dataset object.

        Parameters
        ----------
        background_model : `~gammapy.maps.RegionNDMap`
            Background model.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
        """
        random_state = get_random_state(random_state)

        npred = self.npred()
        npred.data = random_state.poisson(npred.data)

        npred_bkg = background_model.evaluate().copy()
        npred_bkg.data = random_state.poisson(npred_bkg.data)

        self.counts = npred + npred_bkg

        npred_off = background_model.evaluate() / self.alpha
        npred_off.data = random_state.poisson(npred_off.data)
        self.counts_off = npred_off

    @classmethod
    def create(
        cls,
        e_reco,
        e_true=None,
        region=None,
        reference_time="2000-01-01",
        name=None,
        meta_table=None,
    ):
        """Create empty SpectrumDatasetOnOff.

        Empty containers are created with the correct geometry.
        counts, counts_off and aeff are zero and edisp is diagonal.

        The safe_mask is set to False in every bin.

        Parameters
        ----------
        e_reco : `~gammapy.maps.MapAxis`
            counts energy axis. Its name must be "energy".
        e_true : `~gammapy.maps.MapAxis`
            effective area table energy axis. Its name must be "energy-true".
            If not set use reco energy values. Default : None
        region : `~regions.SkyRegion`
            Region to define the dataset for.
        reference_time : `~astropy.time.Time`
            reference time of the dataset, Default is "2000-01-01"
        meta_table : `~astropy.table.Table`
            Table listing informations on observations used to create the dataset.
            One line per observation for stacked datasets.
        """
        dataset = super().create(
            e_reco=e_reco,
            e_true=e_true,
            region=region,
            reference_time=reference_time,
            name=name,
        )

        counts_off = dataset.counts.copy()
        acceptance = RegionNDMap.from_geom(counts_off.geom, dtype=int)
        acceptance.data += 1

        acceptance_off = RegionNDMap.from_geom(counts_off.geom, dtype=int)
        acceptance_off.data += 1

        return cls.from_spectrum_dataset(
            dataset=dataset,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            counts_off=counts_off,
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


        The stacking of :math:`j` elements is implemented as follows.  :math:`k`
        and :math:`l` denote a bin in reconstructed and true energy, respectively.

        .. math::
            \epsilon_{jk} =\left\{\begin{array}{cl} 1, & \mbox{if
                bin k is inside the energy thresholds}\\ 0, & \mbox{otherwise} \end{array}\right.

            \overline{t} = \sum_{j} t_i

            \overline{\mathrm{aeff}}_l = \frac{\sum_{j}\mathrm{aeff}_{jl}
                \cdot t_j}{\overline{t}}

            \overline{\mathrm{edisp}}_{kl} = \frac{\sum_{j} \mathrm{edisp}_{jkl}
                \cdot \mathrm{aeff}_{jl} \cdot t_j \cdot \epsilon_{jk}}{\sum_{j} \mathrm{aeff}_{jl}
                \cdot t_j}


        Parameters
        ----------
        other : `~gammapy.spectrum.SpectrumDatasetOnOff`
            the dataset to stack to the current one

        Examples
        --------
        >>> from gammapy.datasets import SpectrumDatasetOnOff
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

        geom = self.counts.geom
        total_off = RegionNDMap.from_geom(geom)
        total_alpha = RegionNDMap.from_geom(geom)

        total_off.stack(self.counts_off, weights=self.mask_safe)
        total_off.stack(other.counts_off, weights=other.mask_safe)

        total_alpha.stack(self.alpha * self.counts_off, weights=self.mask_safe)
        total_alpha.stack(other.alpha * other.counts_off, weights=other.mask_safe)

        with np.errstate(divide="ignore", invalid="ignore"):
            acceptance_off = total_off / total_alpha
            average_alpha = total_alpha.data.sum() / total_off.data.sum()

        # For the bins where the stacked OFF counts equal 0, the alpha value is performed by weighting on the total
        # OFF counts of each run
        is_zero = total_off.data == 0
        acceptance_off.data[is_zero] = 1 / average_alpha

        self.acceptance = RegionNDMap.from_geom(geom)
        self.acceptance.data += 1
        self.acceptance_off = acceptance_off

        if self.counts_off is not None:
            self.counts_off *= self.mask_safe
            self.counts_off.stack(other.counts_off, weights=other.mask_safe)

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
        counts_table["QUALITY"] = np.logical_not(self.mask_safe.data[:, 0, 0])
        counts_table["BACKSCAL"] = self.acceptance.data[:, 0, 0]
        counts_table["AREASCAL"] = np.ones(self.acceptance.data.size)
        meta = self._ogip_meta()

        meta["respfile"] = rmffile
        meta["backfile"] = bkgfile
        meta["ancrfile"] = arffile
        meta["hduclas2"] = "TOTAL"
        counts_table.meta = meta

        name = counts_table.meta["name"]
        hdu = fits.BinTableHDU(counts_table, name=name)

        energy_axis = self.counts.geom.axes[0]

        hdu_format = "ogip-sherpa" if use_sherpa else "ogip"

        hdulist = fits.HDUList(
            [fits.PrimaryHDU(), hdu, energy_axis.to_table_hdu(format=hdu_format)]
        )

        if self.gti is not None:
            hdu = fits.BinTableHDU(self.gti.table, name="GTI")
            hdulist.append(hdu)

        if self.counts.geom._region is not None and self.counts.geom.wcs is not None:
            region_table = self.counts.geom._to_region_table()
            region_hdu = fits.BinTableHDU(region_table, name="REGION")
            hdulist.append(region_hdu)

        hdulist.writeto(str(outdir / phafile), overwrite=overwrite)

        self.aeff.write(
            outdir / arffile,
            overwrite=overwrite,
            format=hdu_format,
            ogip_column="SPECRESP",
        )

        if self.counts_off is not None:
            counts_off_table = self.counts_off.to_table()
            counts_off_table["QUALITY"] = np.logical_not(self.mask_safe.data[:, 0, 0])
            counts_off_table["BACKSCAL"] = self.acceptance_off.data[:, 0, 0]
            counts_off_table["AREASCAL"] = np.ones(self.acceptance.data.size)
            meta = self._ogip_meta()
            meta["hduclas2"] = "BKG"

            counts_off_table.meta = meta
            name = counts_off_table.meta["name"]
            hdu = fits.BinTableHDU(counts_off_table, name=name)
            hdulist = fits.HDUList(
                [fits.PrimaryHDU(), hdu, energy_axis.to_table_hdu(format=hdu_format)]
            )
            if (
                self.counts_off.geom._region is not None
                and self.counts_off.geom.wcs is not None
            ):
                region_table = self.counts_off.geom._to_region_table()
                region_hdu = fits.BinTableHDU(region_table, name="REGION")
                hdulist.append(region_hdu)

            hdulist.writeto(str(outdir / bkgfile), overwrite=overwrite)

        if self.edisp is not None:
            self._edisp_kernel.write(
                outdir / rmffile, overwrite=overwrite, use_sherpa=use_sherpa
            )

    def _ogip_meta(self):
        """Meta info for the OGIP data format"""
        return {
            "name": "SPECTRUM",
            "hduclass": "OGIP",
            "hduclas1": "SPECTRUM",
            "corrscal": "",
            "chantype": "PHA",
            "detchans": self.counts.geom.axes[0].nbin,
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

        with fits.open(str(filename), memmap=False) as hdulist:
            counts = RegionNDMap.from_hdulist(hdulist, format="ogip")
            acceptance = RegionNDMap.from_hdulist(
                hdulist, format="ogip", ogip_column="BACKSCAL"
            )
            if "GTI" in hdulist:
                gti = GTI(Table.read(hdulist["GTI"]))
            else:
                gti = None

            mask_safe = RegionNDMap.from_hdulist(
                hdulist, format="ogip", ogip_column="QUALITY"
            )
            mask_safe.data = np.logical_not(mask_safe.data)

        phafile = filename.name

        try:
            rmffile = phafile.replace("pha", "rmf")
            kernel = EDispKernel.read(dirname / rmffile)
            edisp = EDispKernelMap.from_edisp_kernel(kernel, geom=counts.geom)

        except OSError:
            # TODO : Add logger and echo warning
            edisp = None

        try:
            bkgfile = phafile.replace("pha", "bkg")
            with fits.open(str(dirname / bkgfile), memmap=False) as hdulist:
                counts_off = RegionNDMap.from_hdulist(hdulist, format="ogip")
                acceptance_off = RegionNDMap.from_hdulist(
                    hdulist, ogip_column="BACKSCAL"
                )
        except OSError:
            # TODO : Add logger and echo warning
            counts_off, acceptance_off = None, None

        arffile = phafile.replace("pha", "arf")
        aeff = RegionNDMap.read(dirname / arffile, format="ogip-arf")

        return cls(
            counts=counts,
            aeff=aeff,
            counts_off=counts_off,
            edisp=edisp,
            livetime=counts.meta["EXPOSURE"] * u.s,
            mask_safe=mask_safe,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            name=str(counts.meta["OBS_ID"]),
            gti=gti,
        )

    def info_dict(self, in_safe_energy_range=True, **kwargs):
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
        mask = self.mask_safe.data if in_safe_energy_range else slice(None)

        # TODO: handle energy dependent a_on / a_off
        info["a_on"] = self.acceptance.data[0, 0, 0].copy()

        if self.counts_off is not None:
            info["n_off"] = self.counts_off.data[mask].sum()
            info["a_off"] = self.acceptance_off.data[0, 0, 0].copy()
        else:
            info["n_off"] = 0
            info["a_off"] = 1

        info["alpha"] = self.alpha.data[0, 0, 0].copy()
        info["significance"] = WStatCountsStatistic(
            self.counts.data[mask].sum(),
            self.counts_off.data[mask].sum(),
            self.alpha.data[0, 0, 0],
        ).significance

        return info

    def to_dict(self, filename, *args, **kwargs):
        """Convert to dict for YAML serialization."""
        outdir = Path(filename).parent
        filename = str(outdir / f"pha_obs{self.name}.fits")

        return {"name": self.name, "type": self.tag, "filename": filename}

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
    def from_dict(cls, data, models, **kwargs):
        """Create flux point dataset from dict.

        Parameters
        ----------
        data : dict
            Dict containing data to create dataset from.
        models : list of `SkyModel`
            List of model components.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.

        """

        filename = make_path(data["filename"])
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
        counts_off : `~gammapy.maps.RegionNDMap`
            Off counts spectrum . If the dataset provides a background model,
            and no off counts are defined. The off counts are deferred from
            counts_off / alpha.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.

        """
        if counts_off is None and dataset.background_model is not None:
            alpha = acceptance / acceptance_off
            counts_off = dataset.background_model.evaluate() / alpha

        return cls(
            models=dataset.models,
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
            meta_table=dataset.meta_table,
        )

    def slice_by_idx(self, slices, name=None):
        """Slice sub dataset.

        The slicing only applies to the maps that define the corresponding axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.
        name : str
            Name of the sliced dataset.

        Returns
        -------
        map_out : `Map`
            Sliced map object.
        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name}

        if self.counts is not None:
            kwargs["counts"] = self.counts.slice_by_idx(slices=slices)

        if self.exposure is not None:
            kwargs["aeff"] = self.aeff.slice_by_idx(slices=slices)

        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.slice_by_idx(slices=slices)

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.slice_by_idx(slices=slices)

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.slice_by_idx(slices=slices)

        kwargs["acceptance"] = self.acceptance.slice_by_idx(slices=slices)
        kwargs["acceptance_off"] = self.acceptance_off.slice_by_idx(slices=slices)
        kwargs["counts_off"] = self.counts_off.slice_by_idx(slices=slices)
        kwargs["livetime"] = self.livetime

        return self.__class__(**kwargs)
