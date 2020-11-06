# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.data import GTI
from gammapy.irf import EDispKernel, EDispKernelMap
from gammapy.maps import RegionNDMap
from gammapy.stats import WStatCountsStatistic, cash, get_wstat_mu_bkg, wstat
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_name, make_path
from .map import MapDataset
from .utils import get_axes, get_figure

__all__ = ["SpectrumDatasetOnOff", "SpectrumDataset"]

log = logging.getLogger(__name__)


class SpectrumDataset(MapDataset):
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
    exposure : `~gammapy.maps.RegionNDMap`
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
        exposure=None,
        background=None,
        edisp=None,
        mask_safe=None,
        mask_fit=None,
        name=None,
        gti=None,
        meta_table=None,
    ):

        self._name = make_name(name)
        self._evaluators = {}

        if mask_fit is not None and mask_fit.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.counts = counts
        self.mask_fit = mask_fit
        self.exposure = exposure
        self.edisp = edisp
        self.background = background
        self.mask_safe = mask_safe
        self.gti = gti
        self.meta_table = meta_table
        self.models = models

    @property
    def psf(self):
        return None

    @property
    def mask_safe(self):
        if self._mask_safe is None:
            data = np.ones(self._geom.data_shape, dtype=bool)
            self._mask_safe = RegionNDMap.from_geom(self._geom, data=data)

        return self._mask_safe

    @mask_safe.setter
    def mask_safe(self, mask):
        if mask is None or isinstance(mask, RegionNDMap):
            self._mask_safe = mask
        else:
            raise ValueError(f"Must be `RegionNDMap` and not {type(mask)}")

    def stat_array(self):
        """Likelihood per bin given the current model parameters"""
        return cash(n_on=self.counts.data, mu_on=self.npred().data)

    def stat_sum(self):
        """Total statistic given the current model parameters."""
        stat = self.stat_array()

        if self.mask is not None:
            stat = stat[self.mask.data]

        return np.sum(stat, dtype=np.float64)

    def write(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def to_hdulist(self):
        raise NotImplementedError

    def from_hdulist(self):
        raise NotImplementedError

    def from_dict(self):
        raise NotImplementedError

    # TODO: decide what to about these "useless" methods
    def to_spectrum_dataset(self, *args, **kwargs):
        """Returns self"""
        return self

    def cutout(self, *args, **kwargs):
        """Returns self"""
        return self

    def pad(self, *args, **kwargs):
        """Returns self"""
        return self

    @property
    # TODO: make this a method to support different methods?
    def energy_range(self):
        """Energy range defined by the safe mask"""
        energy = self._geom.axes["energy"].edges
        energy_min, energy_max = energy[:-1], energy[1:]

        if self.mask_safe is not None:
            if self.mask_safe.data.any():
                energy_min = energy_min[self.mask_safe.data[:, 0, 0]]
                energy_max = energy_max[self.mask_safe.data[:, 0, 0]]
            else:
                return None, None

        return u.Quantity([energy_min.min(), energy_max.max()])

    def plot_fit(
        self,
        ax_spectrum=None,
        ax_residuals=None,
        kwargs_spectrum=None,
        kwargs_residuals=None,
    ):
        """Plot spectrum and residuals in two panels.

        Calls `~SpectrumDataset.plot_excess` and `~SpectrumDataset.plot_residuals`.

        Parameters
        ----------
        ax_spectrum : `~matplotlib.axes.Axes`
            Axes to plot spectrum on.
        ax_residuals : `~matplotlib.axes.Axes`
            Axes to plot residuals on.
        kwargs_spectrum : dict
            Keyword arguments passed to `~SpectrumDataset.plot_excess`.
        kwargs_residuals : dict
            Keyword arguments passed to `~SpectrumDataset.plot_residuals`.

        Returns
        -------
        ax_spectrum, ax_residuals : `~matplotlib.axes.Axes`
            Spectrum and residuals plots.
        """
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(7, 1)
        ax_spectrum, ax_residuals = get_axes(
            ax_spectrum,
            ax_residuals,
            8,
            7,
            [gs[:5, :]],
            [gs[5:, :]],
            kwargs2={"sharex": ax_spectrum},
        )
        kwargs_spectrum = kwargs_spectrum or {}
        kwargs_residuals = kwargs_residuals or {}

        self.plot_excess(ax_spectrum, **kwargs_spectrum)
        ax_spectrum.label_outer()

        self.plot_residuals(ax_residuals, **kwargs_residuals)
        method = kwargs_residuals.get("method", "diff")
        label = self._residuals_labels[method]
        ax_residuals.set_ylabel(f"Residuals\n{label}")

        return ax_spectrum, ax_residuals

    @property
    def _energy_unit(self):
        return self._geom.axes[0].unit

    def _plot_energy_range(self, ax):
        energy_min, energy_max = self.energy_range
        kwargs = {"color": "black", "linestyle": "dashed"}
        ax.axvline(energy_min.to_value(self._energy_unit), label="fit range", **kwargs)
        ax.axvline(energy_max.to_value(self._energy_unit), **kwargs)

    def plot_counts(
        self, ax=None, kwargs_counts=None, kwargs_background=None, **kwargs
    ):
        """Plot counts and background.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        kwargs_counts: dict
            Keyword arguments passed to `~matplotlib.axes.Axes.hist` for the counts.
        kwargs_background: dict
            Keyword arguments passed to `~matplotlib.axes.Axes.hist` for the background.
        **kwargs: dict
            Keyword arguments passed to both `~matplotlib.axes.Axes.hist`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        kwargs_counts = kwargs_counts or {}
        kwargs_background = kwargs_background or {}

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_counts)
        plot_kwargs.setdefault("label", "Counts")
        ax = self.counts.plot_hist(ax, **plot_kwargs)

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_background)

        plot_kwargs.setdefault("label", "Background")
        self.background.plot_hist(ax, **plot_kwargs)

        self._plot_energy_range(ax)
        energy_min, energy_max = self.energy_range
        ax.set_xlim(0.7 * energy_min.value, 1.3 * energy_max.value)

        ax.legend(numpoints=1)
        return ax

    def plot_excess(
        self, ax=None, kwargs_excess=None, kwargs_npred_signal=None, **kwargs
    ):
        """Plot excess and predicted signal.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        kwargs_excess: dict
            Keyword arguments passed to `~matplotlib.axes.Axes.errorbar` for
            the excess.
        kwargs_npred_signal : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.hist` for the
            predicted signal.
        **kwargs: dict
            Keyword arguments passed to both plot methods.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        kwargs_excess = kwargs_excess or {}
        kwargs_npred_signal = kwargs_npred_signal or {}

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_excess)
        plot_kwargs.setdefault("label", "Excess counts")
        ax = self.excess.plot(
            ax, yerr=np.sqrt(np.abs(self.excess.data.flatten())), **plot_kwargs
        )

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_npred_signal)
        plot_kwargs.setdefault("label", "Predicted signal counts")
        self.npred_signal().plot_hist(ax, **plot_kwargs)

        self._plot_energy_range(ax)
        ax.legend(numpoints=1)
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
        residuals : `RegionNDMap`
            Residual spectrum
        """
        residuals = self._compute_residuals(self.counts, self.npred(), method)
        return residuals

    def plot_residuals(self, ax=None, method="diff", **kwargs):
        """Plot spectrum residuals.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        method : {"diff", "diff/model", "diff/sqrt(model)"}
            Normalization used to compute the residuals, see `SpectrumDataset.residuals`.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.errorbar`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        # TODO: remove code duplication with `MapDataset.plot_residuals_spectral()`
        residuals = self.residuals(method)
        if method == "diff":
            yerr = np.sqrt((self.counts.data + self.npred().data).flatten())
        else:
            yerr = np.ones_like(residuals.data.flatten())

        kwargs.setdefault("color", kwargs.pop("c", "black"))
        ax = residuals.plot(ax, yerr=yerr, **kwargs)
        ax.axhline(0, color=kwargs["color"], lw=0.5)

        label = self._residuals_labels[method]
        ax.set_ylabel(f"Residuals ({label})")
        ax.set_yscale("linear")
        ymin = 1.05 * np.nanmin(residuals.data - yerr)
        ymax = 1.05 * np.nanmax(residuals.data + yerr)
        ax.set_ylim(ymin, ymax)
        return ax

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
        exposure = RegionNDMap.create(
            region=region, axes=[e_true], unit="cm2 s", meta={"livetime": 0 * u.s}
        )
        edisp = EDispKernelMap.from_diagonal_response(e_reco, e_true, geom=counts.geom)
        mask_safe = RegionNDMap.from_geom(counts.geom, dtype="bool")
        gti = GTI.create(u.Quantity([], "s"), u.Quantity([], "s"), reference_time)

        return SpectrumDataset(
            counts=counts,
            exposure=exposure,
            background=background,
            edisp=edisp,
            mask_safe=mask_safe,
            gti=gti,
            name=name,
        )

    def peek(self, fig=None):
        """Quick-look summary plots.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            Figure to add AxesSubplot on.

        Returns
        -------
        ax1, ax2, ax3 : `~matplotlib.axes.AxesSubplot`
            Counts, effective area and energy dispersion subplots.
        """
        fig = get_figure(fig, 16, 4)
        ax1, ax2, ax3 = fig.subplots(1, 3)

        ax1.set_title("Counts")
        self.plot_counts(ax1)

        ax2.set_title("Exposure")
        self.exposure.plot(ax2)
        self._plot_energy_range(ax2)
        energy_min, energy_max = self.energy_range
        ax2.set_xlim(0.7 * energy_min.value, 1.3 * energy_max.value)

        ax3.set_title("Energy Dispersion")
        if self.edisp is not None:
            kernel = self.edisp.get_edisp_kernel()
            kernel.plot_matrix(ax3, vmin=0, vmax=1)

        # TODO: optimize layout
        fig.subplots_adjust(wspace=0.3)
        return ax1, ax2, ax3


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
    exposure : `~gammapy.maps.RegionNDMap`
        Exposure
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
        exposure=None,
        edisp=None,
        mask_safe=None,
        mask_fit=None,
        acceptance=None,
        acceptance_off=None,
        name=None,
        gti=None,
        meta_table=None,
    ):

        self._name = make_name(name)
        self._evaluators = {}

        self.counts = counts
        self.counts_off = counts_off

        self.mask_fit = mask_fit
        self.exposure = exposure
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

        self.gti = gti
        self.models = models

    def __str__(self):
        str_ = super().__str__()

        str_list = str_.split("\n")

        if getattr(self, "counts_off", None) is not None:
            counts_off = np.sum(self.counts_off.data)
            str_cts = "\t{:32}: {:.2f}".format("Total off counts", counts_off)

        str_list.insert(6, str_cts)

        acceptance = np.nan
        if self.acceptance is not None:
            acceptance = np.mean(self.acceptance.data)

        str_acc = "\n\t{:32}: {:.3f}\n".format("Acceptance mean", acceptance)

        acceptance_off = np.nan
        if self.acceptance_off is not None:
            acceptance_off = np.sum(self.acceptance_off.data)
        str_acc += "\t{:32}: {:.3f}".format("Acceptance off", acceptance_off)

        str_list.insert(16, str_acc)
        str_ = "\n".join(str_list)

        return str_.expandtabs(tabsize=2)

    def npred_background(self):
        """Background counts estimated from the marginalized likelihood estimate.
         See :ref:`wstat`
         """
        mu_bkg = self.alpha.data * get_wstat_mu_bkg(
            n_on=self.counts.data,
            n_off=self.counts_off.data,
            alpha=self.alpha.data,
            mu_sig=self.npred_signal().data,
        )
        return RegionNDMap.from_geom(geom=self._geom, data=mu_bkg)

    def npred_off(self):
        """Predicted counts in the off region

        Returns
        -------
        npred_off : `Map`
            Predicted off counts
        """
        return self.npred_background() / self.alpha

    @property
    def background(self):
        """ alpha * noff"""
        return self.alpha * self.counts_off

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions"""
        alpha = self.acceptance / self.acceptance_off
        np.nan_to_num(alpha.data, copy=False)
        return alpha

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
        mu_sig = self.npred_signal().data
        on_stat_ = wstat(
            n_on=self.counts.data,
            n_off=self.counts_off.data,
            alpha=self.alpha.data,
            mu_sig=mu_sig,
        )
        return np.nan_to_num(on_stat_)

    def fake(self, npred_background, random_state="random-seed"):
        """Simulate fake counts for the current model and reduced irfs.

        This method overwrites the counts and off counts defined on the dataset object.

        Parameters
        ----------
        npred_background : `~gammapy.maps.RegionNDMap`
            Predicted background to be used in the on region.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
        """
        random_state = get_random_state(random_state)

        npred = self.npred_signal()
        npred.data = random_state.poisson(npred.data)
        npred_bkg = random_state.poisson(npred_background.data)
        self.counts = npred + npred_bkg

        npred_off = npred_background / self.alpha
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
        other : `~gammapy.datasets.SpectrumDatasetOnOff`
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
        >>> print(stacked)
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

        aeff = self.exposure / self.exposure.meta["livetime"]

        aeff.write(
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
            kernel = self.edisp.get_edisp_kernel()
            kernel.write(outdir / rmffile, overwrite=overwrite, use_sherpa=use_sherpa)

    def _ogip_meta(self):
        """Meta info for the OGIP data format"""
        try:
            livetime = self.exposure.meta["livetime"]
        except KeyError:
            raise ValueError(
                "Storing in ogip format require the livetime "
                "to be defined in the exposure meta data"
            )
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
            "exposure": livetime.to_value("s"),
            "obs_id": self.name,
        }

    @classmethod
    def from_ogip_files(cls, filename):
        """Read `~gammapy.datasets.SpectrumDatasetOnOff` from OGIP files.

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
            livetime = counts.meta["EXPOSURE"] * u.s

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
        exposure = aeff * livetime
        exposure.meta["livetime"] = livetime

        if edisp is not None:
            edisp.exposure_map.data = exposure.data[:, :, np.newaxis, :]

        return cls(
            counts=counts,
            exposure=exposure,
            counts_off=counts_off,
            edisp=edisp,
            mask_safe=mask_safe,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            name=str(counts.meta["OBS_ID"]),
            gti=gti,
        )

    def info_dict(self, in_safe_data_range=True):
        """Info dict with summary statistics, summed over energy

        Parameters
        ----------
        in_safe_data_range : bool
            Whether to sum only in the safe energy range

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        info = super().info_dict(in_safe_data_range)

        if self.mask_safe and in_safe_data_range:
            mask = self.mask_safe.data.astype(bool)
        else:
            mask = slice(None)

        counts_off = np.nan
        if self.counts_off is not None:
            counts_off = self.counts_off.data[mask].sum()

        info["counts_off"] = counts_off

        acceptance = 1
        if self.acceptance:
            # TODO: handle energy dependent a_on / a_off
            acceptance = self.acceptance.data[mask].sum()

        info["acceptance"] = acceptance

        acceptance_off = np.nan
        if self.acceptance_off:
            acceptance_off = acceptance * counts_off / info["background"]

        info["acceptance_off"] = acceptance_off

        alpha = np.nan
        if self.acceptance_off and self.acceptance:
            alpha = np.mean(self.alpha.data[mask])

        info["alpha"] = alpha

        info["sqrt_ts"] = WStatCountsStatistic(
            info["counts"], info["counts_off"], acceptance / acceptance_off,
        ).sqrt_ts
        info["stat_sum"] = self.stat_sum()
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
    def from_dict(cls, data, **kwargs):
        """Create flux point dataset from dict.

        Parameters
        ----------
        data : dict
            Dict containing data to create dataset from.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.

        """

        filename = make_path(data["filename"])
        dataset = cls.from_ogip_files(filename=filename)
        dataset.mask_fit = None
        return dataset

    @classmethod
    def from_spectrum_dataset(
        cls, dataset, acceptance, acceptance_off, counts_off=None
    ):
        """Create spectrum dataseton off from another dataset.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Spectrum dataset defining counts, edisp, exposure etc.
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
        if counts_off is None and dataset.background is not None:
            alpha = acceptance / acceptance_off
            counts_off = dataset.npred_background() / alpha

        return cls(
            models=dataset.models,
            counts=dataset.counts,
            exposure=dataset.exposure,
            counts_off=counts_off,
            edisp=dataset.edisp,
            mask_safe=dataset.mask_safe,
            mask_fit=dataset.mask_fit,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            gti=dataset.gti,
            name=dataset.name,
            meta_table=dataset.meta_table,
        )

    def to_spectrum_dataset(self, name=None):
        """ Convert a SpectrumDatasetOnOff to a SpectrumDataset
        The background model template is taken as alpha*counts_off

        Parameters
        ----------
        name: str
            Name of the new dataset

        Returns
        -------
        dataset: `SpectrumDataset`
            SpectrumDatset with cash statistics
        """

        name = make_name(name)
        return SpectrumDataset(
            counts=self.counts,
            exposure=self.exposure,
            edisp=self.edisp,
            name=name,
            gti=self.gti,
            mask_fit=self.mask_fit,
            mask_safe=self.mask_safe,
            meta_table=self.meta_table,
            background=self.background,
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
            kwargs["exposure"] = self.exposure.slice_by_idx(slices=slices)

        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.slice_by_idx(slices=slices)

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.slice_by_idx(slices=slices)

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.slice_by_idx(slices=slices)

        kwargs["acceptance"] = self.acceptance.slice_by_idx(slices=slices)
        kwargs["acceptance_off"] = self.acceptance_off.slice_by_idx(slices=slices)
        kwargs["counts_off"] = self.counts_off.slice_by_idx(slices=slices)
        return self.__class__(**kwargs)

    def resample_energy_axis(self, energy_axis, name=None):
        """Resample SpectrumDatasetOnOff over new reconstructed energy axis.

        Counts are summed taking into account safe mask.

        Parameters
        ----------
        energy_axis : `~gammapy.maps.MapAxis`
            New reconstructed energy axis
        name: str
            Name of the new dataset.

        Returns
        -------
        dataset: `SpectrumDataset`
            Resampled spectrum dataset .
        """
        dataset = super().resample_energy_axis(energy_axis=energy_axis, name=name)

        axis = dataset.counts.geom.axes["energy"]

        counts_off = None
        if self.counts_off is not None:
            counts_off = self.counts_off
            counts_off = counts_off.resample_axis(axis=axis, weights=self.mask_safe)

        acceptance = 1
        acceptance_off = None
        if self.acceptance is not None:
            acceptance = self.acceptance
            acceptance = acceptance.resample_axis(axis=axis, weights=self.mask_safe)

            background = self.alpha * self.counts_off
            background = background.resample_axis(axis=axis, weights=self.mask_safe)

            acceptance_off = acceptance * counts_off / background

        return self.__class__.from_spectrum_dataset(
            dataset,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            counts_off=counts_off,
        )
