# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from gammapy.modeling.models import DatasetModels
from gammapy.utils.scripts import make_name, make_path
from .core import Dataset

log = logging.getLogger(__name__)

__all__ = ["FluxPointsDataset"]


class FluxPointsDataset(Dataset):
    """Bundle a set of flux points with a parametric model,
    to compute fit statistic function using chi2 statistics.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Models (only spectral part needs to be set)
    data : `~gammapy.estimators.FluxPoints`
        Flux points. Must be sorted along the energy axis
    mask_fit : `numpy.ndarray`
        Mask to apply for fitting
    mask_safe : `numpy.ndarray`
        Mask defining the safe data range. By default upper limit values are excluded.
    meta_table : `~astropy.table.Table`
        Table listing information on observations used to create the dataset.
        One line per observation for stacked datasets.

    Examples
    --------
    Load flux points from file and fit with a power-law model::

    >>> from gammapy.modeling import Fit
    >>> from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
    >>> from gammapy.estimators import FluxPoints
    >>> from gammapy.datasets import FluxPointsDataset
    >>> filename = "$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits"
    >>> dataset = FluxPointsDataset.read(filename)
    >>> model = SkyModel(spectral_model=PowerLawSpectralModel())
    >>> dataset.models = model

    Make the fit

    >>> fit = Fit()
    >>> result = fit.run([dataset])
    >>> print(result)
    OptimizeResult
    <BLANKLINE>
        backend    : minuit
        method     : migrad
        success    : True
        message    : Optimization terminated successfully.
        nfev       : 135
        total stat : 25.21
    <BLANKLINE>
    CovarianceResult
    <BLANKLINE>
        backend    : minuit
        method     : hesse
        success    : True
        message    : Hesse terminated successfully.

    >>> print(result.parameters.to_table())
      type      name     value         unit      ... max frozen is_norm link
    -------- --------- ---------- -------------- ... --- ------ ------- ----
    spectral     index 2.2159e+00                ... nan  False   False
    spectral amplitude 2.1619e-13 cm-2 s-1 TeV-1 ... nan  False    True
    spectral reference 1.0000e+00            TeV ... nan   True   False

    Note: In order to reproduce the example you need the tests datasets folder.
    You may download it with the command
    ``gammapy download datasets --tests --out $GAMMAPY_DATA``
    """

    stat_type = "chi2"
    tag = "FluxPointsDataset"

    def __init__(
        self,
        models=None,
        data=None,
        mask_fit=None,
        mask_safe=None,
        name=None,
        meta_table=None,
    ):
        if data.geom.ndim != 3 or not data.geom.has_energy_axis:
            raise ValueError("FluxPointsDataset only supports an energy axis")
        self.data = data
        self.mask_fit = mask_fit
        self._name = make_name(name)
        self.models = models
        self.meta_table = meta_table

        if mask_safe is None:
            mask_safe = (~data.is_ul).data[:, 0, 0]

        self.mask_safe = mask_safe

    @property
    def name(self):
        return self._name

    @property
    def gti(self):
        """Good time interval info (`GTI`)"""
        return self.data.gti

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models):
        if models is None:
            self._models = None
        else:
            models = DatasetModels(models)
            self._models = models.select(datasets_names=self.name)

    def write(self, filename, overwrite=False, **kwargs):
        """Write flux point dataset to file.

        Parameters
        ----------
        filename : str
            Filename to write to.
        overwrite : bool
            Overwrite existing file.
        **kwargs : dict
             Keyword arguments passed to `~astropy.table.Table.write`.
        """
        table = self.data.to_table()

        if self.mask_fit is None:
            mask_fit = self.mask_safe
        else:
            mask_fit = self.mask_fit

        table["mask_fit"] = mask_fit
        table["mask_safe"] = self.mask_safe
        table.write(make_path(filename), overwrite=overwrite, **kwargs)

    @classmethod
    def read(cls, filename, name=None, format="gadf-sed"):
        """Read pre-computed flux points and create a dataset

        Parameters
        ----------
        filename : str
            Filename to read from.
        name : str
            Name of the new dataset.
        format : {"gadf-sed"}
            Format of the dataset file.

        Returns
        -------
        dataset : `FluxPointsDataset`
            FluxPointsDataset
        """
        from gammapy.estimators import FluxPoints

        filename = make_path(filename)
        table = Table.read(filename)
        mask_fit = None
        mask_safe = None

        if "mask_safe" in table.colnames:
            mask_safe = table["mask_safe"].data.astype("bool")

        if "mask_fit" in table.colnames:
            mask_fit = table["mask_fit"].data.astype("bool")

        return cls(
            name=make_name(name),
            data=FluxPoints.from_table(table, format=format),
            mask_fit=mask_fit,
            mask_safe=mask_safe,
        )

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create flux point dataset from dict.

        Parameters
        ----------
        data : dict
            Dict containing data to create dataset from.

        Returns
        -------
        dataset : `FluxPointsDataset`
            Flux point datasets.
        """
        from gammapy.estimators import FluxPoints

        filename = make_path(data["filename"])
        table = Table.read(filename)
        mask_fit = table["mask_fit"].data.astype("bool")
        mask_safe = table["mask_safe"].data.astype("bool")
        table.remove_columns(["mask_fit", "mask_safe"])
        return cls(
            name=data["name"],
            data=FluxPoints.from_table(table, format="gadf-sed"),
            mask_fit=mask_fit,
            mask_safe=mask_safe,
        )

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"

        str_ += "\t{:32}: {} \n\n".format("Name", self.name)

        # data section
        n_bins = 0
        if self.data is not None:
            n_bins = self.data.energy_axis.nbin
        str_ += "\t{:32}: {} \n".format("Number of total flux points", n_bins)

        n_fit_bins = 0
        if self.mask is not None:
            n_fit_bins = np.sum(self.mask.data)
        str_ += "\t{:32}: {} \n\n".format("Number of fit bins", n_fit_bins)

        # likelihood section
        str_ += "\t{:32}: {}\n".format("Fit statistic type", self.stat_type)

        stat = np.nan
        if self.data is not None and self.models is not None:
            stat = self.stat_sum()
        str_ += "\t{:32}: {:.2f}\n\n".format("Fit statistic value (-2 log(L))", stat)

        # model section
        n_models = 0
        if self.models is not None:
            n_models = len(self.models)

        str_ += "\t{:32}: {} \n".format("Number of models", n_models)

        if self.models is not None:
            str_ += "\t{:32}: {}\n".format(
                "Number of parameters", len(self.models.parameters)
            )
            str_ += "\t{:32}: {}\n\n".format(
                "Number of free parameters", len(self.models.parameters.free_parameters)
            )
            str_ += "\t" + "\n\t".join(str(self.models).split("\n")[2:])

        return str_.expandtabs(tabsize=2)

    def data_shape(self):
        """Shape of the flux points data (tuple)."""
        return self.data.energy_ref.shape

    def flux_pred(self):
        """Compute predicted flux."""
        flux = 0.0
        for model in self.models:
            flux_model = model.spectral_model(self.data.energy_ref)

            if model.temporal_model is not None:
                integral = model.temporal_model.integral(
                    self.gti.time_start, self.gti.time_stop
                )
                flux_model *= np.sum(integral)

            flux += flux_model
        return flux

    def stat_array(self):
        """Fit statistic array."""
        model = self.flux_pred()
        data = self.data.dnde.quantity[:, 0, 0]
        try:
            sigma = self.data.dnde_err
        except AttributeError:
            sigma = (self.data.dnde_errn + self.data.dnde_errp) / 2
        return ((data - model) / sigma.quantity[:, 0, 0]).to_value("") ** 2

    def residuals(self, method="diff"):
        """Compute flux point residuals

        Parameters
        ----------
        method: {"diff", "diff/model"}
            Method used to compute the residuals. Available options are:
                - `diff` (default): data - model
                - `diff/model`: (data - model) / model

        Returns
        -------
        residuals : `~numpy.ndarray`
            Residuals array.
        """
        fp = self.data

        model = self.flux_pred()

        residuals = self._compute_residuals(fp.dnde.quantity[:, 0, 0], model, method)
        # Remove residuals for upper_limits
        residuals[fp.is_ul.data[:, 0, 0]] = np.nan
        return residuals

    def plot_fit(
        self,
        ax_spectrum=None,
        ax_residuals=None,
        kwargs_spectrum=None,
        kwargs_residuals=None,
    ):
        """Plot flux points, best fit model and residuals in two panels.

        Calls `~FluxPointsDataset.plot_spectrum` and `~FluxPointsDataset.plot_residuals`.

        Parameters
        ----------
        ax_spectrum : `~matplotlib.axes.Axes`
            Axes to plot flux points and best fit model on.
        ax_residuals : `~matplotlib.axes.Axes`
            Axes to plot residuals on.
        kwargs_spectrum : dict
            Keyword arguments passed to `~FluxPointsDataset.plot_spectrum`.
        kwargs_residuals : dict
            Keyword arguments passed to `~FluxPointsDataset.plot_residuals`.

        Returns
        -------
        ax_spectrum, ax_residuals : `~matplotlib.axes.Axes`
            Flux points, best fit model and residuals plots.

        Examples
        --------
        >>> from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
        >>> from gammapy.estimators import FluxPoints
        >>> from gammapy.datasets import FluxPointsDataset

        >>> #load precomputed flux points
        >>> filename = "$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits"
        >>> flux_points = FluxPoints.read(filename)
        >>> model = SkyModel(spectral_model=PowerLawSpectralModel())
        >>> dataset = FluxPointsDataset(model, flux_points)
        >>> #configuring optional parameters
        >>> kwargs_spectrum = {"kwargs_model": {"color":"red", "ls":"--"}, "kwargs_fp":{"color":"green", "marker":"o"}}  # noqa: E501
        >>> kwargs_residuals = {"color": "blue", "markersize":4, "marker":'s', }
        >>> dataset.plot_fit(kwargs_residuals=kwargs_residuals, kwargs_spectrum=kwargs_spectrum) # doctest: +SKIP noqa: E501
        """
        fig = plt.figure(figsize=(9, 7))

        gs = GridSpec(7, 1)
        if ax_spectrum is None:
            ax_spectrum = fig.add_subplot(gs[:5, :])

        if ax_residuals is None:
            ax_residuals = fig.add_subplot(gs[5:, :], sharex=ax_spectrum)

        kwargs_spectrum = kwargs_spectrum or {}
        kwargs_residuals = kwargs_residuals or {}
        kwargs_residuals.setdefault("method", "diff/model")

        self.plot_spectrum(ax=ax_spectrum, **kwargs_spectrum)
        self.plot_residuals(ax=ax_residuals, **kwargs_residuals)
        return ax_spectrum, ax_residuals

    @property
    def _energy_bounds(self):
        try:
            return u.Quantity([self.data.energy_min.min(), self.data.energy_max.max()])
        except KeyError:
            return u.Quantity([self.data.energy_ref.min(), self.data.energy_ref.max()])

    @property
    def _energy_unit(self):
        return self.data.energy_ref.unit

    def plot_residuals(self, ax=None, method="diff", **kwargs):
        """Plot flux point residuals.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        method : {"diff", "diff/model"}
            Normalization used to compute the residuals, see `FluxPointsDataset.residuals`.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.errorbar`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.

        """
        ax = ax or plt.gca()

        fp = self.data
        residuals = self.residuals(method)

        xerr = self.data.energy_axis.as_plot_xerr

        yerr = fp._plot_get_flux_err(sed_type="dnde")

        if method == "diff/model":
            model = self.flux_pred()
            yerr = (yerr[0].quantity[:, 0, 0] / model), (
                yerr[1].quantity[:, 0, 0] / model
            )
        elif method == "diff":
            yerr = yerr[0].quantity[:, 0, 0], yerr[1].quantity[:, 0, 0]
        else:
            raise ValueError('Invalid method, choose between "diff" and "diff/model"')

        kwargs.setdefault("color", kwargs.pop("c", "black"))
        kwargs.setdefault("marker", "+")
        kwargs.setdefault("linestyle", kwargs.pop("ls", "none"))

        with quantity_support():
            ax.errorbar(fp.energy_ref, residuals, xerr=xerr, yerr=yerr, **kwargs)

        ax.axhline(0, color=kwargs["color"], lw=0.5)

        # format axes
        ax.set_xlabel(f"Energy [{self._energy_unit}]")
        ax.set_xscale("log")
        label = self._residuals_labels[method]
        ax.set_ylabel(f"Residuals\n {label}")
        ymin = np.nanmin(residuals - yerr[0])
        ymax = np.nanmax(residuals + yerr[1])
        ymax = max(abs(ymin), ymax)
        ax.set_ylim(-1.05 * ymax, 1.05 * ymax)
        return ax

    def plot_spectrum(self, ax=None, kwargs_fp=None, kwargs_model=None):
        """Plot spectrum including flux points and model.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        kwargs_fp : dict
            Keyword arguments passed to `gammapy.estimators.FluxPoints.plot`.
        kwargs_model : dict
            Keyword arguments passed to `gammapy.modeling.models.SpectralModel.plot` and
            `gammapy.modeling.models.SpectralModel.plot_error`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.

        Examples
        --------
        >>> from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
        >>> from gammapy.estimators import FluxPoints
        >>> from gammapy.datasets import FluxPointsDataset

        >>> #load precomputed flux points
        >>> filename = "$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits"
        >>> flux_points = FluxPoints.read(filename)
        >>> model = SkyModel(spectral_model=PowerLawSpectralModel())
        >>> dataset = FluxPointsDataset(model, flux_points)
        >>> #configuring optional parameters
        >>> kwargs_model = {"color":"red", "ls":"--"}
        >>> kwargs_fp = {"color":"green", "marker":"o"}
        >>> dataset.plot_spectrum(kwargs_fp=kwargs_fp, kwargs_model=kwargs_model) # doctest: +SKIP
        """
        kwargs_fp = (kwargs_fp or {}).copy()
        kwargs_model = (kwargs_model or {}).copy()

        # plot flux points
        kwargs_fp.setdefault("label", "Flux points")
        kwargs_fp.setdefault("sed_type", "e2dnde")
        ax = self.data.plot(ax, **kwargs_fp)

        kwargs_model.setdefault("energy_bounds", self._energy_bounds)
        kwargs_model.setdefault("label", "Best fit model")
        kwargs_model.setdefault("sed_type", "e2dnde")
        kwargs_model.setdefault("zorder", 10)

        for model in self.models:
            if model.datasets_names is None or self.name in model.datasets_names:
                model.spectral_model.plot(ax=ax, **kwargs_model)

        kwargs_model["color"] = ax.lines[-1].get_color()
        kwargs_model.pop("label")

        for model in self.models:
            if model.datasets_names is None or self.name in model.datasets_names:
                model.spectral_model.plot_error(ax=ax, **kwargs_model)

        return ax
