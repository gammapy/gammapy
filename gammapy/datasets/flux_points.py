# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.table import Table
from gammapy.modeling.models import DatasetModels
from gammapy.utils.scripts import make_name, make_path
from .core import Dataset
from .utils import get_axes

log = logging.getLogger(__name__)

__all__ = ["FluxPointsDataset"]


class FluxPointsDataset(Dataset):
    """
    Fit a set of flux points with a parametric model.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Models (only spectral part needs to be set)
    data : `~gammapy.estimators.FluxPoints`
        Flux points.
    mask_fit : `numpy.ndarray`
        Mask to apply for fitting
    mask_safe : `numpy.ndarray`
        Mask defining the safe data range.
    meta_table : `~astropy.table.Table`
        Table listing informations on observations used to create the dataset.
        One line per observation for stacked datasets.

    Examples
    --------
    Load flux points from file and fit with a power-law model::

        from gammapy.modeling import Fit
        from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
        from gammapy.estimators import FluxPoints
        from gammapy.datasets import FluxPointsDataset

        filename = "$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits"
        flux_points = FluxPoints.read(filename)

        model = SkyModel(spectral_model=PowerLawSpectralModel())

        dataset = FluxPointsDataset(model, flux_points)
        fit = Fit([dataset])
        result = fit.run()
        print(result)
        print(result.parameters.to_table())

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
        self.data = data
        self.mask_fit = mask_fit
        self._name = make_name(name)
        self.models = models
        self.meta_table = meta_table

        if data.sed_type != "dnde":
            raise ValueError("Currently only flux points of type 'dnde' are supported.")

        if mask_safe is None:
            mask_safe = np.isfinite(data.table["dnde"])

        self.mask_safe = mask_safe

    @property
    def name(self):
        return self._name

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models):
        if models is None:
            self._models = None
        else:
            self._models = DatasetModels(models).select(self.name)

    def write(self, filename, overwrite=True, **kwargs):
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
        table = self.data.table.copy()
        if self.mask_fit is None:
            mask_fit = self.mask_safe
        else:
            mask_fit = self.mask_fit

        table["mask_fit"] = mask_fit
        table["mask_safe"] = self.mask_safe
        table.write(make_path(filename), overwrite=overwrite, **kwargs)

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
            data=FluxPoints(table),
            mask_fit=mask_fit,
            mask_safe=mask_safe,
        )

    def to_dict(self, filename=""):
        """Convert to dict for YAML serialization."""
        return {
            "name": self.name,
            "type": self.tag,
            "filename": str(filename),
        }

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"

        str_ += "\t{:32}: {} \n\n".format("Name", self.name)

        # data section
        n_bins = 0
        if self.data is not None:
            n_bins = len(self.data.table)
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

        str_ += "\t{:32}: {}\n".format(
            "Number of parameters", len(self.models.parameters)
        )
        str_ += "\t{:32}: {}\n\n".format(
            "Number of free parameters", len(self.models.parameters.free_parameters)
        )

        if self.models is not None:
            str_ += "\t" + "\n\t".join(str(self.models).split("\n")[2:])

        return str_.expandtabs(tabsize=2)

    def data_shape(self):
        """Shape of the flux points data (tuple)."""
        return self.data.energy_ref.shape

    def flux_pred(self):
        """Compute predicted flux."""
        flux = 0.0
        for model in self.models:
            flux += model.spectral_model(self.data.energy_ref)
        return flux

    def stat_array(self):
        """Fit statistic array."""
        model = self.flux_pred()
        data = self.data.table["dnde"].quantity
        sigma = self.data.table["dnde_err"].quantity
        return ((data - model) / sigma).to_value("") ** 2

    def residuals(self, method="diff"):
        """Compute the flux point residuals ().

        Parameters
        ----------
        method: {"diff", "diff/model", "diff/sqrt(model)"}
            Method used to compute the residuals. Available options are:
                - `diff` (default): data - model
                - `diff/model`: (data - model) / model
                - `diff/sqrt(model)`: (data - model) / sqrt(model)
                - `norm='sqrt_model'` for: (flux points - model)/sqrt(model)


        Returns
        -------
        residuals : `~numpy.ndarray`
            Residuals array.
        """
        fp = self.data
        data = fp.table[fp.sed_type]

        model = self.flux_pred()

        residuals = self._compute_residuals(data, model, method)
        # Remove residuals for upper_limits
        residuals[fp.is_ul] = np.nan
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
        kwargs_residuals.setdefault("method", "diff/model")

        self.plot_spectrum(ax_spectrum, **kwargs_spectrum)
        ax_spectrum.label_outer()

        self.plot_residuals(ax_residuals, **kwargs_residuals)
        method = kwargs_residuals["method"]
        label = self._residuals_labels[method]
        unit = (
            self.data._plot_get_flux_err(self.data.sed_type)[0].unit
            if method == "diff"
            else ""
        )
        ax_residuals.set_ylabel("Residuals\n" + label + (f"\n[{unit}]" if unit else ""))

        return ax_spectrum, ax_residuals

    @property
    def _energy_range(self):
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
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()

        fp = self.data
        residuals = self.residuals(method)

        xerr = fp._plot_get_energy_err()
        if xerr is not None:
            xerr = (
                xerr[0].to_value(self._energy_unit),
                xerr[1].to_value(self._energy_unit),
            )

        yerr = fp._plot_get_flux_err(fp.sed_type)
        if method == "diff":
            unit = yerr[0].unit
            yerr = yerr[0].to_value(unit), yerr[1].to_value(unit)
        elif method == "diff/model":
            model = self.flux_pred()
            unit = ""
            yerr = (yerr[0] / model).to_value(unit), (yerr[1] / model).to_value(unit)
        else:
            raise ValueError('Invalid method, choose between "diff" and "diff/model"')

        kwargs.setdefault("color", kwargs.pop("c", "black"))
        kwargs.setdefault("marker", "+")
        kwargs.setdefault("linestyle", kwargs.pop("ls", "none"))

        ax.errorbar(
            fp.energy_ref.value, residuals.value, xerr=xerr, yerr=yerr, **kwargs
        )
        ax.axhline(0, color=kwargs["color"], lw=0.5)

        # format axes
        ax.set_xlabel(f"Energy [{self._energy_unit}]")
        ax.set_xscale("log")
        ax.set_xlim(self._energy_range.to_value(self._energy_unit))
        label = self._residuals_labels[method]
        ax.set_ylabel(f"Residuals ({label}){f' [{unit}]' if unit else ''}")
        ymin = 1.05 * np.nanmin(residuals.value - yerr[0])
        ymax = 1.05 * np.nanmax(residuals.value + yerr[1])
        ax.set_ylim(ymin, ymax)
        return ax

    def plot_spectrum(self, ax=None, kwargs_fp=None, kwargs_model=None, **kwargs):
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
        **kwargs: dict
            Keyword arguments passed to all plot methods.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        kwargs_fp = kwargs_fp or {}
        kwargs_model = kwargs_model or {}

        kwargs.setdefault("energy_power", 2)
        kwargs.setdefault("energy_unit", "TeV")
        kwargs.setdefault("flux_unit", "erg-1 cm-2 s-1")

        # plot flux points
        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_fp)
        plot_kwargs.setdefault("label", "Flux points")
        ax = self.data.plot(ax, **plot_kwargs)

        plot_kwargs = kwargs.copy()
        plot_kwargs.update(kwargs_model)
        plot_kwargs.setdefault("energy_range", self._energy_range)
        plot_kwargs.setdefault("label", "Best fit model")
        plot_kwargs.setdefault("zorder", 10)

        for model in self.models:
            if model.datasets_names is None or self.name in model.datasets_names:
                model.spectral_model.plot(ax=ax, **plot_kwargs)

        plot_kwargs.setdefault("color", plot_kwargs.pop("c", ax.lines[-1].get_color()))
        del plot_kwargs["label"]

        for model in self.models:
            if model.datasets_names is None or self.name in model.datasets_names:
                if not np.all(model == 0):
                    model.spectral_model.plot_error(ax=ax, **plot_kwargs)

        # format axes
        ax.set_xlim(self._energy_range.to_value(self._energy_unit))
        return ax
