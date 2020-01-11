# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.table import Table
from astropy import units as u
from .core import Dataset
from gammapy.modeling.models import SkyModels, SkyModel
from gammapy.modeling import Parameters
from gammapy.spectrum import FluxPoints


__all__ = ["FluxPointsDataset"]


class FluxPointsDataset(Dataset):
    """
    Fit a set of flux points with a parametric model.

    Parameters
    ----------
    models : `~gammapy.modeling.models.SkyModels`
        Models (only spectral part needs to be set)
    data : `~gammapy.spectrum.FluxPoints`
        Flux points.
    mask_fit : `numpy.ndarray`
        Mask to apply for fitting
    mask_safe : `numpy.ndarray`
        Mask defining the safe data range.

    Examples
    --------
    Load flux points from file and fit with a power-law model::

        from gammapy.modeling import Fit
        from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
        from gammapy.spectrum import FluxPoints, FluxPointsDataset

        filename = "$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits"
        flux_points = FluxPoints.read(filename)

        model = SkyModel(spectral_model=PowerLawSpectralModel())

        dataset = FluxPointsDataset(model, flux_points)
        fit = Fit([dataset])
        result = fit.run()
        print(result)
        print(result.parameters.to_table())
    """

    likelihood_type = "chi2"
    tag = "FluxPointsDataset"

    def __init__(self, models, data, mask_fit=None, mask_safe=None, name=""):
        self.data = data
        self.mask_fit = mask_fit
        self.name = name
        self.models = models
        if data.sed_type != "dnde":
            raise ValueError("Currently only flux points of type 'dnde' are supported.")

        if mask_safe is None:
            mask_safe = np.isfinite(data.table["dnde"])

        self.mask_safe = mask_safe

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        if isinstance(value, SkyModels):
            models = value
        elif isinstance(value, list):
            models = SkyModels(value)
        elif isinstance(value, SkyModel):
            models = SkyModels([value])
        else:
            raise TypeError(f"Invalid: {value!r}")

        self._models = models

    @property
    def parameters(self):
        """List of parameters (`~gammapy.modeling.Parameters`)"""
        parameters = []

        for component in self.models:
            parameters.append(component.spectral_model.parameters)

        return Parameters.from_stack(parameters)

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
        table.write(filename, overwrite=overwrite, **kwargs)

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
        dataset : `FluxPointsDataset`
            Flux point datasets.
        """
        models = [model for model in models if model.name in data["models"]]
        # TODO: assumes that the model is a skymodel
        # so this will work only when this change will be effective
        table = Table.read(data["filename"])
        mask_fit = table["mask_fit"].data.astype("bool")
        mask_safe = table["mask_safe"].data.astype("bool")
        table.remove_columns(["mask_fit", "mask_safe"])
        return cls(
            models=models,
            name=data["name"],
            data=FluxPoints(table),
            mask_fit=mask_fit,
            mask_safe=mask_safe,
        )

    def to_dict(self, filename=""):
        """Convert to dict for YAML serialization."""
        if self.models is not None:
            models = [_.name for _ in self.models]
        else:
            models = []

        return {
            "name": self.name,
            "type": self.tag,
            "models": models,
            "filename": str(filename),
        }

    def __str__(self):
        str_ = f"{self.__class__.__name__}: \n"
        str_ += "\n"
        if self.models is None:
            str_ += "\t{:32}:   {} \n".format("Model Name", "No Model")
        else:
            str_ += "\t{:32}:   {} \n".format("Total flux points", len(self.data.table))
            str_ += "\t{:32}:   {} \n".format(
                "Points used for the fit", self.mask.sum()
            )
            str_ += "\t{:32}:   {} \n".format(
                "Excluded for safe energy range", (~self.mask_safe).sum()
            )
            if self.mask_fit is None:
                str_ += "\t{:32}:   {} \n".format("Excluded by user", "0")
            else:
                str_ += "\t{:32}:   {} \n".format(
                    "Excluded by user", (~self.mask_fit).sum()
                )
            str_ += "\t{:32}:   {}\n".format(
                "Model Name", self.models.__class__.__name__
            )
            str_ += "\t{:32}:   {}\n".format("N parameters", len(self.parameters))
            str_ += "\t{:32}:   {}\n".format(
                "N free parameters", len(self.parameters.free_parameters)
            )
            str_ += "\tList of parameters\n"
            for par in self.parameters:
                if par.frozen:
                    if par.name == "amplitude":
                        str_ += "\t \t {:14} (Frozen):   {:.2e} {} \n".format(
                            par.name, par.value, par.unit
                        )
                    else:
                        str_ += "\t \t {:14} (Frozen):   {:.2f} {} \n".format(
                            par.name, par.value, par.unit
                        )
                else:
                    if par.name == "amplitude":
                        str_ += "\t \t {:23}:   {:.2e} {} \n".format(
                            par.name, par.value, par.unit
                        )
                    else:
                        str_ += "\t \t {:23}:   {:.2f} {} \n".format(
                            par.name, par.value, par.unit
                        )
            str_ += "\t{:32}:   {:.2f}\n".format("Likelihood value", self.stat_sum())
        return str_

    def data_shape(self):
        """Shape of the flux points data (tuple)."""
        return self.data.e_ref.shape

    def flux_pred(self):
        """Compute predicted flux."""
        flux = 0.0
        for component in self.models:
            flux += component.spectral_model(self.data.e_ref)
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

    def peek(self, method="diff/model", **kwargs):
        """Plot flux points, best fit model and residuals.

        Parameters
        ----------
        method : {"diff", "diff/model", "diff/sqrt(model)"}
            Method used to compute the residuals, see `MapDataset.residuals()`
        """
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt

        gs = GridSpec(7, 1)

        ax_spectrum = plt.subplot(gs[:5, :])
        self.plot_spectrum(ax=ax_spectrum, **kwargs)

        ax_spectrum.set_xticks([])

        ax_residuals = plt.subplot(gs[5:, :])
        self.plot_residuals(ax=ax_residuals, method=method)
        return ax_spectrum, ax_residuals

    @property
    def _e_range(self):
        try:
            return u.Quantity([self.data.e_min.min(), self.data.e_max.max()])
        except KeyError:
            return u.Quantity([self.data.e_ref.min(), self.data.e_ref.max()])

    @property
    def _e_unit(self):
        return self.data.e_ref.unit

    def plot_residuals(self, ax=None, method="diff", **kwargs):
        """Plot flux point residuals.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.
        method : {"diff", "diff/model", "diff/sqrt(model)"}
            Method used to compute the residuals, see `MapDataset.residuals()`
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.errorbar`.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        residuals = self.residuals(method=method)

        fp = self.data

        xerr = fp._plot_get_energy_err()
        if xerr is not None:
            xerr = xerr[0].to_value(self._e_unit), xerr[1].to_value(self._e_unit)

        model = self.flux_pred()
        yerr = fp._plot_get_flux_err(fp.sed_type)

        if method == "diff":
            unit = yerr[0].unit
            yerr = yerr[0].to_value(unit), yerr[1].to_value(unit)
        elif method == "diff/model":
            unit = ""
            yerr = (yerr[0] / model).to_value(""), (yerr[1] / model).to_value(unit)
        else:
            raise ValueError("Invalid method, choose between 'diff' and 'diff/model'")

        kwargs.setdefault("marker", "+")
        kwargs.setdefault("ls", "None")
        kwargs.setdefault("color", "black")

        ax.errorbar(
            self.data.e_ref.value, residuals.value, xerr=xerr, yerr=yerr, **kwargs
        )

        # format axes
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("Residuals {}".format(unit.__str__()))
        ax.set_xlabel(f"Energy ({self._e_unit})")
        ax.set_xscale("log")
        ax.set_xlim(self._e_range.to_value(self._e_unit))
        y_max = 2 * np.nanmax(residuals).value
        ax.set_ylim(-y_max, y_max)
        return ax

    def plot_spectrum(self, ax=None, fp_kwargs=None, model_kwargs=None):
        """
        Plot spectrum including flux points and model.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.
        fp_kwargs : dict
            Keyword arguments passed to `FluxPoints.plot`.
        model_kwargs : dict
            Keywords passed to `SpectralModel.plot` and `SpectralModel.plot_error`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        fp_kwargs = {} if fp_kwargs is None else fp_kwargs
        model_kwargs = {} if model_kwargs is None else model_kwargs

        kwargs = {
            "flux_unit": "erg-1 cm-2 s-1",
            "energy_unit": "TeV",
            "energy_power": 2,
        }

        # plot flux points
        plot_kwargs = kwargs.copy()
        plot_kwargs.update(fp_kwargs)
        plot_kwargs.setdefault("label", "Flux points")
        ax = self.data.plot(ax=ax, **plot_kwargs)

        plot_kwargs = kwargs.copy()
        plot_kwargs.setdefault("energy_range", self._e_range)
        plot_kwargs.setdefault("zorder", 10)
        plot_kwargs.update(model_kwargs)
        plot_kwargs.setdefault("label", "Best fit model")
        for _ in self.models:
            _.spectral_model.plot(ax=ax, **plot_kwargs)

        plot_kwargs.setdefault("color", ax.lines[-1].get_color())
        del plot_kwargs["label"]

        if self.models.parameters.covariance is not None:
            try:
                self.models.plot_error(ax=ax, **plot_kwargs)
            except AttributeError:
                log.debug("Model does not support evaluation of errors")

        # format axes
        ax.set_xlim(self._e_range.to_value(self._e_unit))
        return ax
