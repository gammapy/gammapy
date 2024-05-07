# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from gammapy.maps.axes import UNIT_STRING_FORMAT, MapAxis
from gammapy.modeling.models import (
    DatasetModels,
    Models,
    SkyModel,
    TemplateSpatialModel,
)
from gammapy.utils.scripts import make_name, make_path
from .core import Dataset

log = logging.getLogger(__name__)

__all__ = ["FluxPointsDataset"]


def _get_reference_model(model, energy_bounds, margin_percent=70):
    if isinstance(model.spatial_model, TemplateSpatialModel):
        geom = model.spatial_model.map.geom
        emin = energy_bounds[0] * (1 - margin_percent / 100)
        emax = energy_bounds[-1] * (1 + margin_percent / 100)
        energy_axis = MapAxis.from_energy_bounds(
            emin, emax, nbin=20, per_decade=True, name="energy_true"
        )
        geom = geom.to_image().to_cube([energy_axis])
        return Models([model]).to_template_spectral_model(geom)
    else:
        return model.spectral_model


class FluxPointsDataset(Dataset):
    """Bundle a set of flux points with a parametric model,
    to compute fit statistic function using chi2 statistics.

    For more information see :ref:`datasets`.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Models (only spectral part needs to be set).
    data : `~gammapy.estimators.FluxPoints`
        Flux points. Must be sorted along the energy axis.
    mask_fit : `numpy.ndarray`
        Mask to apply for fitting.
    mask_safe : `numpy.ndarray`
        Mask defining the safe data range. By default, upper limit values are excluded.
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

    Make the fit:

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
    type    name     value         unit      ... frozen is_norm link prior
    ---- --------- ---------- -------------- ... ------ ------- ---- -----
             index 2.2159e+00                ...  False   False
         amplitude 2.1619e-13 TeV-1 s-1 cm-2 ...  False    True
         reference 1.0000e+00            TeV ...   True   False

    Note: In order to reproduce the example, you need the tests datasets folder.
    You may download it with the command:
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
        if not data.geom.has_energy_axis:
            raise ValueError("FluxPointsDataset needs an energy axis")
        self.data = data
        self.mask_fit = mask_fit
        self._name = make_name(name)
        self.models = models
        self.meta_table = meta_table

        if mask_safe is None:
            mask_safe = (~data.is_ul).data

        self.mask_safe = mask_safe

    @property
    def name(self):
        return self._name

    @property
    def gti(self):
        """Good time interval info (`GTI`)."""
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

    def write(self, filename, overwrite=False, checksum=False, **kwargs):
        """Write flux point dataset to file.

        Parameters
        ----------
        filename : str
            Filename to write to.
        overwrite : bool, optional
            Overwrite existing file. Default is False.
        checksum : bool
            When True adds both DATASUM and CHECKSUM cards to the headers written to the FITS file.
            Applies only if filename has .fits suffix. Default is False.
        **kwargs : dict, optional
             Keyword arguments passed to `~astropy.table.Table.write`.
        """
        table = self.data.to_table()

        if self.mask_fit is None:
            mask_fit = self.mask_safe
        else:
            mask_fit = self.mask_fit

        table["mask_fit"] = mask_fit
        table["mask_safe"] = self.mask_safe

        filename = make_path(filename)

        if "fits" in filename.suffixes:
            primary_hdu = fits.PrimaryHDU()
            hdu_table = fits.BinTableHDU(table, name="TABLE")
            fits.HDUList([primary_hdu, hdu_table]).writeto(
                filename, overwrite=overwrite, checksum=checksum
            )
        else:
            table.write(make_path(filename), overwrite=overwrite, **kwargs)

    @classmethod
    def read(cls, filename, name=None):
        """Read pre-computed flux points and create a dataset.

        Parameters
        ----------
        filename : str
            Filename to read from.
        name : str, optional
            Name of the new dataset. Default is None.
        Returns
        -------
        dataset : `FluxPointsDataset`
            FluxPointsDataset.
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
            data=FluxPoints.from_table(table),
            mask_fit=mask_fit,
            mask_safe=mask_safe,
        )

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create flux point dataset from dict.

        Parameters
        ----------
        data : dict
            Dictionary containing data to create dataset from.
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
            data=FluxPoints.from_table(table),
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
            n_bins = np.prod(self.data.geom.data_shape)
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
            reference_model = _get_reference_model(model, self._energy_bounds)
            sky_model = SkyModel(
                spectral_model=reference_model, temporal_model=model.temporal_model
            )
            flux_model = sky_model.evaluate_geom(
                self.data.geom.as_energy_true, self.gti
            )
            flux += flux_model
        return flux

    def stat_array(self):
        """Fit statistic array."""
        model = self.flux_pred()
        data = self.data.dnde.quantity
        try:
            sigma = self.data.dnde_err
        except AttributeError:
            sigma = (self.data.dnde_errn + self.data.dnde_errp) / 2
        return ((data - model) / sigma.quantity).to_value("") ** 2

    def residuals(self, method="diff"):
        """Compute flux point residuals.

        Parameters
        ----------
        method: {"diff", "diff/model"}
            Method used to compute the residuals. Available options are:

            - ``"diff"`` (default): data - model.
            - ``"diff/model"``: (data - model) / model.

        Returns
        -------
        residuals : `~numpy.ndarray`
            Residuals array.
        """
        fp = self.data

        model = self.flux_pred()

        residuals = self._compute_residuals(fp.dnde.quantity, model, method)
        # Remove residuals for upper_limits
        residuals[fp.is_ul.data] = np.nan
        return residuals

    def plot_fit(
        self,
        ax_spectrum=None,
        ax_residuals=None,
        kwargs_spectrum=None,
        kwargs_residuals=None,
        axis_name="energy",
    ):
        """Plot flux points, best fit model and residuals in two panels.

        Calls `~FluxPointsDataset.plot_spectrum` and `~FluxPointsDataset.plot_residuals`.

        Parameters
        ----------
        ax_spectrum : `~matplotlib.axes.Axes`, optional
            Axes to plot flux points and best fit model on. Default is None.
        ax_residuals : `~matplotlib.axes.Axes`, optional
            Axes to plot residuals on. Default is None.
        kwargs_spectrum : dict, optional
            Keyword arguments passed to `~FluxPointsDataset.plot_spectrum`. Default is None.
        kwargs_residuals : dict, optional
            Keyword arguments passed to `~FluxPointsDataset.plot_residuals`. Default is None.
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
        >>> kwargs_spectrum = {"kwargs_model": {"color":"red", "ls":"--"}, "kwargs_fp":{"color":"green", "marker":"o"}}
        >>> kwargs_residuals = {"color": "blue", "markersize":4, "marker":'s', }
        >>> dataset.plot_fit(kwargs_residuals=kwargs_residuals, kwargs_spectrum=kwargs_spectrum) # doctest: +SKIP
        """

        if self.data.geom.ndim > 3:
            raise ValueError("Plot fit works with only one energy axis")
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
        ax : `~matplotlib.axes.Axes`, optional
            Axes to plot on. Default is None.
        method : {"diff", "diff/model"}
            Normalization used to compute the residuals, see `FluxPointsDataset.residuals`. Default is "diff".
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.errorbar`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.

        """
        if self.data.geom.ndim > 3:
            raise ValueError("Plot residuals works with only one energy axis")
        ax = ax or plt.gca()

        fp = self.data
        residuals = self.residuals(method)

        xerr = self.data.energy_axis.as_plot_xerr

        yerr = fp._plot_get_flux_err(sed_type="dnde")

        if method == "diff/model":
            model = self.flux_pred()
            yerr = (yerr[0].quantity / model).squeeze(), (
                yerr[1].quantity / model
            ).squeeze()
        elif method == "diff":
            yerr = yerr[0].quantity.squeeze(), yerr[1].quantity.squeeze()
        else:
            raise ValueError('Invalid method, choose between "diff" and "diff/model"')

        kwargs.setdefault("color", kwargs.pop("c", "black"))
        kwargs.setdefault("marker", "+")
        kwargs.setdefault("linestyle", kwargs.pop("ls", "none"))

        with quantity_support():
            ax.errorbar(
                fp.energy_ref, residuals.squeeze(), xerr=xerr, yerr=yerr, **kwargs
            )

        ax.axhline(0, color=kwargs["color"], lw=0.5)

        # format axes
        ax.set_xlabel(f"Energy [{self._energy_unit.to_string(UNIT_STRING_FORMAT)}]")
        ax.set_xscale("log")
        label = self._residuals_labels[method]
        ax.set_ylabel(f"Residuals\n {label}")
        ymin = np.nanmin(residuals - yerr[0])
        ymax = np.nanmax(residuals + yerr[1])
        ymax = max(abs(ymin), ymax)
        ax.set_ylim(-1.05 * ymax, 1.05 * ymax)
        return ax

    def plot_spectrum(
        self, ax=None, kwargs_fp=None, kwargs_model=None, axis_name="energy"
    ):
        """Plot flux points and model.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axes to plot on. Default is None.
        kwargs_fp : dict, optional
            Keyword arguments passed to `gammapy.estimators.FluxPoints.plot` to configure the plot style.
            Default is None.
        kwargs_model : dict, optional
            Keyword arguments passed to `gammapy.modeling.models.SpectralModel.plot` and
            `gammapy.modeling.models.SpectralModel.plot_error` to configure the plot style. Default is None.
        axis_name : str
            Axis along which to plot the flux points for multiple axes. Default is energy.

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
        kwargs_fp.setdefault("sed_type", "e2dnde")
        if axis_name == "time":
            kwargs_fp["sed_type"] = "norm"
        ax = self.data.plot(ax=ax, **kwargs_fp, axis_name=axis_name)

        kwargs_model.setdefault("label", "Best fit model")

        kwargs_model.setdefault("zorder", 10)

        for model in self.models:
            if model.datasets_names is None or self.name in model.datasets_names:
                if axis_name == "energy":
                    kwargs_model.setdefault("sed_type", "e2dnde")
                    kwargs_model.setdefault("energy_bounds", self._energy_bounds)
                    model.spectral_model.plot(ax=ax, **kwargs_model)
                if axis_name == "time":
                    kwargs_model.setdefault(
                        "time_range", self.data.geom.axes["time"].time_bounds
                    )
                    model.temporal_model.plot(ax=ax, **kwargs_model)

        if axis_name == "energy":
            kwargs_model["color"] = ax.lines[-1].get_color()
            kwargs_model.pop("label")
            for model in self.models:
                if model.datasets_names is None or self.name in model.datasets_names:
                    model.spectral_model.plot_error(ax=ax, **kwargs_model)

        return ax
