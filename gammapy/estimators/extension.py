# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.datasets import Datasets
from gammapy.estimators.parameter import ParameterEstimator
from gammapy.modeling import Fit
from gammapy.utils.pbar import progress_bar


class ExtensionEstimator(ParameterEstimator):
    """Source extension estimator.

    Estimates source size for a given list of datasets and a given model.

    The size can be estimated in various energy intervals.

    Parameters
    ----------
    energy_edges : `~astropy.units.Quantity`
        Edges of the energy intervals where the extension is estimated.
    source : str or int
        For which source in the model to compute the extension.
    size_min : float
        Minimum value for the size used for the fit statistic profile evaluation.
    size_max : float
        Maximum value for the size used for the fit statistic profile evaluation.
    size_n_values : int
        Number of size values used for the fit statistic profile.
    size_values : `~astropy.coordinates.Angle`
        Array of size values to be used for the fit statistic profile.
    n_sigma : int
        Number of sigma to use for asymmetric error computation. Default is 1.
    n_sigma_ul : int
        Number of sigma to use for upper limit computation. Default is 2.
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors on size.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optional steps are not executed.
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    reoptimize : bool
        Re-optimize other free model parameters. Default is True.
    """

    tag = "ExtensionEstimator"

    def __init__(
        self,
        energy_edges=[1, 10] * u.TeV,
        source=0,
        size_min="0.0001 deg",
        size_max="0.5 deg",
        size_n_values=11,
        size_values=None,
        n_sigma=1,
        n_sigma_ul=2,
        selection_optional=None,
        fit=None,
        reoptimize=False,
    ):
        self.energy_edges = energy_edges

        if fit is None:
            fit = Fit(confidence_opts={"backend": "scipy"})

        self.size_values = Angle(size_values) if size_values is not None else None
        self.size_min = Angle(size_min)
        self.size_max = Angle(size_max)
        self.size_n_values = size_n_values
        self.source = source
        super().__init__(
            null_value=1e-3,
            n_sigma=n_sigma,
            n_sigma_ul=n_sigma_ul,
            selection_optional=selection_optional,
            fit=fit,
            reoptimize=reoptimize,
        )

    def run(self, datasets):
        """Run."""
        datasets = Datasets(datasets)
        # find extension parameter
        model = datasets.models[self.source].spatial_model

        if hasattr(model, "sigma"):
            self.size_parameter = model.sigma
        elif hasattr(model, "r_0"):
            self.size_parameter = model.r_0
        elif hasattr(model, "radius"):
            self.size_parameter = model.radius
        else:
            raise ValueError(f"Cannot find size parameter on model {self.source}")

        rows = []

        for energy_min, energy_max in progress_bar(
            zip(self.energy_edges[:-1], self.energy_edges[1:]), desc="Energy bins"
        ):
            datasets_sliced = datasets.slice_by_energy(
                energy_min=energy_min, energy_max=energy_max
            )
            datasets_sliced = Datasets(
                [_.to_image(name=_.name) for _ in datasets_sliced]
            )
            datasets_sliced.models = datasets.models
            row = self.estimate_size(datasets_sliced)
            rows.append(row)
        return rows

    def estimate_size(self, datasets):
        """Estimate size for a given energy range.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets` or list of `~gammapy.datasets.MapDataset`
            Map datasets.

        Returns
        -------
        result : dict
            Dict with results for the extension measurement.
        """
        datasets = Datasets(datasets)

        if self.size_values:
            self.size_parameter.scan_values = self.size_values.to_value(
                self.size_parameter.unit
            )
        self.size_parameter.scan_min = self.size_min.to_value(self.size_parameter.unit)
        self.size_parameter.scan_max = self.size_max.to_value(self.size_parameter.unit)
        self.size_parameter.scan_n_values = self.size_n_values

        result = super().run(datasets, self.size_parameter)
        return result
