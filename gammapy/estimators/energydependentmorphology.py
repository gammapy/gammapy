# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Implementation of energy-dependent morphology estimator tool."""

import numpy as np
from gammapy.datasets import Datasets
from gammapy.modeling import Fit
from gammapy.modeling.models import FoVBackgroundModel, Models
from gammapy.modeling.selection import TestStatisticNested
from gammapy.stats.utils import ts_to_sigma
from .core import Estimator

__all__ = ["weighted_chi2_parameter", "EnergyDependentMorphologyEstimator"]


def weighted_chi2_parameter(results_edep, parameters=["sigma"]):
    r"""Calculate the weighted chi-squared value for the parameters of interest.

    The chi-squared parameter is defined as follows:

    .. math::
        \chi^2 = \sum_i \frac{(x_i - \bar{\mu})^2}{\sigma_i ^ 2}

    where the :math:`x_i` and :math:`\sigma_i` are the value and error of the
    parameter of interest, and the weighted mean is

    .. math::
        \bar{\mu} = \sum_i \frac{(x_i/ \sigma_i ^ 2)}{(1/\sigma_i ^ 2)}


    Parameters
    ----------
    result_edep : `dict`
        Dictionary of results for the energy-dependent estimator.
    parameters : list of str, optional
        The model parameters to calculate the chi-squared value for.
        Default is ["sigma"].

    Returns
    -------
    chi2_result : `dict`
        Dictionary with the chi-squared values for the parameters of interest.

    Notes
    -----
    This chi-square should be utilised with caution as it does not take into
    account any correlation between the parameters.
    To properly utilise the chi-squared parameter one must ensure each of the parameters
    are independent, which cannot be guaranteed in this use case.

    """
    chi2_value = []
    df = []
    for parameter in parameters:
        values = results_edep[parameter][1:]
        errors = results_edep[f"{parameter}_err"][1:]
        weights = 1 / errors**2
        avg = np.average(values, weights=weights)
        chi2_value += [np.sum((values - avg) ** 2 / errors**2).to_value()]
        df += [len(values) - 1]

    significance = [ts_to_sigma(chi2_value[i], df[i]) for i in range(len(chi2_value))]

    chi2_result = {}
    chi2_result["parameter"] = parameters
    chi2_result["chi2"] = chi2_value
    chi2_result["df"] = df
    chi2_result["significance"] = significance

    return chi2_result


class EnergyDependentMorphologyEstimator(Estimator):
    """Test if there is any energy-dependent morphology in a map dataset for a given set of energy bins.

    Parameters
    ----------
    energy_edges : list of `~astropy.units.Quantity`
        Energy edges for the energy-dependence test.
    source : str or int
        For which source in the model to compute the estimator.
    fit : `~gammapy.modeling.Fit`, optional
        Fit instance specifying the backend and fit options.
        If None, the fit backend default is minuit.
        Default is None.

    Examples
    --------
    For a usage example see :doc:`/tutorials/analysis-3d/energy_dependent_estimation` tutorial.
    """

    tag = "EnergyDependentMorphologyEstimator"

    def __init__(self, energy_edges, source=0, fit=None):
        self.energy_edges = energy_edges
        self.source = source
        self.num_energy_bands = len(self.energy_edges) - 1

        if fit is None:
            fit = Fit()

        self.fit = fit

    def _slice_datasets(self, datasets):
        """Calculate a dataset for each energy slice.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Input datasets to use.

        Returns
        -------
        slices_src : `~gammapy.datasets.Datasets`
            Sliced datasets.
        """
        model = datasets.models[self.source]

        filtered_names = [name for name in datasets.models.names if name != self.source]
        other_models = Models()
        for name in filtered_names:
            other_models.append(datasets.models[name])

        slices_src = Datasets()
        for i, (emin, emax) in enumerate(
            zip(self.energy_edges[:-1], self.energy_edges[1:])
        ):
            for dataset in datasets:
                sliced_src = dataset.slice_by_energy(
                    emin, emax, name=f"{self.source}_{i}"
                )
                bkg_sliced_model = FoVBackgroundModel(dataset_name=sliced_src.name)
                sliced_src.models = [
                    model.copy(name=f"{sliced_src.name}-model"),
                    *other_models,
                    bkg_sliced_model,
                ]
                slices_src.append(sliced_src)
        return slices_src

    def _estimate_source_significance(self, datasets):
        """Estimate the significance of the source above the background.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Input datasets to use.

        Returns
        -------
        result_bkg_src : `dict`
            Dictionary with the results of the null hypothesis with no source, and alternative
            hypothesis with the source added in. Entries are:

                * "Emin" : the minimum energy of the energy band
                * "Emax" : the maximum energy of the energy band
                * "delta_ts" : difference in ts
                * "df" : the degrees of freedom between null and alternative hypothesis
                * "significance" : significance of the result

        """
        slices_src = self._slice_datasets(datasets)

        # Norm is free and fit
        test_results = []
        for sliced in slices_src:
            parameters = [
                param
                for param in sliced.models[
                    f"{sliced.name}-model"
                ].parameters.free_parameters
            ]
            null_values = [0] + [
                param.value
                for param in sliced.models[
                    f"{sliced.name}-model"
                ].spatial_model.parameters.free_parameters
            ]

            test = TestStatisticNested(
                parameters=parameters,
                null_values=null_values,
                n_sigma=-np.inf,
                fit=self.fit,
            )
            test_results.append(test.run(sliced))

        delta_ts_bkg_src = [_["ts"] for _ in test_results]
        df_src = [
            len(_["fit_results"].parameters.free_parameters.names) for _ in test_results
        ]
        df_bkg = 1
        df_bkg_src = df_src[0] - df_bkg
        sigma_ts_bkg_src = ts_to_sigma(delta_ts_bkg_src, df=df_bkg_src)

        # Prepare results dictionary for signal above background
        result_bkg_src = {}

        result_bkg_src["Emin"] = self.energy_edges[:-1]
        result_bkg_src["Emax"] = self.energy_edges[1:]
        result_bkg_src["delta_ts"] = delta_ts_bkg_src
        result_bkg_src["df"] = [df_bkg_src] * self.num_energy_bands
        result_bkg_src["significance"] = [elem for elem in sigma_ts_bkg_src]

        return result_bkg_src

    def estimate_energy_dependence(self, datasets):
        """Estimate the potential of energy-dependent morphology.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Input datasets to use.

        Returns
        -------
        results : `dict`
            Dictionary with results of the energy-dependence test. Entries are:

                * "delta_ts" : difference in ts between fitting each energy band individually (sliced fit) and the joint fit
                * "df" : the degrees of freedom between fitting each energy band individually (sliced fit) and the joint fit
                * "result" : the results for the fitting each energy band individually (sliced fit) and the joint fit
        """
        model = datasets.models[self.source]

        # Calculate the individually sliced components
        slices_src = self._slice_datasets(datasets)
        results_src = []
        for sliced in slices_src:
            results_src.append(self.fit.run(sliced))

        results_src_total_stat = [result.total_stat for result in results_src]
        free_x, free_y = np.shape(
            [result.parameters.free_parameters.names for result in results_src]
        )
        df_src = free_x * free_y

        # Calculate the joint fit
        parameters = model.spatial_model.parameters.free_parameters.names
        slice0 = slices_src[0]
        for i, slice_j in enumerate(slices_src[1:]):
            for param in parameters:
                setattr(
                    slice_j.models[f"{self.source}_{i+1}-model"].spatial_model,
                    param,
                    slice0.models[f"{self.source}_0-model"].spatial_model.parameters[
                        param
                    ],
                )
        result_joint = self.fit.run(slices_src)

        # Compare fit of individual energy slices to the results with joint fit
        delta_ts_joint = result_joint.total_stat - np.sum(results_src_total_stat)
        df_joint = len(slices_src.parameters.free_parameters.names)
        df = df_src - df_joint

        # Prepare results dictionary
        joint_values = [result_joint.parameters[param].value for param in parameters]
        joint_errors = [result_joint.parameters[param].error for param in parameters]

        parameter_values = np.empty((len(parameters), self.num_energy_bands))
        parameter_errors = np.empty((len(parameters), self.num_energy_bands))
        for i in range(self.num_energy_bands):
            parameter_values[:, i] = [
                results_src[i].parameters[param].value for param in parameters
            ]
            parameter_errors[:, i] = [
                results_src[i].parameters[param].error for param in parameters
            ]

        result = {}

        result["Hypothesis"] = ["H0"] + ["H1"] * self.num_energy_bands

        result["Emin"] = np.append(self.energy_edges[0], self.energy_edges[:-1])
        result["Emax"] = np.append(self.energy_edges[-1], self.energy_edges[1:])

        units = [result_joint.parameters[param].unit for param in parameters]

        # Results for H0 in the first row and then H1 -- i.e. individual bands in other rows
        for i in range(len(parameters)):
            result[f"{parameters[i]}"] = np.append(
                joint_values[i] * units[i], parameter_values[i] * units[i]
            )
            result[f"{parameters[i]}_err"] = np.append(
                joint_errors[i] * units[i], parameter_errors[i] * units[i]
            )

        return dict(delta_ts=delta_ts_joint, df=df, result=result)

    def run(self, datasets):
        """Run the energy-dependence estimator.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Input datasets to use.

        Returns
        -------
        results : `dict`
            Dictionary with the various energy-dependence estimation values.
        """
        if not isinstance(datasets, Datasets) or datasets.is_all_same_type is False:
            raise ValueError("Unsupported datasets type.")

        results = {}
        results["energy_dependence"] = self.estimate_energy_dependence(datasets)
        results["src_above_bkg"] = self._estimate_source_significance(datasets)

        return results
