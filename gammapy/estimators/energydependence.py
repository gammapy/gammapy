import numpy as np
from gammapy.datasets import Datasets, MapDataset
from gammapy.modeling import Fit
from gammapy.modeling.models import FoVBackgroundModel
from gammapy.modeling.selection import TestStatisticNested
from gammapy.stats.utils import ts_to_sigma
from .core import Estimator


def weighted_chi2_parameter_results(table_edep, parameter="sigma"):
    """Calculate the weighted chi2 value for the parameter of interest"""

    values = table_edep[f"{parameter}"][1:]
    errors = table_edep[f"{parameter}_err"][1:]

    weights = 1 / errors**2
    avg = np.average(values, weights=weights)

    chi2_value = np.sum((values - avg) ** 2 / errors**2)
    df = len(values) - 1
    sigma_value = ts_to_sigma(chi2_value, df)

    chi2_result = {}
    chi2_result[f"chi2 {parameter}"] = [chi2_value]
    chi2_result["df"] = [df]
    chi2_result["significance"] = [sigma_value]

    return chi2_result


class EnergyDependenceEstimator(Estimator):
    """Test if there is any energy-dependent morphology in a map dataset for a given set of energy bins.

    Parameters
    ----------
    energy_edges : `~astropy.units.Quantity`
        Energy edges for the energy-dependence test.
    model : `~gammapy.modeling.model.SkyModel`
        Source model kernel.
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:
            * "src-sig": the significance above the background is
            calculated in each energy band

    Examples
    --------

    """

    tag = "EnergyDependenceEstimator"
    _available_selection_optional = ["src-sig"]

    def __init__(self, energy_edges, model, fit=None, selection_optional=None):

        self.energy_edges = energy_edges
        self.model = model
        self.num_energy_bands = len(self.energy_edges) - 1

        if fit is None:
            fit = Fit(optimize_opts={"print_level": 1})

        self.fit = fit
        self.selection_optional = selection_optional

    def estimate_source_significance(self, dataset):
        """Estimate the significance of the source above the background.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset to use.

        Returns
        -------
        table_bkg_src : `~astropy.table.Table`
            Table with the results of the null hypothesis with no source, and alternative
            hypothesis with the source added in. Entries are:
            * "Emin" : the minimum energy of the energy band
            * "Emax" : the maximum energy of the energy band
            * "delta ts" : difference in ts
            * "df" : the degrees of freedom between null and alternative hypothesis
            * "significance" : significance of the result
        """

        dataset.mask_fit = dataset.counts.geom.energy_mask(
            energy_min=self.energy_edges[0], energy_max=None
        )

        # Calculate the dataset for each energy slice
        slices_src = Datasets()
        for emin, emax in zip(self.energy_edges[:-1], self.energy_edges[1:]):
            sliced_src = dataset.slice_by_energy(emin, emax)
            bkg_sliced_model = FoVBackgroundModel(dataset_name=sliced_src.name)
            sliced_src.models = [self.model.copy(), bkg_sliced_model]
            slices_src.append(sliced_src)

        # Norm is free and fit
        test_results = []
        for sliced in slices_src:
            parameters = [param for param in sliced.models.parameters.free_parameters]
            null_values = [0] + [
                param.value
                for param in sliced.models[0].spatial_model.parameters.free_parameters
            ]

            test = TestStatisticNested(
                parameters=parameters,
                null_values=null_values,
                n_sigma=-np.inf,
                fit=self.fit,
            )
            test_results.append(test.run(sliced))

        delta_ts_bkg_src = [_["ts"] for _ in test_results]
        df_src = np.sum(
            [
                len(_["fit_results"].parameters.free_parameters.names)
                for _ in test_results
            ]
        )
        df_bkg = 1 * self.num_energy_bands
        df_bkg_src = df_src - df_bkg
        sigma_ts_bkg_src = ts_to_sigma(delta_ts_bkg_src, df=df_bkg_src)

        # Prepare results dictionary for signal above background
        result_bkg_src = {}

        result_bkg_src["Emin"] = self.energy_edges[:-1]
        result_bkg_src["Emax"] = self.energy_edges[1:]
        result_bkg_src["delta ts"] = [val for val in delta_ts_bkg_src]
        result_bkg_src["df"] = [df_bkg_src] * self.num_energy_bands
        result_bkg_src["significance"] = [elem for elem in sigma_ts_bkg_src]

        return result_bkg_src

    def estimate_energy_dependence(self, dataset):
        """Estimate the potential of energy-dependent morphology.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset to use.

        Returns
        -------
        results : `dict`
            Dict with results of the energy-dependence test.
            * "delta_ts" : difference in ts between sliced fit and joint fit
            * "df" : the degrees of freedom between sliced fit and joint fit
            * "result" : the results for the sliced fit and joint fit
        """
        dataset.mask_fit = dataset.counts.geom.energy_mask(
            energy_min=self.energy_edges[0], energy_max=None
        )

        # Calculate the individually sliced components
        slices_src = Datasets()
        for emin, emax in zip(self.energy_edges[:-1], self.energy_edges[1:]):
            sliced_src = dataset.slice_by_energy(emin, emax)
            bkg_sliced_model = FoVBackgroundModel(dataset_name=sliced_src.name)
            sliced_src.models = [self.model.copy(), bkg_sliced_model]
            slices_src.append(sliced_src)

        results_src = []
        for sliced in slices_src:
            results_src.append(self.fit.run(sliced))

        results_src_total_stat = [result.total_stat for result in results_src]
        free_x, free_y = np.shape(
            [result.parameters.free_parameters.names for result in results_src]
        )
        df_src = free_x * free_y

        # Calculate the joint fit
        parameters = self.model.spatial_model.parameters.free_parameters.names
        slice0 = slices_src[0]
        for slice_j in slices_src[1:]:
            for param in parameters:
                setattr(
                    slice_j.models[0].spatial_model,
                    param,
                    slice0.models[0].spatial_model.parameters[param],
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

    def run(self, dataset):
        """Run the energy-dependence estimator.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Input dataset to use.

        Returns
        -------
        results : dict
            Dict with the various energy-dependence estimation values.
        """

        if not isinstance(dataset, MapDataset):
            raise ValueError("Unsupported dataset type.")

        results = self.estimate_energy_dependence(dataset)

        if "src-sig" in self.selection_optional:
            results = dict(
                energy_dependence=results,
                src_above_bkg=self.estimate_source_significance(dataset),
            )

        return results
