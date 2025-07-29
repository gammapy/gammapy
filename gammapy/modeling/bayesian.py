# Licensed under a 3-clause BSD style license - see LICENSE.rst
from IPython.display import display
import numpy as np
from astropy.table import Table, vstack, Column
from scipy.stats import gaussian_kde
from .inference_data import inference_data_from_sampler


class BayesianModelSelection:
    """
    Run Bayesian inference for a set of alternative models.

    Parameters
    ----------
    datasets : `Datasets`
        The datasets on which to perform the model fitting.
    alternative_models : dict
        Dictionary of model names and corresponding `Models` objects.

    Returns
    -------
    result : `BayesianModelSelectionResult`
        Object containing the results of the model comparisons.
    """

    def __init__(self, sampler, prosterior_reduction_factor=4, n_prior_samples=1000):
        self.sampler = sampler
        self.prosterior_reduction_factor = prosterior_reduction_factor
        self.n_prior_samples = n_prior_samples

    def run(self, datasets, alternative_models):
        results = {}
        for models_name, models in alternative_models.items():
            print(f"Evaluating {models_name}")
            datasets.models = models
            sampler_results = self.sampler.run(datasets)
            results[models_name] = InferenceResult(
                models_name,
                sampler_results,
                datasets,
                self.prosterior_reduction_factor,
                self.n_prior_samples,
            )
            display(results[models_name].parameters_table())
            print(results[models_name])
        return BayesianModelSelectionResult(results)


class BayesianModelSelectionResult:
    """
    Container for results of multiple Bayesian model fits.

    Provides utilities to compare models and compute differences
    in statistical metrics.

    Parameters
    ----------
    results_dict : dict
        Dictionary of model names and `ExtendedSamplerResult` objects.
    """

    def __init__(self, results_dict):
        self._results_dict = results_dict

    @property
    def models_names(self):
        return list(self._results_dict.keys())

    def __getitem__(self, key):
        return self._results_dict[key]

    def stats_table(self, format=".3f"):
        """
        Statistics summary table for all models.


        Parameters
        ----------
        format : str, optional
            Format string for numerical columns
            Default is ".3f"

        Returns
        -------
        table : `~astropy.table.Table`
            Table with statistics for each model.
        """

        tables = []
        for results in self._results_dict.values():
            tables.append(results.stats_table(format=format))
        return vstack(tables)

    def stats_difference_table(self, reference_models_name):
        """
        Compute difference in statistics relative to a reference model.

        Parameters
        ----------
        reference_models_name : str
            Name of the model to use as reference (H0).

        Returns
        -------
        table : `~astropy.table.Table`
            Table with differences in statistics (H0 - H1).
        """

        table = self.stats_table()
        idx = np.where(table["Model (prior)"] == reference_models_name)[0][0]
        diff = Table()
        for col in table.colnames:
            if table[col].dtype.kind in "iuf":  # numeric types only
                diff[col] = table[col][idx] - table[col]
            else:
                label = [
                    f"H0: {table[col][idx]} - H1: {table[col][k]}"
                    for k in range(len(table))
                ]
                diff[col] = Column(label)
        for col in [
            "logz error",
            "elpd error (waic)",
            "elpd error (loo)",
            "good Pareto k fraction",
        ]:
            diff.remove_column(col)
        diff.remove_row(idx)
        return diff

    @property
    def elpds_waic(self):
        """Expected log predictive density using WAIC."""
        return {
            model_name: results.elpd_waic
            for model_name, results in self._results_dict.items()
        }

    @property
    def elpds_loo(self):
        """Expected log predictive density using PSIS-LOO-CV."""
        return {
            model_name: results.elpd_loo
            for model_name, results in self._results_dict.items()
        }

    def compare(self, **kwargs):
        """
        Compare models using ArviZ's model comparison utilities.

        This method extracts the LOO-CV expected log predictive densities (ELPDs)
        from each model result and passes them to `arviz.compare` for statistical
        comparison. By default the method is set to 'BB-pseudo-BMA' which performs
        a Bayesian bootstrap to compute standard errors, and weights.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to `arviz.compare`.

        Returns
        -------
        comparison : `~pandas.DataFrame`
            A DataFrame with model comparison metrics such as ELPD differences,
            standard errors, and weights.
        """
        import arviz as az

        kwargs.setdefault("method", "BB-pseudo-BMA")
        return az.compare(self.elpds_loo, **kwargs)

    def prior_sensitivity_table(self):
        "Display prior and likelihood sentivity table computed from power scaling for each model"
        for name in self.models_names:
            print(name)
            display(self[name].prior_sensitivity_table())


class InferenceResult:
    """
    Container for the results of a Bayesian model fit.

    Stores the sampler output, inference data, and computed statistics
    for a given model.

    Parameters
    ----------
    models_name : str
        Name of the model.
    sampler_results : `SamplerResult`
        Output from the sampler run.
    datasets : `Datasets`
        The datasets used for the fit.
    prosterior_reduction_factor : int, optional
        Factor to reduce posterior samples for efficiency.
    n_prior_samples : int, optional
        Number of prior samples to generate.
    """

    def __init__(
        self,
        models_name,
        sampler_results,
        datasets,
        prosterior_reduction_factor=4,
        n_prior_samples=1000,
    ):
        import arviz as az

        self._models_name = models_name
        self._models = sampler_results.models
        self._sampler_results = sampler_results.sampler_results

        if prosterior_reduction_factor > 1:
            n_samples_redu = int(
                np.sum(self.sampler_results["weighted_samples"]["weights"] > 0)
                / prosterior_reduction_factor
            )
        else:
            n_samples_redu = None
        self._idata = inference_data_from_sampler(
            sampler_results,
            datasets,
            n_prosterior_samples=n_samples_redu,
            n_prior_samples=n_prior_samples,
        )
        self._parameters_table = compute_parameters_values(
            self._idata, sampler_results, group_name="posterior"
        )
        if n_prior_samples is not None:
            self._parameters_table_prior = compute_parameters_values(
                self._idata, sampler_results, group_name="prior"
            )
        self._elpd_waic = az.waic(self._idata, scale="deviance")
        self._elpd_loo = az.loo(self._idata, scale="deviance")

    @property
    def models(self):
        return self._models

    @property
    def models_name(self):
        return self._models_name

    @property
    def sampler_results(self):
        """Sampler results dictionary."""
        return self._sampler_results

    @property
    def idata(self):
        """ArviZ `InferenceData` object with unweighted posterior samples."""
        return self._idata

    @property
    def dof(self):
        """Number of free unique parameters (degrees of freedom)."""
        return len(self.models.parameters.free_unique_parameters)

    @property
    def logz(self):
        """Log-evidence (logZ) from the sampler."""
        return self.sampler_results["logz"]

    @property
    def logz_err(self):
        """Uncertainty on the log-evidence."""
        return self.sampler_results["logzerr"]

    @property
    def logl(self):
        """Maximum log-likelihood value."""
        return self.sampler_results["maximum_likelihood"]["logl"]

    @property
    def aic(self):
        """Akaike Information Criterion (AIC)."""
        return -2 * self.sampler_results["maximum_likelihood"]["logl"] + 2 * self.dof

    @property
    def elpd_waic(self):
        """Expected log predictive density using WAIC."""
        return self._elpd_waic

    @property
    def elpd_loo(self):
        """Expected log predictive density using PSIS-LOO-CV."""
        return self._elpd_loo

    @property
    def loo_good_pareto_k_fraction(self):
        """
        Fraction of PSIS-LOO-CV points with good Pareto k diagnostics.

        Returns
        -------
        fraction : float
            Fraction of points with Pareto k < threshold.
        """

        return np.sum(self.elpd_loo.pareto_k < self.elpd_loo.good_k) / len(
            self.elpd_loo.pareto_k
        )

    def parameters_table(self, group="posterior"):
        """
        Summary table of parameter estimates.

        Parameters
        ----------
        group : {"posterior", "prior"}, optional
            Which group to summarize.

        Returns
        -------
        table : `~astropy.table.Table`
            Table with mean, median, mode, and value at the max-likelihood.
        """
        if group == "posterior":
            return self._parameters_table
        elif group == "prior":
            return self._parameters_table_prior

    def stats_table(self, format=".3f"):
        """
        Statistics summary table for all models.


        Parameters
        ----------
        format : str, optional
            Format string for numerical columns
            Default is ".3f"

        Returns
        -------
        table : `~astropy.table.Table`
            Table with statistics for each model.
        """
        values = [
            [f"{self.models_name}"],
            [-2 * self.logz],
            [-2 * self.logl],
            [self.aic],
            [self.elpd_waic.elpd_waic],
            [self.elpd_loo.elpd_loo],
            [self.dof],
            [self.elpd_waic.p_waic],
            [self.elpd_loo.p_loo],
            [2 * self.logz_err],
            [self.elpd_waic.se],
            [self.elpd_loo.se],
            [self.loo_good_pareto_k_fraction],
        ]
        names = (
            "Model (prior)",
            "-2logz",
            "-2logl",
            "AIC",
            "elpd (waic)",
            "elpd (loo)",
            "dof",
            "waic eff. dof",
            "loo eff. dof",
            "logz error",
            "elpd error (waic)",
            "elpd error (loo)",
            "good Pareto k fraction",
        )
        table = Table(values, names=names)
        for name in names[1:]:
            table[name].format = format
        return table

    def prior_sensitivity_table(self):
        "Prior and likelihood sentivity table computed from power scaling."
        import arviz.preview as azp  # contains methods from future version

        return azp.psense_summary(self.idata)

    @property
    def priors(self):
        "Dict of random variable objects used to generate the prior"
        return {
            p.name: p.prior.random_variable
            for p in self.models.parameters.free_unique_parameters
        }

    def __str__(self):
        s = "Statistics summary in deviance scale : -2log(score)\n"
        s += "A lower deviance corresponds to a model with better predictive accuracy. \n"
        s += f"-2log(Z)  : {-2*self.logz}" + " +/- " + f"{2*self.logz_err}" + "\n"
        s += f"-2log(L)  : {-2*self.logl}" + "\n"
        s += f"AIC       : {self.aic}, dof {self.dof}" + "\n"
        s += (
            f"elpd_waic : {self.elpd_waic.elpd_waic}"
            + "+/-"
            + f"{self.elpd_waic.se}, effective dof {self.elpd_waic.p_waic}"
            + "\n"
        )
        s += (
            f"elpd_loo  : {self.elpd_loo.elpd_loo},"
            + "+/-"
            + f"{self.elpd_loo.se}, effective dof {self.elpd_loo.p_loo}"
            + "\n"
        )
        return s


def compute_parameters_values(inference_data, results, group_name="posterior"):
    """
    Compute summary statistics for model parameters from an InferenceData object.

    This function calculates the mean, median, mode, and (if available) the
    value at the maximum likelihood for each parameter in the specified group
    (posterior or prior). The mode is estimated using a Gaussian KDE.

    Parameters
    ----------
    inference_data : `arviz.InferenceData`
        The inference data object containing posterior or prior samples.
    results : `SamplerResult`
        The sampler result object containing maximum likelihood estimates.
    group_name : str, optional
        The group to extract parameters from. Can be "posterior" or "prior".

    Returns
    -------
    summary_table : `~astropy.table.Table`
        Table summarizing the mean, median, mode, and value at maximum likelihood
        (if available) for each parameter.
    """

    group = getattr(inference_data, group_name)
    maximum_likelihood = results.sampler_results["maximum_likelihood"]
    # could use inference_data and apply weight in calculations
    # for mean = np.sum(weights*samples)/np.sum(weights)
    # for median it's more compliated https://gist.github.com/robbibt/c7ec5f0cb3e4e0cee5ed3156bcb666de)
    # for mode use : kde = gaussian_kde(samples, weights=weights)
    # Prepare lists to store results
    param_names = []
    means = []
    medians = []
    modes = []
    maxlnl = []
    # Iterate over each parameter
    for k, var in enumerate(group.data_vars):
        samples = group[var].values.flatten()
        param_names.append(var)
        means.append(np.mean(samples))
        medians.append(np.median(samples))
        if group_name == "posterior":
            maxlnl.append(maximum_likelihood["point"][k])
        else:
            maxlnl.append(None)
        # Estimate mode using KDE
        kde = gaussian_kde(samples)
        x_vals = np.linspace(np.min(samples), np.max(samples), 1000)
        mode_val = x_vals[np.argmax(kde(x_vals))]
        modes.append(mode_val)

    summary_table = Table(
        [param_names, means, medians, modes, maxlnl],
        names=("parameter", "mean", "median", "mode", "value at max ln(L)"),
    )

    return summary_table
