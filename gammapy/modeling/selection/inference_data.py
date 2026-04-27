# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from gammapy.modeling.parameter import Parameters


def inference_data_from_ultranest(sampler_results, weighted=False):
    """
     Convert UltraNest result dictionary to Xarray DataTree.

     Parameters
     ----------
     sampler_results : dict
         The result dictionary returned by `ReactiveNestedSampler.run()`.
     weighted : bool
         If True, uses the weighted samples (more accurate) otherwise use directly the unweighted samples.

     Returns
     -------
    inferencedata : `xarray.DataTree`
        Returns an xarray.DataTree instance
    """

    import numpy as np
    import arviz as az
    import xarray as xr

    var_names = sampler_results["paramnames"]

    if weighted:
        ws = sampler_results["weighted_samples"]

        posterior = xr.Dataset(
            {
                var: (["chain", "draw"], ws["points"][:, i][np.newaxis, :])
                for i, var in enumerate(var_names)
            }
        )

        unconstrained_posterior = xr.Dataset(
            {
                var: (["chain", "draw"], ws["upoints"][:, i][np.newaxis, :])
                for i, var in enumerate(var_names)
            }
        )

        sample_stats = xr.Dataset(
            {
                "weights": (["chain", "draw"], ws["weights"][np.newaxis, :]),
                "log_likelihood": (["chain", "draw"], ws["logl"][np.newaxis, :]),
            }
        )

        return az.from_dict(
            dict(
                posterior=posterior,
                unconstrained_posterior=unconstrained_posterior,
                sample_stats=sample_stats,
            )
        )

    ws = sampler_results["samples"]

    posterior = xr.Dataset(
        {
            var: (["chain", "draw"], ws[:, i][np.newaxis, :])
            for i, var in enumerate(var_names)
        }
    )

    return az.from_dict(dict(posterior=posterior))


def inference_data_from_sampler(
    results,
    datasets,
    backend="ultranest",
    n_posterior_samples=None,
    n_prior_samples=None,
    random_seed=42,
    predictives=True,
):
    """
    Convert Sampler results to an xarray DataTree object with optional resampling and prior inclusion.

    Parameters
    ----------
    results : `~gammapy.modeling.SamplerResult`
        The sampler results and model information.
    datasets: `gammapy.datasets.Datasets`
        Datasets that were used to obtain the results.
    backend : {"ultranest"}
        Global backend used for sampler. Default is "ultranest".
        UltraNest: Most options can be found in the
        `UltraNest doc <https://johannesbuchner.github.io/UltraNest/>`__.
    n_posterior_samples : int, optional
        Number of samples to generate after resampling the posterior to take into account weights.
        Default is None, which use the unweighted samples from ultranest.
     n_prior_samples : int, optional
        If provided, generates this number of samples from the prior distribution using the model's
        prior transform and include them in the 'prior' group of the DataTree.
    random_seed : int, optional
        Seed for reproducibility when resampling posterior or generating prior samples. Default is 42.
    predictives : bool, optional
        If True, computes predicted counts and pointwise likelihood matrix. Default is True.

    Returns
    -------
    inference_data : `xarray.DataTree`
        An DataTree object containing posterior samples, optionally resampled,
        prior samples (if requested), and log-evidence attributes ('logz' and 'logzerr').
    """
    if backend == "ultranest":
        inference_data_constructor = inference_data_from_ultranest
    else:
        raise ValueError(f"Only ultranest backend is supported, got {backend}")

    sampler_results = results.sampler_results
    if n_posterior_samples is None:
        inference_data = inference_data_constructor(sampler_results)
    else:
        inference_data = inference_data_constructor(sampler_results, weighted=True)
        inference_data = resample_posterior(
            inference_data, n_samples=n_posterior_samples, random_seed=random_seed
        )  # unweighting

    # Add prior  group
    if n_prior_samples is not None:
        inference_data = add_prior_samples(
            inference_data, results, n_prior_samples, random_seed
        )
        if predictives:
            inference_data = add_sample_wise_quantities(
                inference_data, datasets, "prior"
            )

    # Add posterior and log likelihood groups
    if predictives:
        inference_data = add_sample_wise_quantities(
            inference_data, datasets, "posterior"
        )
    return inference_data


def resample_posterior(inference_data, n_samples=None, random_seed=42):
    """
    Resample posterior samples from an DataTree object using importance weights.

    Parameters
    ----------
    inference_data : `xarray.DataTree`
        The input DataTree object with posterior and sample_stats["weights"].
    n_samples : int, optional
        Number of resampled draws. If None, uses the number of original draws.
    random_seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    inferencedata : `xarray.DataTree`
        A new DataTree object with resampled posterior and sample_stats.
    """

    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(random_seed)

    # Extract posterior samples and weights
    posterior = inference_data.posterior
    weights = inference_data.sample_stats["weights"].values[0]  # shape: (draw,)
    weights = weights / weights.sum()  # normalize

    n_draws = posterior.sizes["draw"]
    n_samples = n_samples or n_draws

    # Resample indices
    resampled_idx = rng.choice(n_draws, size=n_samples, replace=True, p=weights)

    # Resample each group
    resampled_groups = {}
    for group in list(inference_data.children):
        dataset = getattr(inference_data, group)
        if group in ["prior", "prior_predictive", "observed_data"]:
            resampled_groups[group] = dataset
        else:
            resampled_data = {}
            for var in dataset.data_vars:
                values = dataset[var].values
                # Resample along draw dimension (axis=1)
                resampled_values = values[:, resampled_idx]
                resampled_data[var] = (["chain", "draw"], resampled_values)
            resampled_groups[group] = xr.Dataset(resampled_data)

    # Create new DataTree with resampled groups
    return az.from_dict(resampled_groups)


def generate_prior_samples(parameters, n_prior_samples=1000, random_seed=42):
    """Generate prior samples. This function draws samples from the prior distributions of the model
    parameters using inverse transform sampling.

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters` or list of `~gammapy.modeling.Parameter`
        The list of model parameters with priors to sample.
    n_prior_samples : int, optional
        Number of prior samples to generate. If None, no samples are added.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility. Default is 42.

    Returns
    -------
    prior_samples : `numpy.array`
        An array with dimension `n_prior_samples` times len(parameters).
    """

    def _prior_inverse_cdf(values):
        """Returns a list of model parameters for a given list of values (that are bound in [0,1])."""
        if None in Parameters(parameters).prior:
            raise ValueError(
                "Some parameters have no prior set. You need priors on all parameters."
            )
        return [par.prior._inverse_cdf(val) for par, val in zip(parameters, values)]

    # Generate uniform samples in unit cube
    rng = np.random.default_rng(random_seed)
    unit_cube_samples = rng.uniform(size=(n_prior_samples, len(parameters)))

    # Transform to physical space using the prior transform
    prior_samples = np.array([_prior_inverse_cdf(u) for u in unit_cube_samples])
    return prior_samples


def add_prior_samples(inference_data, results, n_prior_samples=None, random_seed=42):
    """
    Generate and add prior samples to an Xarray DataTree object.

    This function draws samples from the prior distributions of the model
    parameters using inverse transform sampling. The samples are added to
    the `prior` group of the provided `DataTree` object.

    Parameters
    ----------
    inference_data : `xarray.DataTree`
        The inference data object to which the prior samples will be added.
    results : `SamplerResult`
        The sampler result object containing the model and its priors.
    n_prior_samples : int, optional
        Number of prior samples to generate. If None, no samples are added.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility. Default is 42.

    Raises
    ------
    `ValueError`
        If any model parameter does not have a prior defined.
    """
    import arviz as az
    import xarray as xr

    parameters = results.models.parameters.free_unique_parameters
    prior_samples = generate_prior_samples(parameters, n_prior_samples, random_seed)

    data = {child: getattr(inference_data, child) for child in inference_data.children}

    # Create prior group
    prior = xr.Dataset(
        {
            name: (["chain", "draw"], prior_samples[:, i][np.newaxis, :])
            for i, name in enumerate(parameters.names)
        }
    )
    data["prior"] = prior
    return az.from_dict(data)


def add_sample_wise_quantities(inference_data, datasets, group_name):
    """
    Compute sample-wise quantities such as log-likelihood, log-prior, and predicted counts.

    This function evaluates the model for each sample in the specified group
    (posterior or prior) and computes the corresponding log-likelihood,
    log-prior, and predicted counts. These are added to the `DataTree`
    object under appropriate groups.

    Parameters
    ----------
    inference_data : `xarray.DataTree`
        The inference data object containing posterior or prior samples.
    datasets : `Datasets`
        The datasets used for computing likelihood and predictions.
    group_name : {"posterior", "prior"}
        The group of samples to evaluate.

    Notes
    -----
    - Assumes that the `stat_array()` method returns `-2 * log-likelihood`.
    - The function modifies the `DataTree` object in-place.
    """
    import arviz as az
    import xarray as xr

    group = getattr(inference_data, group_name)
    n_samples = group.sizes["draw"]
    log_likelihood_matrix = []
    log_prior_matrix = []
    npred_matrix = []
    with datasets.models.parameters.restore_status():
        for idx in range(
            n_samples
        ):  # this should be vectorialized, or parallelize or gammapy should dump stat_array on disk
            for p in datasets.models.parameters.free_parameters:
                p.value = group[p.name][0, idx].item()

            if group_name == "posterior":
                stat_array = (
                    np.hstack([d.stat_array()[d.mask] for d in datasets]) / -2.0
                )  # assuming we have stat as -2lnL
                prior_stat_sum = np.sum(
                    [
                        p.prior._random_variable.logpdf(p.value)
                        for p in datasets.models.parameters.free_unique_parameters
                    ]
                )
                log_likelihood_matrix.append(stat_array)
                log_prior_matrix.append(prior_stat_sum)
            npred = np.hstack([d.npred().data[d.mask] for d in datasets])
            npred_matrix.append(npred)

        if group_name == "posterior":
            log_likelihood_matrix = np.array(log_likelihood_matrix)[np.newaxis, :, :]
            log_prior_matrix = np.array(log_prior_matrix)[np.newaxis, :]
            counts = np.hstack([d.counts.data[d.mask] for d in datasets])
        npred_matrix = np.array(npred_matrix)[np.newaxis, :, :]

    data = {child: getattr(inference_data, child) for child in inference_data.children}

    if group_name == "posterior":
        log_likelihood = xr.Dataset(
            {"log_likelihood": (["chain", "draw", "pixel"], log_likelihood_matrix)}
        )
        data["log_likelihood"] = log_likelihood

        log_prior = xr.Dataset({"log_prior": (["chain", "draw"], log_prior_matrix)})
        data["log_prior"] = log_prior

        posterior_predictive = xr.Dataset(
            {"counts": (["chain", "draw", "pixel"], npred_matrix)}
        )
        data["posterior_predictive"] = posterior_predictive

        observed_data = xr.Dataset({"counts": (["pixel"], counts)})
        data["observed_data"] = observed_data

    elif group_name == "prior":
        prior_predictive = xr.Dataset(
            {"counts": (["chain", "draw", "pixel"], npred_matrix)}
        )
        data["prior_predictive"] = prior_predictive
    return az.from_dict(data)
