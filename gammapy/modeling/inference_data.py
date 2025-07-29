# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np


def inference_data_from_ultranest(sampler_results, weighted=False):
    """
    Convert UltraNest result dictionary to ArviZ InferenceData.

    Parameters
    ----------
    result : dict
        The result dictionary returned by `ReactiveNestedSampler.run()`.

    Returns
    -------
    arviz.InferenceData
    """

    import arviz as az
    import xarray as xr

    var_names = sampler_results["paramnames"]

    if weighted:
        ws = sampler_results["weighted_samples"]

        # Posterior samples (1 chain)
        posterior_data = {
            var: (["chain", "draw"], ws["points"][:, i][np.newaxis, :])
            for i, var in enumerate(var_names)
        }
        posterior = xr.Dataset(posterior_data)

        unconstrained_posterior_data = {
            var: (["chain", "draw"], ws["upoints"][:, i][np.newaxis, :])
            for i, var in enumerate(var_names)
        }
        unconstrained_posterior = xr.Dataset(unconstrained_posterior_data)

        # Sample stats
        sample_stats = xr.Dataset(
            {
                "weights": (["chain", "draw"], ws["weights"][np.newaxis, :]),
                "log_likelihood": (["chain", "draw"], ws["logl"][np.newaxis, :]),
            }
        )

        return az.InferenceData(
            posterior=posterior,
            unconstrained_posterior=unconstrained_posterior,
            sample_stats=sample_stats,
        )
    else:
        ws = sampler_results["samples"]

        posterior_data = {
            var: (["chain", "draw"], ws[:, i][np.newaxis, :])
            for i, var in enumerate(var_names)
        }
        posterior = xr.Dataset(posterior_data)

        return az.InferenceData(posterior=posterior)


def inference_data_from_sampler(
    results,
    datasets,
    backend="ultranest",
    n_prosterior_samples=None,
    n_prior_samples=None,
    random_seed=42,
    predictives=True,
):
    """
    Convert Sampler results to an ArviZ InferenceData object with optional resampling and prior inclusion.

    Parameters
    ----------
    backend : {"ultranest"}

    results : gammapy.modeling.SamplerResult
        The sampler results and model informations.
    n_prosterior_samples : int, optional
        Number of samples to generate after resampling the posterior to take into account weights.
        Default is None, which use the unweighted samples from ultranest.
     n_prior_samples : int, optional
        If provided, generate this number of samples from the prior distribution using the model's
        prior transform and include them in the 'prior' group of the InferenceData.
    random_seed : int, optional
        Seed for reproducibility when resampling posterior or generating prior samples.
    predictives : bool, optional
        If True computes predicted counts and pointwise likelihood matrix. Defalut is True.

    Returns
    -------
    arviz.InferenceData
        An InferenceData object containing posterior samples, optionally resampled,
        prior samples (if requested), and log-evidence attributes ('logz' and 'logzerr').
    """
    if backend == "ultranest":
        inference_data_constructor = inference_data_from_ultranest
    else:
        raise ValueError(f"Only ultranest backend is supported, got {backend}")

    sampler_results = results.sampler_results
    if n_prosterior_samples is None:
        inference_data = inference_data_constructor(sampler_results)
    else:
        inference_data = inference_data_constructor(sampler_results, weighted=True)
        inference_data = resample_posterior(
            inference_data, n_samples=n_prosterior_samples, random_seed=random_seed
        )  # unweighting

    # Add prior  group
    if n_prior_samples is not None:
        add_prior_samples(inference_data, results, n_prior_samples, random_seed)
        if predictives:
            add_sample_wise_quantities(inference_data, datasets, "prior")

    # Add posterior and log likelihood groups
    if predictives:
        add_sample_wise_quantities(inference_data, datasets, "posterior")
    return inference_data


def resample_posterior(inference_data, n_samples=None, random_seed=42):
    """
    Resample posterior samples from an InferenceData object using importance weights.

    Parameters
    ----------
    inference_data : arviz.InferenceData
        The input InferenceData object with posterior and sample_stats["weights"].
    n_samples : int, optional
        Number of resampled draws. If None, use the number of original draws.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    arviz.InferenceData
        A new InferenceData object with resampled posterior and sample_stats.
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
    for group in inference_data.groups():
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

    # Create new InferenceData with resampled groups
    return az.InferenceData(**resampled_groups)


def generate_prior_samples(parameters, n_prior_samples=1000, random_seed=42):
    """Generate prior samples. This function draws samples from the prior distributions of the model
    parameters using inverse transform sampling.

    Parameters
    ----------
    parameters : list
        The list of model parameters with priors to sample.
    n_prior_samples : int, optional
        Number of prior samples to generate. If None, no samples are added.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    `numpy.array`
        An array with dimension `n_prior_samples` times len(parameters).
    """

    def _prior_inverse_cdf(values):
        """Returns a list of model parameters for a given list of values (that are bound in [0,1])."""
        if None in parameters:
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
    Generate and add prior samples to an ArviZ InferenceData object.

    This function draws samples from the prior distributions of the model
    parameters using inverse transform sampling. The samples are added to
    the `prior` group of the provided `InferenceData` object.

    Parameters
    ----------
    inference_data : `arviz.InferenceData`
        The inference data object to which the prior samples will be added.
    results : `SamplerResult`
        The sampler result object containing the model and its priors.
    n_prior_samples : int, optional
        Number of prior samples to generate. If None, no samples are added.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility.

    Raises
    ------
    ValueError
        If any model parameter does not have a prior defined.
    """
    import xarray as xr

    parameters = results.models.parameters.free_unique_parameters
    prior_samples = generate_prior_samples(parameters, n_prior_samples, random_seed)

    # Create prior group
    prior = xr.Dataset(
        {
            name: (["chain", "draw"], prior_samples[:, i][np.newaxis, :])
            for i, name in enumerate(parameters.names)
        }
    )
    inference_data.add_groups(prior=prior)


def add_sample_wise_quantities(inference_data, datasets, group_name):
    """
    Compute sample-wise quantities such as log-likelihood, log-prior, and predicted counts.

    This function evaluates the model for each sample in the specified group
    (posterior or prior) and computes the corresponding log-likelihood,
    log-prior, and predicted counts. These are added to the `InferenceData`
    object under appropriate groups.

    Parameters
    ----------
    inference_data : `arviz.InferenceData`
        The inference data object containing posterior or prior samples.
    datasets : `Datasets`
        The datasets used for computing likelihood and predictions.
    group_name : {"posterior", "prior"}
        The group of samples to evaluate.

    Notes
    -----
    - Assumes that the `stat_array()` method returns `-2 * log-likelihood`.
    - The function modifies the `InferenceData` object in-place.
    """
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
                        p.prior.random_variable.logpdf(p.value)
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

    if group_name == "posterior":
        log_ligkelihood = xr.Dataset(
            {"log_ligkelihood": (["chain", "draw", "pixel"], log_likelihood_matrix)}
        )
        inference_data.add_groups(log_likelihood=log_ligkelihood)

        log_prior = xr.Dataset({"log_prior": (["chain", "draw"], log_prior_matrix)})
        inference_data.add_groups(log_prior=log_prior)

        posterior_predictive = xr.Dataset(
            {"counts": (["chain", "draw", "pixel"], npred_matrix)}
        )
        inference_data.add_groups(posterior_predictive=posterior_predictive)

        observed_data = xr.Dataset({"counts": (["pixel"], counts)})
        inference_data.add_groups(observed_data=observed_data)
    elif group_name == "prior":
        prior_predictive = xr.Dataset(
            {"counts": (["chain", "draw", "pixel"], npred_matrix)}
        )
        inference_data.add_groups(prior_predictive=prior_predictive)
