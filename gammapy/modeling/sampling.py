# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""MCMC sampling helper functions using ``emcee``."""
import logging
import numpy as np

__all__ = ["uniform_prior", "run_mcmc", "plot_trace", "plot_corner"]

log = logging.getLogger(__name__)


# TODO: so far only works with a uniform prior on parameters
# as there is no way yet to enter min,mean,max in parameters for normal prior
# lnprob() uses a uniform prior. hard coded for now.


def uniform_prior(value, umin, umax):
    """Uniform prior distribution."""
    if umin <= value <= umax:
        return 0.0
    else:
        return -np.inf


def normal_prior(value, mean, sigma):
    """Normal prior distribution."""
    return -0.5 * (2 * np.pi * sigma) - (value - mean) ** 2 / (2.0 * sigma)


def par_to_model(dataset, pars):
    """Update model in dataset with a list of free parameters factors"""
    for i, p in enumerate(dataset.models.parameters.free_parameters):
        p.factor = pars[i]


def ln_uniform_prior(dataset):
    """LogLike associated with prior and data/model evaluation.

    Return probability of parameter values according to prior knowledge.
    Parameter limits should be done here through uniform prior ditributions
    """
    logprob = 0
    for par in dataset.models.parameters.free_parameters:
        logprob += uniform_prior(par.value, par.min, par.max)

    return logprob


def lnprob(pars, dataset):
    """Estimate the likelihood of a model including prior on parameters."""
    # Update model parameters factors inplace
    for factor, par in zip(pars, dataset.models.parameters.free_parameters):
        par.factor = factor

    lnprob_priors = ln_uniform_prior(dataset)

    # dataset.likelihood returns Cash statistics values
    # emcee will maximisise the LogLikelihood so we need -dataset.likelihood
    total_lnprob = -dataset.stat_sum() + lnprob_priors

    return total_lnprob


def run_mcmc(dataset, nwalkers=8, nrun=1000, threads=1):
    """Run the MCMC sampler.

    Parameters
    ----------
    dataset : `~gammapy.modeling.Dataset`
        Dataset
    nwalkers : int
        Number of walkers
    nrun : int
        Number of steps each walker takes
    threads : (optional)
        Number of threads or processes to use

    Returns
    -------
    sampler : `emcee.EnsembleSampler`
        sampler object containing the trace of all walkers.
    """
    import emcee

    dataset.models.parameters.autoscale()  # Autoscale parameters

    # Initialize walkers in a ball of relative size 0.5% in all dimensions if the
    # parameters have been fit, or to 10% otherwise
    # Handle source position spread differently with a spread of 0.1°
    # TODO: the spread of 0.5% below is valid if a pre-fit of the model has been obtained.
    # currently the run_mcmc() doesn't know the status of previous fit.
    p0var = []
    pars = []
    spread = 0.5 / 100
    spread_pos = 0.1  # in degrees
    for par in dataset.models.parameters.free_parameters:
        pars.append(par.factor)
        if par.name in ["lon_0", "lat_0"]:
            p0var.append(spread_pos / par.scale)
        else:
            p0var.append(spread * par.factor)

    ndim = len(pars)
    p0 = emcee.utils.sample_ball(pars, p0var, nwalkers)

    labels = []
    for par in dataset.models.parameters.free_parameters:
        labels.append(par.name)
        if (par.min is np.nan) and (par.max is np.nan):
            log.warning(
                f"Missing prior for parameter: {par.name}.\nMCMC will likely fail!"
            )

    log.info(f"Free parameters: {labels}")

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=[dataset], threads=threads
    )

    log.info(f"Starting MCMC sampling: nwalkers={nwalkers}, nrun={nrun}")
    for idx, result in enumerate(sampler.sample(p0, iterations=nrun)):
        if idx % (nrun / 4) == 0:
            log.info("{:5.0%}".format(idx / nrun))
    log.info("100% => sampling completed")

    return sampler


def plot_trace(sampler, dataset):
    """
    Plot the trace of walkers for every steps

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler object containing the trace of all walkers
    dataset : `~gammapy.modeling.Dataset`
        Dataset
    """
    import matplotlib.pyplot as plt

    labels = [par.name for par in dataset.models.parameters.free_parameters]

    fig, axes = plt.subplots(len(labels), sharex=True)

    for idx, ax in enumerate(axes):
        ax.plot(sampler.chain[:, :, idx].T, "-k", alpha=0.2)
        ax.set_ylabel(labels[idx])

    plt.xlabel("Nrun")
    plt.show()


def plot_corner(sampler, dataset, nburn=0):
    """Corner plot for each parameter explored by the walkers.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler object containing the trace of all walkers
    dataset : `~gammapy.modeling.Dataset`
        Dataset
    nburn : int
        Number of runs to discard, because considered part of the burn-in phase
    """
    from corner import corner

    labels = [par.name for par in dataset.models.parameters.free_parameters]

    samples = sampler.chain[:, nburn:, :].reshape((-1, len(labels)))

    corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
