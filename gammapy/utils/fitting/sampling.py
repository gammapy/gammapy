# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""MCMC sampling helper functions using ``emcee``."""
import logging
import numpy as np
import emcee
import corner

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
    for i, p in enumerate(dataset.parameters.free_parameters):
        p.factor = pars[i]


def ln_uniform_prior(dataset):
    """LogLike associated with prior and data/model evaluation.

    Return probability of parameter values according to prior knowledge.
    Parameter limits should be done here through uniform prior ditributions
    """
    logprob = 0
    for par in dataset.parameters.free_parameters:
        logprob += uniform_prior(par.value, par.min, par.max)

    return logprob


def lnprob(pars, dataset):
    """Estimate the likelihood of a model including prior on parameters."""
    # Update model parameters factors inplace
    for factor, par in zip(pars, dataset.parameters.free_parameters):
        par.factor = factor

    lnprob_priors = ln_uniform_prior(dataset)

    # dataset.likelihood returns Cash statistics values
    # emcee will maximisise the LogLikelihood so we need -dataset.likelihood
    total_lnprob = -dataset.likelihood() + lnprob_priors

    return total_lnprob


def run_mcmc(dataset, nwalkers=8, nrun=1000, threads=1):
    """Run the MCMC sampler.

    Parameters
    ----------
    dataset : `gammapy.utils.fitting.Dataset`
        A gammapy dataset object. This contains the observed counts cube,
        the exposure cube, the psf cube, and the sky model and model.
        Each free parameter in the sky model is considered as parameter for the MCMC.
    nwalkers : int
        Required integer number of walkers to use in ensemble.
        Minimum is 2*nparam+2, but more than that is usually better.
        Must be even to use MPI mode.
    nrun : int
        Number of steps for walkers. Typically at least a few hundreds (but depends on dimensionality).
        Low nrun (<100?) will underestimate the errors.
        Samples that would populate the distribution are nrun*nwalkers.
        This step can be ~seen as the error estimation step.
    threads : (optional)
        The number of threads to use for parallelization. If ``threads == 1``,
        then the ``multiprocessing`` module is not used but if
        ``threads > 1``, then a ``Pool`` object is created and calls to
        ``lnpostfn`` are run in parallel.

    Returns
    -------
    sampler : `emcee.EnsembleSampler`
        sampler object containing the trace of all walkers.
    """
    dataset.parameters.autoscale()  # Autoscale parameters
    pars = [par.factor for par in dataset.parameters.free_parameters]
    ndim = len(pars)

    # Initialize walkers in a ball of relative size 0.5% in all dimensions if the
    # parameters have been fit, or to 10% otherwise
    # TODO: the spread of 0.5% below is valid if a pre-fit of the model has been obtained.
    # currently the run_mcmc() doesn't know the status of previous fit.
    spread = 0.5 / 100
    p0var = np.array([spread * pp for pp in pars])
    p0 = emcee.utils.sample_ball(pars, p0var, nwalkers)

    labels = []
    for par in dataset.parameters.free_parameters:
        labels.append(par.name)
        if (par.min is np.nan) and (par.max is np.nan):
            log.warning(
                "Missing prior for parameter: {}.\n"
                "MCMC will likely fail!".format(par.name)
            )

    log.info("Free parameters: {}".format(labels))

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=[dataset], threads=threads
    )

    log.info("Starting MCMC sampling: nwalkers={}, nrun={}".format(nwalkers, nrun))
    for idx, result in enumerate(sampler.sample(p0, iterations=nrun)):
        if idx % (nrun / 4) == 0:
            log.info("{0:5.0%}".format(idx / nrun))
    log.info("100% => sampling completed")

    return sampler


def plot_trace(sampler, dataset):
    """
    Plot the trace of walkers for every steps

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler object containing the trace of all walkers.

    dataset : `gammapy.utils.fitting.Dataset`
        A gammapy dataset object. This contains the observed counts cube,
        the exposure cube, the psf cube, and the sky model and model.
        Each free parameter in the sky model is considered as parameter for the MCMC.

    """
    import matplotlib.pyplot as plt

    labels = []
    for par in dataset.parameters.free_parameters:
        labels.append(par.name)

    fig, axes = plt.subplots(len(labels), sharex=True)
    for i, ax in range(len(axes)):
        ax.plot(sampler.chain[:, :, i].T, "-k", alpha=0.2)
        ax.set_ylabel(labels[i])
    plt.xlabel("Nrun")
    plt.show()

def plot_corner(sampler, dataset, nburn=0):
    """Corner plot for each parameter explored by the walkers.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler object containing the trace of all walkers.
    dataset : `gammapy.utils.fitting.Dataset`
        Dataset
    nburn: int
        Number of runs that will be discarded (burnt) until reaching ~stationary states for walkers.
        Hard to guess. Depends how close to best-fit you are.
        A good nbrun value can be estimated from the trace plot.
        This step can be ~seen as the fitting step.
    """
    labels = [par.name for par in dataset.parameters.free_parameters]

    samples = sampler.chain[:, nburn:, :].reshape((-1, len(labels)))

    corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
