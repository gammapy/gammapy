# Licensed under a 3-clause BSD style license - see LICENSE.rst
# helper functions for mcmc smapling
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner

__all__ = ["uniform_prior", "run_mcmc", "plot_trace", "plot_corner"]

#TODO: so far only works with a uniform prior on parameters

# Prior functions
def uniform_prior(value, umin, umax):
    """Uniform prior distribution."""
    if umin <= value <= umax:
        return 0.0
    else:
        return -np.inf


def normal_prior(value, mean, sigma):
    """Normal prior distribution."""
    return -0.5 * (2 * np.pi * sigma) - (value - mean) ** 2 / (2.0 * sigma)


# Read/write parameters in the dataset
def model_to_par(dataset):
    """
    Return a tuple of the factor parameters of all
    free parameters in the dataset sky model.
    """
    pars = []
    for p in dataset.parameters.free_parameters:
        pars.append(p.factor)

    return pars


def par_to_model(dataset, pars):
    """Update model in dataset with a list of free parameters factors"""
    for i, p in enumerate(dataset.parameters.free_parameters):
        p.factor = pars[i]


# Compute LogLike associated with prior and data/model evaluation
def lnprior(dataset):
    """
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

    lnprob_priors = lnprior(dataset)

    # dataset.likelihood returns Cash statistics values
    # emcee will maximisise the LogLikelihood so we need -dataset.likelihood
    total_lnprob = -dataset.likelihood() + lnprob_priors

    return total_lnprob


def run_mcmc(dataset, nwalkers=8, nrun=1000, threads=1):
    """
    Run the MCMC sampler.

    Parameters
    ----------
    dataset : `MapDataset`
        A gammapy dataset object. This contains the observed counts cube,
        the exposure cube, the psf cube, and the sky model and model.
        Each free parameter in the sky model is considered as parameter for the MCMC.
    nwalkers: int
        Required integer number of walkers to use in ensemble.
        Minimum is 2*nparam+2, but more than that is usually better.
        Must be even to use MPI mode.
    nrun: int
        Number of steps for walkers. Typically at least a few hundreds (but depends on dimensionality).
        Low nrun (<100?) will underestimate the errors.
        Samples that would populate the distribution are nrun*nwalkers.
        This step can be ~seen as the error estimation step.
    """
    dataset.parameters.autoscale()  # Autoscale parameters
    pars = model_to_par(dataset)  # get a tuple of parameters from dataset
    ndim = len(pars)

    # Initialize walkers in a ball of relative size 0.5% in all dimensions if the
    # parameters have been fit, or to 10% otherwise
    spread = 0.5 / 100
    p0var = np.array([spread * pp for pp in pars])
    p0 = emcee.utils.sample_ball(pars, p0var, nwalkers)

    labels = []
    for par in dataset.parameters.free_parameters:
        labels.append(par.name)
        if (par.min is np.nan) and (par.max is np.nan):
            print(
                    "Warning: no priors have been set for parameter %s\n The MCMC will likely not work !"
                    % (par.name)
            )

    print("List of free parameters:", labels)
    print("{} walkers will run for {} steps".format(nwalkers, nrun))
    print("Parameters init value for 1st walker:", p0[0])
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=[dataset], threads=threads
    )

    for idx, result in enumerate(sampler.sample(p0, iterations=nrun)):
        if (idx + 1) % 100 == 0:
            print("{0:5.0%}".format(idx / nrun))

    return sampler

def plot_trace(sampler, dataset):
    """
    Plot the trace of walkers for every steps
    """
    labels = []
    for par in dataset.parameters.free_parameters:
        labels.append(par.name)

    fig, ax = plt.subplots(len(labels), sharex=True)
    for i in range(len(ax)):
        ax[i].plot(sampler.chain[:, :, i].T, "-k", alpha=0.2)
        ax[i].set_ylabel(labels[i])
    plt.xlabel("Nrun")


def plot_corner(sampler, dataset, nburn=0):
    """
    Corner plot for each parameter explored by the walkers.

    Parameters
    ----------
    sampler : `EnsembleSample`
        Sample instance.

    nburn: int
        Number of runs that will be discarded (burnt) until reaching ~stationary states for walkers.
        Hard to guess. Depends how close to best-fit you are.
        A good nbrun value can be estimated from the trace plot.
        This step can be ~seen as the fitting step.

    """
    labels = [par.name for par in dataset.parameters.free_parameters]

    samples = sampler.chain[:, nburn:, :].reshape((-1, len(labels)))

    corner.corner(
        samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True
    )