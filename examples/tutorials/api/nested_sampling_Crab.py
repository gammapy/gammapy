"""
Bayesian analysis with nested sampling
======================================

A demonstration of a Bayesian analysis using the nested sampling technique.

"""


######################################################################
# Context
# -------
#
# 1. Bayesian analysis
# ~~~~~~~~~~~~~~~~~~~~
#
# Bayesian inference uses prior knowledge, in the form of a prior
# distribution, in order to estimate posterior probabilities which we
# traditionally visualise in the form of corner plots. These distributions
# contain more information than a maximum likelihood fit as they reveal not
# only the “best model” but provide a more accurate representation of errors and
# correlation between parameters. In particular, non-Gaussian degeneracies are
# complex to estimate with a maximum likelihood approach.
#
# 2. Limitations of the Markov Chain Monte Carlo approach
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A well-known approach to estimate this posterior distribution is the
# Markov Chain Monte Carlo (MCMC). This uses an ensemble of walkers to
# produce a chain of samples that after a convergence period will reach a
# stationary state. *Once convergence* is reached, the successive elements
# of the chain are samples of the target posterior distribution. However,
# the weakness of the MCMC approach lies in the "*Once convergence*" part.
# If the walkers are started far from the best likelihood region, the convergence time can be
# long or never reached if the walkers fall in a local minima. The choice
# of the initialisation point can become critical for complex models with
# a high number of dimensions and the ability of these walkers to escape a
# local minimum or to accurately describe a complex likelihood space is
# not guaranteed.
#
# 3. Nested sampling approach
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To overcome these issues, the nested sampling (NS) algorithm has
# gained traction in physics and astronomy. It is a Monte Carlo
# algorithm for computing an integral of the likelihood function over
# the prior model parameter space introduced in
# `Skilling, 2004 <https://ui.adsabs.harvard.edu/abs/2004AIPC..735..395S>`__.
# The method performs this integral by evolving a collection of points
# through the parameter space (see recent reviews from `Ashton et al.,
# 2022 <https://ui.adsabs.harvard.edu/abs/2022NRvMP...2...39A>`__, and
# `Buchner, 2023 <http://arxiv.org/abs/2101.09675>`__). Without going
# into too many details, one important specificity of the NS method is
# that it starts from the entire parameter space and evolves a
# collection of live points to map all minima (including multiple modes
# if any), whereas Markov Chain Monte Carlo methods require an
# initialisation point and the walkers will explore the local
# likelihood. The ability of these walkers to escape a local minimum or
# to accurately describe a complex likelihood space is not guaranteed.
# This is a fundamental difference with MCMC or Minuit which will
# only ever probe the vicinity along their minimisation paths and do not
# have an overview of the global likelihood landscape. The analysis
# using the NS framework is more CPU time consuming than a standard
# classical fit, but it provides the full posterior distribution for all
# parameters, which is out of reach with traditional fitting techniques
# (N*(N-1)/2 contour plots to generate). In addition, it is more robust
# to the choice of initialisation, requires less human intervention and
# is therefore readily integrated in pipeline analysis. In Gammapy, we
# used the NS implementation of the UltraNest package
# (see `here <https://johannesbuchner.github.io/UltraNest/>`__ for more information), one of the
# leading package in Astronomy (already used in Cosmology and in
# X-rays).
# For a nice visualisation of the NS method see here : `sampling
# visualisation <https://johannesbuchner.github.io/UltraNest/method.html#visualisation>`__.
# And for a tutorial of UltraNest applied to X-ray fitting with concrete examples and questions see : `BXA
# Tutorial <https://peterboorman.com/tutorial_bxa.html>`__.
#
#
# **Note: please cite UltraNest if used for a paper**
#
# If you are using the "UltraNest" library for a paper, please follow its citation scheme:
# `Cite UltraNest < https: // johannesbuchner.github.io / UltraNest / issues.html  # how-should-i-cite-ultranest>`__.
#
#

######################################################################
# Proposed approach
# -----------------
#
# In this example, we will perform a Bayesian analysis with multiple 1D
# spectra of the Crab nebula data and investigate their posterior
# distributions.
#


######################################################################
# Setup
# -----
#
# As usual, we’ll start with some setup …
#

import matplotlib.pyplot as plt
import numpy as np
from gammapy.datasets import Datasets
from gammapy.datasets import SpectrumDatasetOnOff

from gammapy.modeling.models import (
    SkyModel,
    UniformPrior,
    LogUniformPrior,
)

from gammapy.modeling.sampler import Sampler


######################################################################
# Loading the spectral datasets
# -----------------------------
#
# Here we will load a few Crab 1D spectral data for which we will do a
# fit.
#

path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"

datasets = Datasets()
for id in ["23526", "23559", "23592"]:
    dataset = SpectrumDatasetOnOff.read(f"{path}pha_obs{id}.fits")
    datasets.append(dataset)


######################################################################
# Model definition
# ----------------
#
# Now we want to define the spectral model that will be fitted to the
# data.
# The Crab spectra will be fitted here with a simple powerlaw for
# simplicity.
#

model = SkyModel.create(spectral_model="pl", name="crab")


######################################################################
#
# .. WARNING:: Priors definition:
#    Unlike a traditional fit where priors on the
#    parameters are optional, here it is inherent to the Bayesian approach and
#    are therefore mandatory.
#
# In this case we will set (min,max) prior that will define the
# boundaries in which the sampling will be performed.
# Note that it is usually recommended to use a `~gammapy.modeling.models.LogUniformPrior` for
# the parameters that have a large amplitude range like the
# `amplitude` parameter.
# A `~gammapy.modeling.models.UniformPrior` means that the samples will be drawn with uniform
# probability between the (min,max) values in the linear or log space
# in the case of a `~gammapy.modeling.models.LogUniformPrior`.
#

model.spectral_model.amplitude.prior = LogUniformPrior(min=1e-12, max=1e-10)
model.spectral_model.index.prior = UniformPrior(min=1, max=5)
datasets.models = [model]
print(datasets.models)


######################################################################
# Defining the sampler and options
# --------------------------------
#
# As for the `~gammapy.modeling.Fit` object, the `~gammapy.modeling.Sampler` object can receive
# different backend (although just one is available for now).
# The `~gammapy.modeling.Sampler` comes with “reasonable” default parameters, but you can
# change them via the `sampler_opts` dictionnary.
# Here is a short description of the most relevant parameters that you
# could change :
#
# -  `live_points`: minimum number of live points throughout the run.
#    More points allow to discover multiple peaks if existing, but is
#    slower. To test the Prior boundaries and for debugging, a lower
#    number (~100) can be used before a production run with more points
#    (~400 or more).
# -  `frac_remain`: the cut-off condition for the integration, set by the maximum
#    allowed fraction of posterior mass left in the live points vs the dead points. High
#    values (e.g., 0.5) are faster and can be used if the posterior
#    distribution is a relatively simple shape. A low value (1e-1, 1e-2)
#    is optimal for finding peaks, but slower.
# -  `log_dir`: directory where the output files will be stored.
#
# **Important note:** unlike the MCMC method, you don’t need to define the
# number of steps for which the sampler will run. The algorithm will
# automatically stop once a convergence criteria has been reached.
#

sampler_opts = {
    "live_points": 300,
    "frac_remain": 0.3,
    "log_dir": None,
}

sampler = Sampler(backend="ultranest", sampler_opts=sampler_opts)


######################################################################
# Next we can run the sampler on a given dataset.
# No options are accepted in the run method.
#

result_joint = sampler.run(datasets)


######################################################################
# Understanding the outputs
# -------------------------
#
# In the Jupyter notebook, you should be able to see an interactive
# visualisation of how the parameter space shrinks which starts from the
# (min,max) shrinks down towards the optimal parameters.
#
# The output above is filled with interesting information. Here we
# provide a short description of the most relevant information provided
# above.
# For more detailed information see the `UltraNest
# docs <https://johannesbuchner.github.io/UltraNest/issues.html#what-does-the-status-line-mean>`__.
#
# **During the sampling**
#
# `Z=-68.8(0.53%) | Like=-63.96..-58.75 [-63.9570..-63.9539]*| it/evals=640/1068 eff=73.7327% N=300`
#
# Some important information here is:
#
# -  Progress (0.53%): the completed fraction of the integral. This is not a time progress bar.
#    Stays at zero for a good fraction of the run.
#
# -  Efficiency (eff value) of the sampling: this indicates out of the proposed new points,
#    how many were accepted. If your efficiency is too small (<<1%), maybe
#    you should revise your priors (e.g use a LogUniform prior for the
#    normalisation).
#
# **Final outputs**
#
# The final lines indicate that all three “convergence” strategies are
# satisfied (samples, posterior uncertainty, and evidence uncertainty).
#
# `logZ = -65.104 +- 0.292`
#
# The main goal of the Nested sampling algorithm is to estimate Z (the
# Bayesian evidence) which is given above together with an uncertainty.
# In a similar way to deltaLogLike and deltaAIC, deltaLogZ values can be
# used for model comparison.
# For more information see : `on the use of the evidence for model comparison
# <https://ned.ipac.caltech.edu/level5/Sept13/Trotta/Trotta4.html>`__.
# An interesting comparison of the efficiency and false discovery rate of
# model selection with deltaLogLike and deltaLogZ is given in Appendix C of
# `Buchner et al., 2014 <https://ui.adsabs.harvard.edu/abs/2014A%2526A...564A.125B%252F/>`__.
#
# **Results stored on disk**
#
# if `log_dir` is set to a name where the results will be stored, then
# a directory is created containing many useful results and plots.
# A description of these outputs is given in the `Ultranest
# docs <https://johannesbuchner.github.io/UltraNest/performance.html#output-files>`__.
#


######################################################################
# Results
# -------
#


######################################################################
# Within a Bayesian analysis, the concept of best-fit has to be viewed
# differently from what is done in a gradient descent fit.
#
# The output of the Bayesian analysis is the posterior distribution and
# there is no “best-fit” output.
# One has to define, based on the posteriors, what we want to consider
# as “best-fit” and several options are possible:
#
# -  the mean of the distribution
# -  the median
# -  the lowest likelihood value
#
# By default the `~gammapy.modeling.models.DatasetModels` will be updated with the `mean` of
# the posterior distributions.
#

print(result_joint.models)


######################################################################
# The `~gammapy.modeling.Sampler` class returns a very rich dictionnary.
# The most “standard” information about the posterior distributions can
# be found in :
#

print(result_joint.sampler_results["posterior"])


######################################################################
# Besides mean, errors, etc, an interesting value is the
# `information gain` which estimates how much the posterior
# distribution has shrinked with respect to the prior (i.e. how much
# we’ve learned). A value < 1 means that the parameter is poorly
# constrained within the prior range (we haven't learned much with respect to our prior assumption).
# For a physical example see this
# `example <https://arxiv.org/abs/2205.00009>`__.
#
# The `~gammapy.modeling.SamplerResult` dictionary contains also other interesting
# information :
#

print(result_joint.sampler_results.keys())


######################################################################
# Of particular interest, the samples used in the process to approximate
# the posterior distribution can be accessed via :
#

for i, n in enumerate(model.parameters.free_parameters.names):
    s = result_joint.samples[:, i]
    fig, ax = plt.subplots()
    ax.hist(s, bins=30)
    ax.axvline(np.mean(s), ls="--", color="red")
    ax.set_xlabel(n)
    plt.show()


######################################################################
# While the above plots are interesting, the real strength of the Bayesian
# analysis is to visualise all parameters correlations which is usually
# done using “corner plots”.
# Ultranest corner plot function is a wrapper around the `corner
# <https://corner.readthedocs.io/en/latest/api>`__ package.
# See the above link for optional keywords.
# Other packages exist for corner plots, like
# `chainconsumer <https://chainconsumer.readthedocs.io/en/latest/>`__ which is discussed later in this tutorial.

from ultranest.plot import cornerplot

cornerplot(
    result_joint.sampler_results,
    plot_datapoints=True,
    plot_density=True,
    bins=20,
    title_fmt=".2e",
    smooth=False,
)
plt.show()

# sphinx_gallery_thumbnail_number = 3


######################################################################
# Individual run analysis
# -----------------------
#
# Now we’ll analyse several Crab runs individually so that we can compare
# them.
#

result_0 = sampler.run(datasets[0])
result_1 = sampler.run(datasets[1])
result_2 = sampler.run(datasets[2])


######################################################################
# Comparing the posterior distribution of all runs
# ------------------------------------------------
#
# For a comparison of different posterior distributions, we can use the
# package chainconsumer.
# As this is not a Gammapy dependency, you’ll need to install it.
# More info here : https://samreay.github.io/ChainConsumer/
#

# Uncomment this if you have installed `chainconsumer`.

# from chainconsumer import Chain, ChainConfig, ChainConsumer, PlotConfig, Truth, make_sample
# from pandas import DataFrame
# c = ChainConsumer()
# def create_chain(result, name, color="k"):
#    return Chain(
#        samples=DataFrame(result, columns=["index", "amplitude"]),
#        name=name,
#        color=color,
#        smooth=7,
#        shade=False,
#        linewidth=1.0,
#        cmap="magma",
#        show_contour_labels=True,
#        kde= True
#    )
# c.add_chain(create_chain(result_joint.samples, "joint"))
# c.add_chain(create_chain(result_0.samples, "run0", "g"))
# c.add_chain(create_chain(result_1.samples, "run1", "b"))
# c.add_chain(create_chain(result_2.samples, "run2", "y"))
# fig = c.plotter.plot()
# plt.show()


######################################################################
#
# Corner plot comparison
# ----------------------
#
# .. figure:: ../../_static/cornerplot-multiple-runs-Crab.png
#     :alt: Corner plot of Crab runs
#
#     Corner plot comparing the three Crab runs.
#
#
# We can see the joint analysis allows to better constrain the
# parameters than the individual runs (more observation time is of
# course better).
# One can note as well that one of the run has a notably different
# amplitude (possibly due to calibrations or/and atmospheric issues).
#
