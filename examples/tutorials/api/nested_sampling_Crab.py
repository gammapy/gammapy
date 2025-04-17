"""
Bayesian analysis with nested sampling
======================================

Bayesian analysis with nested sampling.

"""


######################################################################
# Context
# =======
#
# Bayesian analysis
# ~~~~~~~~~~~~~~~~~
#
# Bayesian inference uses prior knowledge, in the form of a prior
# distribution, in order to estimate posterior probabilities which we
# traditionally visualize in the form of corner plots. These distributions
# contain much more information than a single best-fit as they reveal not
# only the “best model” but the (not always Gaussian) errors and
# correlation between parameters.
#
# Limitations of the Markov Chain Monte Carlo approach
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A well-know approach to estimate this posterior distribution is the
# Markov Chain Monte Carlo (MCMC) which uses an ensemble of walkers to
# produce a chain of samples that after a convergence period will reach a
# stationary state. *Once convergence* is reached the successive elements
# of the chain are samples of the target posterior distribution. However
# the weakness of the MCMC approach lies in the *Once convergence* part.
# Started far from the best likelihood region, the convergence time can be
# long or never reached if the walkers fall in a local minima. The choice
# of the initialization point can become critical for complex models with
# a high number of dimensions and the ability of these walkers to escape a
# local minimum or to accurately describe a complex likelihood space is
# not guaranteed.
#
# Nested sampling approach
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# | To overcome these issues, the nested sampling (NS) algorithm has
#   gained traction in physics and astronomy. It is a Monte Carlo
#   algorithm for computing an integral of the likelihood function over
#   the prior model parameter space introduced in 2004 by John Skilling.
#   The method performs this integral by evolving a collection of points
#   through the parameter space (see recent reviews of `Ashton
#   2022 <https://ui.adsabs.harvard.edu/abs/2022NRvMP...2...39A>`__, and
#   of `Buchner 2023 <http://arxiv.org/abs/2101.09675>`__). Without going
#   into too many details, one important specificity of the NS method is
#   that it starts from the entire parameter space and evolves a
#   collection of live points to map all minima (including multiple modes
#   if any) whereas Markov Chain Monte Carlo methods require an
#   initialization point and the walkers will explore the local
#   likelihood. The ability of these walkers to escape a local minimum or
#   to accurately describe a complex likelihood space is not guaranteed.
#   This is a fundamental difference between MCMC (and Minuit) which will
#   only ever probe the vicinity along their minimization paths and do not
#   have an overview of the global likelihood landscape. The analysis
#   using the NS framework is more CPU time consuming than a standard
#   classical fit but provides the full posterior distribution for all
#   parameters, which is out of reach with traditional fitting techniques
#   (N*(N-1)/2 contour plots to generate). In addition it is more robust
#   to the choice of initialization, requires less human intervention and
#   is therefore readily integrated in pipeline analysis. In gammapy, we
#   used the NS implementation of the UltraNest package, one of the
#   leading package in Astronomy (already used in Cosmology and in
#   X-rays).
# | For more information on UltraNest see the docs here : `UltraNest
#   docs <https://johannesbuchner.github.io/UltraNest/>`__
# | And a nice visualization of the NS method : `sampling
#   visulisation <https://johannesbuchner.github.io/UltraNest/method.html#visualisation>`__
#
# Reference :
# ~~~~~~~~~~~
#
# -  Ultranest docs : https://johannesbuchner.github.io/UltraNest
# -  Literature : `Buchner 2023 <http://arxiv.org/abs/2101.09675>`__,
#    `Ashton
#    2022 <https://ui.adsabs.harvard.edu/abs/2022NRvMP...2...39A>`__
#


######################################################################
# Proposed approach
# =================
#
# In this example we will perform a Bayesian analysis with multiple 1D
# spectra of the Crab nebula data and investigate their posterior
# distributions.
#


######################################################################
# Setup
# -----
#
# First, we setup the analysis by performing required imports.
#

# %matplotlib inline
import matplotlib.pyplot as plt
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
# | Now we want to define the spectral model that will be fitted to the
#   data.
# | The Crab spectra will be fitted here with a simple powerlaw for
#   simplicity.
#

model = SkyModel.create(spectral_model="pl")


######################################################################
# **!!Priors!!: unlike a traditional fit where defining priors on the
# parameters is optional, here it is inherent to the Bayesian approach and
# are therefore mandatory.**
#
# | In this case we will set (min,max) prior that will define the
#   boundaries in which the sampling will be performed.
# | Note that it is usually recommended to use a `LogUniformPrior` for
#   the parameters that have a large amplitude range like the
#   `amplitude` parameter.
# | A `UniformPrior` means that the samples will be drawn with uniform
#   probability between the (min,max) values in the linear or log space
#   (`LogUniformPrior`).
#

model.spectral_model.amplitude.prior = LogUniformPrior(min=1e-12, max=1e-10)
model.spectral_model.index.prior = UniformPrior(min=2, max=3)
# Now setting the model to the dataset :
datasets.models = [model]

datasets.models


######################################################################
# Defining the sampler and options
# --------------------------------
#
# | As for the `Fit` object, the `Sampler` object can receive
#   different backend (although just one is available for now).
# | The `Sampler` comes with “reasonable” default parameters but you can
#   change them via the `sampler_opts` dictionnary.
# | Here is a short description of the most relevant parameters that you
#   could change :
#
# -  `live_points`:
# -  `frac_remain`:
# -  `log_dir`:
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
run_opts = {"min_ess": 300}

sampler = Sampler(backend="ultranest", sampler_opts=sampler_opts, run_opts=run_opts)


######################################################################
# Next we can run the sampler on a given dataset using `.run()` in the
# same way that the `Fit` object. No options are accepted in the run
# method.
#

result_full = sampler.run(datasets)


######################################################################
# In the Jupyter notebook, you should be able to see an interactive
# visualization of how the parameter space shrinks which starts from the
# (min,max) shrinks down towards the optimal parameters.
#
# The output above is filled with interesting information which are not
# easy to understand. Here we provide a short description of the most
# relevant information provided above.
#
# | **During the sampling**
# | - Z=-68.8(0.53%) \| Like=-63.96..-58.75 [-63.9570..-63.9539]*\|
#   it/evals=640/1068 eff=73.7327% N=200
#
# **Final output**
#
# -  de"dfze
#
# **Results stored on disk**
#
# -  dze
#


######################################################################
# Understanding the results
# -------------------------
#


######################################################################
# Within a Bayesian analysis, the concept of best-fit has to be viewed
# differently from what is done in a Minuit fit.
#
# | The output of the Bayesian analysis is the posterior distribution and
#   there is no “best-fit” output.
# | One has to define based on the posteriors what we want to consider as
#   “best-fit” and several options are possible:
#
# -  the mean of the distribution
# -  the median
# -  the lowest likelihood value
#
# Be default the `DatasetModels` will be updated with the `mean` of
# the posterior distributions.
#

result_full.models


######################################################################
# | The Sampler class returns a very rich dictionnary.
# | The most “standard” information about the posterior distributions can
#   be found in :
#

result_full.sampler_results["posterior"]


######################################################################
# But they are much more information than just mean, errors, etc
#

result_full.sampler_results.keys()


######################################################################
# Of particular interest, the samples used in the process to approximate
# the posterior distribution can be accessed via :
#

for i, n in enumerate(model.parameters.free_parameters.names):
    plt.hist(result_full.samples[:, i], bins=30)
    plt.xlabel(n)
    plt.show()


######################################################################
# While the above plots are interesting, the real strength of the Bayesian
# analysis is to visualize all parameters correlations which is usually
# done using “corner plots”.
#

from ultranest.plot import cornerplot

cornerplot(result_full.sampler_results, plot_datapoints=True, plot_density=True)
plt.show()

# sphinx_gallery_thumbnail_number = 2


######################################################################
# Individual run analysis
# =======================
#
# Now we’ll analyze several runs individually so that we can compare them.
#

result_0 = sampler.run(datasets[0])
result_1 = sampler.run(datasets[1])
result_2 = sampler.run(datasets[2])


######################################################################
# Comparing the posterior distribution of all runs
# ================================================
#
# | For an easy comparison of different posterior distribution we can use
#   the package chainconsumer.
# | As this is not a gammapy dependency, you’ll need to install it.
# | More info here : https://samreay.github.io/ChainConsumer/
#

# Uncomment this if you have installed chainconsumer

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
# c.add_chain(create_chain(result_full.samples, "joint"))
# c.add_chain(create_chain(result_0.samples, "run0", "g"))
# c.add_chain(create_chain(result_1.samples, "run1", "b"))
# c.add_chain(create_chain(result_2.samples, "run2", "y"))
# fig = c.plotter.plot()
# plt.show()


######################################################################
# Corner plot comparison
# ======================
#

"""
.. figure:: ../../../docs/_static/cornerplot-multiple-runs-Crab.png
    :scale: 100%

    Corner plot comparing the three Crab runs.  
    The joint run allows to better constrain the parameters than individual runs.  
    One can note as well that one of the run has a notably different amplitude (due to calibrations issues ?).

"""
