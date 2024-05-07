"""
Fitting
=======

Learn how the model, dataset and fit Gammapy classes work together in a detailed modeling and fitting use-case.

Prerequisites
-------------

-  Knowledge of spectral analysis to produce 1D On-Off datasets, see
   the :doc:`/tutorials/analysis-1d/spectral_analysis` tutorial.
-  Reading of pre-computed datasets see e.g.
   :doc:`/tutorials/analysis-3d/analysis_mwl` tutorial.
-  General knowledge on statistics and optimization methods

Proposed approach
-----------------

This is a hands-on tutorial to `~gammapy.modeling`, showing how to do
perform a Fit in gammapy. The emphasis here is on interfacing the
`Fit` class and inspecting the errors. To see an analysis example of
how datasets and models interact, see the :doc:`/tutorials/api/model_management` tutorial.
As an example, in this notebook, we are going to work with HESS data of the Crab Nebula and show in
particular how to :

- perform a spectral analysis
- use different fitting backends
- access covariance matrix information and parameter errors
- compute likelihood profile - compute confidence contours

See also: :doc:`/tutorials/api/models` and :ref:`modeling`.

The setup
---------

"""

from itertools import combinations
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from IPython.display import display
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.modeling import Fit
from gammapy.modeling.models import LogParabolaSpectralModel, SkyModel

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup
from gammapy.visualization.utils import plot_contour_line

check_tutorials_setup()


######################################################################
# Model and dataset
# -----------------
#
# First we define the source model, here we need only a spectral model for
# which we choose a log-parabola
#

crab_spectrum = LogParabolaSpectralModel(
    amplitude=1e-11 / u.cm**2 / u.s / u.TeV,
    reference=1 * u.TeV,
    alpha=2.3,
    beta=0.2,
)

crab_spectrum.alpha.max = 3
crab_spectrum.alpha.min = 1
crab_model = SkyModel(spectral_model=crab_spectrum, name="crab")


######################################################################
# The data and background are read from pre-computed ON/OFF datasets of
# HESS observations, for simplicity we stack them together. Then we set
# the model and fit range to the resulting dataset.
#

datasets = []
for obs_id in [23523, 23526]:
    dataset = SpectrumDatasetOnOff.read(
        f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
    )
    datasets.append(dataset)

dataset_hess = Datasets(datasets).stack_reduce(name="HESS")
datasets = Datasets(datasets=[dataset_hess])

# Set model and fit range
dataset_hess.models = crab_model
e_min = 0.66 * u.TeV
e_max = 30 * u.TeV
dataset_hess.mask_fit = dataset_hess.counts.geom.energy_mask(e_min, e_max)


######################################################################
# Fitting options
# ---------------
#
# First let’s create a `Fit` instance:
#

scipy_opts = {
    "method": "L-BFGS-B",
    "options": {"ftol": 1e-4, "gtol": 1e-05},
    "backend": "scipy",
}
fit_scipy = Fit(store_trace=True, optimize_opts=scipy_opts)


######################################################################
# By default the fit is performed using MINUIT, you can select alternative
# optimizers and set their option using the `optimize_opts` argument of
# the `Fit.run()` method. In addition we have specified to store the
# trace of parameter values of the fit.
#
# Note that, for now, covariance matrix and errors are computed only for
# the fitting with MINUIT. However, depending on the problem other
# optimizers can better perform, so sometimes it can be useful to run a
# pre-fit with alternative optimization methods.
#
# | For the “scipy” backend the available options are described in detail
#   here:
# | https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#

# %%time
result_scipy = fit_scipy.run(datasets)


######################################################################
# | For the “sherpa” backend you can choose the optimization algorithm
#   between method = {“simplex”, “levmar”, “moncar”, “gridsearch”}.
# | Those methods are described and compared in detail on
#   http://cxc.cfa.harvard.edu/sherpa/methods/index.html The available
#   options of the optimization methods are described on the following
#   page https://cxc.cfa.harvard.edu/sherpa/methods/opt_methods.html
#

# %%time
sherpa_opts = {"method": "simplex", "ftol": 1e-3, "maxfev": int(1e4)}
fit_sherpa = Fit(store_trace=True, backend="sherpa", optimize_opts=sherpa_opts)
results_simplex = fit_sherpa.run(datasets)


######################################################################
# For the “minuit” backend see
# https://iminuit.readthedocs.io/en/latest/reference.html for a detailed
# description of the available options. If there is an entry
# ‘migrad_opts’, those options will be passed to
# `iminuit.Minuit.migrad <https://iminuit.readthedocs.io/en/latest/reference.html#iminuit.Minuit.migrad>`__.
# Additionally you can set the fit tolerance using the
# `tol <https://iminuit.readthedocs.io/en/latest/reference.html#iminuit.Minuit.tol>`__
# option. The minimization will stop when the estimated distance to the
# minimum is less than 0.001*tol (by default tol=0.1). The
# `strategy <https://iminuit.readthedocs.io/en/latest/reference.html#iminuit.Minuit.strategy>`__
# option change the speed and accuracy of the optimizer: 0 fast, 1
# default, 2 slow but accurate. If you want more reliable error estimates,
# you should run the final fit with strategy 2.
#

# %%time
fit = Fit(store_trace=True)
minuit_opts = {"tol": 0.001, "strategy": 1}
fit.backend = "minuit"
fit.optimize_opts = minuit_opts
result_minuit = fit.run(datasets)


######################################################################
# Fit quality assessment
# ----------------------
#
# There are various ways to check the convergence and quality of a fit.
# Among them:
#
# Refer to the automatically-generated results dictionary:
#

print(result_scipy)

# %%

print(results_simplex)

# %%

print(result_minuit)


######################################################################
# If the fit is performed with minuit you can print detailed information
# to check the convergence
#

print(result_minuit.minuit)


######################################################################
# Check the trace of the fit e.g.  in case the fit did not converge
# properly
#

display(result_minuit.trace)


######################################################################
# The fitted models are copied on the `~gammapy.modeling.FitResult` object.
# They can be inspected to check that the fitted values and errors
# for all parameters are reasonable, and no fitted parameter value is “too close”
# - or even outside - its allowed min-max range
#

display(result_minuit.models.to_parameters_table())


######################################################################
# Plot fit statistic profiles for all fitted parameters, using
# `~gammapy.modeling.Fit.stat_profile`. For a good fit and error
# estimate each profile should be parabolic. The specification for each
# fit statistic profile can be changed on the
# `~gammapy.modeling.Parameter` object, which has `~gammapy.modeling.Parameter.scan_min`,
# `~gammapy.modeling.Parameter.scan_max`, `~gammapy.modeling.Parameter.scan_n_values` and `~gammapy.modeling.Parameter.scan_n_sigma` attributes.
#

total_stat = result_minuit.total_stat

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))

for ax, par in zip(axes, datasets.parameters.free_parameters):
    par.scan_n_values = 17
    idx = datasets.parameters.index(par)
    name = datasets.models.parameters_unique_names[idx]
    profile = fit.stat_profile(datasets=datasets, parameter=par)
    ax.plot(profile[f"{name}_scan"], profile["stat_scan"] - total_stat)
    ax.set_xlabel(f"{par.name} [{par.unit}]")
    ax.set_ylabel("Delta TS")
    ax.set_title(f"{name}:\n {par.value:.1e} +- {par.error:.1e}")
plt.show()


######################################################################
# Inspect model residuals. Those can always be accessed using
# `~gammapy.datasets.Dataset.residuals()`. For more details, we refer here to the dedicated
# :doc:`/tutorials/analysis-3d/analysis_3d` (for `~gammapy.datasets.MapDataset` fitting) and
# :doc:`/tutorials/analysis-1d/spectral_analysis` (for `SpectrumDataset` fitting).
#


######################################################################
# Covariance and parameters errors
# --------------------------------
#
# After the fit the covariance matrix is attached to the models copy
# stored on the `~gammapy.modeling.FitResult` object.
# You can access it directly with:

print(result_minuit.models.covariance)

######################################################################
# And you can plot the total parameter correlation as well:
#

result_minuit.models.covariance.plot_correlation()
plt.show()

# The covariance information is also propagated to the individual models
# Therefore, one can also get the error on a specific parameter by directly
# accessing the `~gammapy.modeling.Parameter.error` attribute:
#

print(crab_model.spectral_model.alpha.error)


######################################################################
# As an example, this step is needed to produce a butterfly plot showing
# the envelope of the model taking into account parameter uncertainties.
#

energy_bounds = [1, 10] * u.TeV
crab_spectrum.plot(energy_bounds=energy_bounds, energy_power=2)
ax = crab_spectrum.plot_error(energy_bounds=energy_bounds, energy_power=2)
plt.show()


######################################################################
# Confidence contours
# -------------------
#
# In most studies, one wishes to estimate parameters distribution using
# observed sample data. A 1-dimensional confidence interval gives an
# estimated range of values which is likely to include an unknown
# parameter. A confidence contour is a 2-dimensional generalization of a
# confidence interval, often represented as an ellipsoid around the
# best-fit value.
#
# Gammapy offers two ways of computing confidence contours, in the
# dedicated methods `~gammapy.modeling.Fit.stat_contour` and `~gammapy.modeling.Fit.stat_profile`. In
# the following sections we will describe them.
#


######################################################################
# An important point to keep in mind is: *what does a* :math:`N\sigma`
# *confidence contour really mean?* The answer is it represents the points
# of the parameter space for which the model likelihood is :math:`N\sigma`
# above the minimum. But one always has to keep in mind that **1 standard
# deviation in two dimensions has a smaller coverage probability than
# 68%**, and similarly for all other levels. In particular, in
# 2-dimensions the probability enclosed by the :math:`N\sigma` confidence
# contour is :math:`P(N)=1-e^{-N^2/2}`.
#


######################################################################
# Computing contours using `~gammapy.modeling.Fit.stat_contour`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# After the fit, MINUIT offers the possibility to compute the confidence
# contours. gammapy provides an interface to this functionality through
# the `~gammapy.modeling.Fit` object using the `~gammapy.modeling.Fit.stat_contour` method. Here we defined a
# function to automate the contour production for the different
# parameter and confidence levels (expressed in terms of sigma):
#


def make_contours(fit, datasets, result, npoints, sigmas):
    cts_sigma = []
    for sigma in sigmas:
        contours = dict()
        for par_1, par_2 in combinations(["alpha", "beta", "amplitude"], r=2):
            idx1, idx2 = datasets.parameters.index(par_1), datasets.parameters.index(
                par_2
            )
            name1 = datasets.models.parameters_unique_names[idx1]
            name2 = datasets.models.parameters_unique_names[idx2]
            contour = fit.stat_contour(
                datasets=datasets,
                x=datasets.parameters[par_1],
                y=datasets.parameters[par_2],
                numpoints=npoints,
                sigma=sigma,
            )
            contours[f"contour_{par_1}_{par_2}"] = {
                par_1: contour[name1].tolist(),
                par_2: contour[name2].tolist(),
            }
        cts_sigma.append(contours)
    return cts_sigma


######################################################################
# Now we can compute few contours.
#

# %%time
sigmas = [1, 2]
cts_sigma = make_contours(
    fit=fit,
    datasets=datasets,
    result=result_minuit,
    npoints=10,
    sigmas=sigmas,
)


######################################################################
# Then we prepare some aliases and annotations in order to make the
# plotting nicer.
#

pars = {
    "phi": r"$\phi_0 \,/\,(10^{-11}\,{\rm TeV}^{-1} \, {\rm cm}^{-2} {\rm s}^{-1})$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
}

panels = [
    {
        "x": "alpha",
        "y": "phi",
        "cx": (lambda ct: ct["contour_alpha_amplitude"]["alpha"]),
        "cy": (lambda ct: np.array(1e11) * ct["contour_alpha_amplitude"]["amplitude"]),
    },
    {
        "x": "beta",
        "y": "phi",
        "cx": (lambda ct: ct["contour_beta_amplitude"]["beta"]),
        "cy": (lambda ct: np.array(1e11) * ct["contour_beta_amplitude"]["amplitude"]),
    },
    {
        "x": "alpha",
        "y": "beta",
        "cx": (lambda ct: ct["contour_alpha_beta"]["alpha"]),
        "cy": (lambda ct: ct["contour_alpha_beta"]["beta"]),
    },
]


######################################################################
# Finally we produce the confidence contours figures.
#

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ["m", "b", "c"]
for p, ax in zip(panels, axes):
    xlabel = pars[p["x"]]
    ylabel = pars[p["y"]]
    for ks in range(len(cts_sigma)):
        plot_contour_line(
            ax,
            p["cx"](cts_sigma[ks]),
            p["cy"](cts_sigma[ks]),
            lw=2.5,
            color=colors[ks],
            label=f"{sigmas[ks]}" + r"$\sigma$",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
plt.legend()
plt.tight_layout()


######################################################################
# Computing contours using `~gammapy.modeling.Fit.stat_surface`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This alternative method for the computation of confidence contours,
# although more time consuming than `~gammapy.modeling.Fit.stat_contour()`, is expected
# to be more stable. It consists of a generalization of
# `~gammapy.modeling.Fit.stat_profile()` to a 2-dimensional parameter space. The algorithm
# is very simple: - First, passing two arrays of parameters values, a
# 2-dimensional discrete parameter space is defined; - For each node of
# the parameter space, the two parameters of interest are frozen. This
# way, a likelihood value (:math:`-2\mathrm{ln}\,\mathcal{L}`, actually)
# is computed, by either freezing (default) or fitting all nuisance
# parameters; - Finally, a 2-dimensional surface of
# :math:`-2\mathrm{ln}(\mathcal{L})` values is returned. Using that
# surface, one can easily compute a surface of
# :math:`TS = -2\Delta\mathrm{ln}(\mathcal{L})` and compute confidence
# contours.
#
# Let’s see it step by step.
#
# First of all, we can notice that this method is “backend-agnostic”,
# meaning that it can be run with MINUIT, sherpa or scipy as fitting
# tools. Here we will stick with MINUIT, which is the default choice:
#
# As an example, we can compute the confidence contour for the `alpha`
# and `beta` parameters of the `dataset_hess`. Here we define the
# parameter space:
#

result = result_minuit
par_alpha = datasets.parameters["alpha"]
par_beta = datasets.parameters["beta"]

par_alpha.scan_values = np.linspace(1.55, 2.7, 20)
par_beta.scan_values = np.linspace(-0.05, 0.55, 20)


######################################################################
# Then we run the algorithm, by choosing `reoptimize=False` for the sake
# of time saving. In real life applications, we strongly recommend to use
# `reoptimize=True`, so that all free nuisance parameters will be fit at
# each grid node. This is the correct way, statistically speaking, of
# computing confidence contours, but is expected to be time consuming.
#

fit = Fit(backend="minuit", optimize_opts={"print_level": 0})
stat_surface = fit.stat_surface(
    datasets=datasets,
    x=par_alpha,
    y=par_beta,
    reoptimize=False,
)


######################################################################
# In order to easily inspect the results, we can convert the
# :math:`-2\mathrm{ln}(\mathcal{L})` surface to a surface of statistical
# significance (in units of Gaussian standard deviations from the surface
# minimum):
#

# Compute TS
TS = stat_surface["stat_scan"] - result.total_stat

# Compute the corresponding statistical significance surface
stat_surface = np.sqrt(TS.T)


######################################################################
# Notice that, as explained before, :math:`1\sigma` contour obtained this
# way will not contain 68% of the probability, but rather
#

# Compute the corresponding statistical significance surface
# p_value = 1 - st.chi2(df=1).cdf(TS)
# gaussian_sigmas = st.norm.isf(p_value / 2).T


######################################################################
# Finally, we can plot the surface values together with contours:
#

fig, ax = plt.subplots(figsize=(8, 6))
x_values = par_alpha.scan_values
y_values = par_beta.scan_values

# plot surface
im = ax.pcolormesh(x_values, y_values, stat_surface, shading="auto")
fig.colorbar(im, label="sqrt(TS)")
ax.set_xlabel(f"{par_alpha.name}")
ax.set_ylabel(f"{par_beta.name}")

# We choose to plot 1 and 2 sigma confidence contours
levels = [1, 2]
contours = ax.contour(x_values, y_values, stat_surface, levels=levels, colors="white")
ax.clabel(contours, fmt="%.0f $\\sigma$", inline=3, fontsize=15)

plt.show()

######################################################################
# Note that, if computed with `reoptimize=True`, this plot would be
# completely consistent with the third panel of the plot produced with
# `~gammapy.modeling.Fit.stat_contour` (try!).
#


######################################################################
# Finally, it is always remember that confidence contours are
# approximations. In particular, when the parameter range boundaries are
# close to the contours lines, it is expected that the statistical meaning
# of the contours is not well defined. That’s why we advise to always
# choose a parameter space that contains the contours you’re interested
# in.
#
