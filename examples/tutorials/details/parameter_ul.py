"""
Constraining parameter limits
=============================

Explore how to deal with upper limits on parameters.

Prerequisites
-------------

It is advisable to understand the general Gammapy modelling and fitting framework before proceeding
with this notebook, e.g. see :doc:`/user-guide/modeling`.

Context
-------

Even with significant detection of a source, constraining specific model parameters may remain difficult,
allowing only for the calculation of confidence intervals.

Proposed approach
-----------------

In this section, we will use 6 observations of the blazar PKS 2155-304, taken in 2008 by
H.E.S.S, to constrain the curvature in the spectrum.

"""

######################################################################
# Setup
# -----
#
# As usual, let’s start with some general imports…
#

# %matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import astropy.units as u
from gammapy.datasets import SpectrumDatasetOnOff, Datasets
from gammapy.modeling import Fit, select_nested_models
from gammapy.modeling.models import SkyModel, LogParabolaSpectralModel
from gammapy.estimators import FluxPointsEstimator

######################################################################
# Load observation
# ----------------
#
# We will use a `~gammapy.datasets.SpectrumDatasetOnOff` to see how to constrain
# model parameters. This dataset was obtained from H.E.S.S. observation of the blazar PKS~2155-304.
# Detailed modeling of this dataset may be found in the
# :doc:`/tutorials/analysis-1d/ebl` notebook.
#

dataset_onoff = SpectrumDatasetOnOff.read(
    "$GAMMAPY_DATA/PKS2155-steady/pks2155-304_steady.fits.gz"
)
dataset_onoff.peek()
plt.show()


######################################################################
# Fit spectrum
# ------------
#
# We will search for the presence of curvature in the spectrum. For this,
# we will use a ~gammapy.modeling.models.LogParabolaSpectralModel to model the
# observed spectrum.
#

spectral_model = LogParabolaSpectralModel(
    amplitude="5e-12 TeV-1 s-1 cm-2", alpha=2, beta=0.5, reference=1.0 * u.TeV
)

model_pks = SkyModel(spectral_model, name="model_pks")
dataset_onoff.models = model_pks

fit = Fit()
result_pks = fit.run(dataset_onoff)
print(result_pks.models)


######################################################################
# We see that the parameter ``beta`` (the curvature parameter) is not well
# constrained as the errors are very large. In this
# case, we will first use a likelihood ratio test to see how significant is the
# curvature, as compared to the null hypothesis, ie no curvature.

LLR = select_nested_models(
    datasets=Datasets(dataset_onoff),
    parameters=[model_pks.parameters["beta"]],
    null_values=[0],
)
print(LLR)


######################################################################
# We can see that the change in improvement in test statistic after adding the
# curvature is only ~0.3, which corresponds to a significance of only 0.5.
# Thus, we can safely conclude that the addition of the curvature parameter does
# not improve the fit. Thus, the function has internally updated
# the best fit model to the one corresponding to the null hypothesis

print(dataset_onoff.models)


######################################################################
# Compute parameter asymmetric erros and upper limits
# ---------------------------------------------------
# In such a case, it can still be useful to be able to constrain
# the allowed range of the non-significant parameter (eg: to rule
# out parameter values, to compare from theoretical predications, etc).
#
# First, we reset the alternative model on the dataset

dataset_onoff.models = LLR["fit_results"].models
parameter = dataset_onoff.models.parameters["beta"]

######################################################################
# then we can compute the asymmetric errors and upper limits on the parameter

res_1sig = fit.confidence(datasets=dataset_onoff, parameter=parameter, sigma=1)
print(res_1sig)

######################################################################
# and the upper limits on the parameter.

res_2sig = fit.confidence(datasets=dataset_onoff, parameter=parameter, sigma=2)
ll_2sigma = res_2sig["errn"] + parameter.value
ul_2sigma = res_2sig["errp"] + parameter.value

print(f"2-sigma lower limit on beta is {ul_2sigma:.2f}")
print(f"2-sigma upper limit on beta is {ul_2sigma:.2f}")

######################################################################
# Likekihood profile
# ------------------
# We can also compute the likelihood profile on the parameter.
# First we define the scan range such as it emcompasses the 1 and 2-sigma parameter limits.
# Then we call `fit.stat_profile` :

parameter.scan_n_values = 25
parameter.scan_min = -1
parameter.scan_max = 10
parameter.interp = "lin"
profile = fit.stat_profile(datasets=dataset_onoff, parameter=parameter, reoptimize=True)

######################################################################
# The resulting `profile` is a dictionary that stores the likelihood value and the fit result
# for each value of beta.

print(profile)


######################################################################
# Let's plot everything together

values = profile["model_pks.spectral.beta_scan"]
loglike = profile["stat_scan"]
ax = plt.gca()
ax.plot(values, loglike - np.min(loglike))
ax.set_xlabel("Beta")
ax.set_ylabel(r"$\Delta$TS")
ax.set_title(r"$\beta$-parameter likelihood profile")
ax.fill_betweenx(
    x1=parameter.value - res_2sig["errn"],
    x2=parameter.value + res_2sig["errp"],
    y=[-0.5, 25],
    alpha=0.3,
    color="pink",
    label="1-sigma range",
)
ax.fill_betweenx(
    x1=parameter.value - res_1sig["errn"],
    x2=parameter.value + res_1sig["errp"],
    y=[-0.5, 25],
    alpha=0.3,
    color="salmon",
    label="2-sigma range",
)
ax.set_ylim(-0.5, 25)
plt.legend()
plt.show()
# sphinx_gallery_thumbnail_number = 4


######################################################################

# Impact of the model choice on the flux upper limits
# ---------------------------------------------------
# The flux points depends on the underlying model assumption.
# This can have an non-negligible impact on the flux upper limits in the energy range
# where the model is not well constrained as illustrated in the following figure.
# So quote preferably upper limits from the model which is the most supported by the data.

energies = dataset_onoff.geoms["geom"].axes["energy"].edges

# null hypothesis
dataset_onoff.models = LLR["fit_results_null"].models
fpe = FluxPointsEstimator(energy_edges=energies, n_jobs=4, selection_optional=["ul"])
fp = fpe.run(dataset_onoff)
ax = fp.plot(sed_type="e2dnde", color="blue")
LLR["fit_results_null"].models[0].spectral_model.plot(
    ax=ax,
    energy_bounds=(energies[0], energies[-1]),
    sed_type="e2dnde",
    color="blue",
    label="No curvature",
)
LLR["fit_results_null"].models[0].spectral_model.plot_error(
    ax=ax,
    energy_bounds=(energies[0], energies[-1]),
    sed_type="e2dnde",
    facecolor="blue",
    alpha=0.2,
)

# alternative hypothesis
dataset_onoff.models = LLR["fit_results"].models
fpe = FluxPointsEstimator(energy_edges=energies, n_jobs=4, selection_optional=["ul"])
fp = fpe.run(dataset_onoff)
ax = fp.plot(sed_type="e2dnde", color="red")
LLR["fit_results"].models[0].spectral_model.plot(
    ax=ax,
    energy_bounds=(energies[0], energies[-1]),
    sed_type="e2dnde",
    color="red",
    label="with curvature",
)
LLR["fit_results"].models[0].spectral_model.plot_error(
    ax=ax,
    energy_bounds=(energies[0], energies[-1]),
    sed_type="e2dnde",
    facecolor="red",
    alpha=0.2,
)

plt.legend()
plt.show()

######################################################################
# This logic can be extended to any spectral or spatial feature. As an
# exercise, try to compute the 95% spatial extent on the MSH 15-52 dataset
# used for the ring background notebook.
#
