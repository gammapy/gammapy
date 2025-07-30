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
from gammapy.modeling.models import (
    SkyModel,
    LogParabolaSpectralModel,
)
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
# the best fit model to the one corresponding to the hull hypothesis

print(dataset_onoff.models)

######################################################################
# Get flux points
# ---------------
# In can be useful to compute the flux points and visualise the difference
# in the two spectral models


energies = dataset_onoff.geoms["geom"].axes["energy"].edges
fpe = FluxPointsEstimator(energy_edges=energies, n_jobs=4, selection_optional=["ul"])
fp = fpe.run(dataset_onoff)


ax = fp.plot(sed_type="e2dnde", color="black")
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
plt.legend()
plt.show()


######################################################################
# Compute parameter limits
# ------------------------
# In such a case, it can still be useful to be able to constrain
# the allowed range of the non-significant parameter (eg: to rule
# out parameter values, to compare from theoretical predications, etc).
# For this, we will look at the stat profile of our parameter of interest.


parameter = model_pks.parameters["beta"]
parameter.scan_n_values = 25
parameter.scan_min = -1
parameter.scan_max = 10
parameter.interp = "lin"
profile = fit.stat_profile(datasets=dataset_onoff, parameter=parameter, reoptimize=True)


######################################################################
# `profile` is a dictionary that stores the likelihood value and the fit result
# for each value of beta.

print(profile)

values = profile["model_pks.spectral.beta_scan"]
loglike = profile["stat_scan"]
ax = plt.gca()
ax.plot(values, loglike - np.min(loglike))
ax.set_xlabel("Beta")
ax.set_ylabel(r"$\Delta$TS")
ax.set_title(r"$\beta$-parameter likelihood profile")
plt.show()

######################################################################
# We can see that the likelihood profile is highly non-symmetric, and thus,
# the error on the parameter quoted from the covariance matrix is not sufficient.
# We can compute the asymmetric errors on the parameter

errors = fit.confidence(datasets=dataset_onoff, parameter=parameter, sigma=1)
print(errors)


######################################################################
# This logic can be extended to any spectral or spatial feature. As an
# exercise, try to compute the 95% spatial extent on the MSH 15-52 dataset
# used for the ring background notebook.
#
