# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Constraining parameter limits
================

Explore how to deal with upper limits on parameters

Prerequisites
-------------
It is advisable to understand the general Gammapy modelling and fitting framework before proceeding
with this notebook.

Context
-------
Even with significant detection of a source, constraining specific model parameters may remain difficult,
allowing only for the calculation of confidence intervals.

Proposed approach
-----------------

In this section, we will use 6 observations of the blazars PKS 2155-304 taken in 2008 by
  H.E.S.S to constrain the curvature in the spectrum

"""

######################################################################
# Setup
# -----
#
# As usual, let’s start with some general imports…
#

# %matplotlib inline
import matplotlib.pyplot as plt

from gammapy.datasets import SpectrumDatasetOnOff
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    SkyModel,
    ExpCutoffPowerLawSpectralModel,
)
from gammapy.stats.utils import ts_to_sigma, sigma_to_ts
import numpy as np


######################################################################
# Check setup
# -----------
#

from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()

######################################################################
# Load observation
# -----------
#
# We will now use a precomputed blazar dataset to see how to contrain
# model parameters. Detailed modeling of this dataset may be found in the
# :doc:`/tutorials/utorials/analysis-1d/ebl` notebook. We will try to
# constrain the spectral cutoff in the source. To see
#

dataset_onoff = SpectrumDatasetOnOff.read(
    "$GAMMAPY_DATA/PKS2155-steady/pks2155-304_steady.fits.gz"
)
dataset_onoff.peek()
plt.show()


######################################################################
# Fit spectrum
# -----------
#
# Fit a spectral model with cutoff
#

spectral1 = ExpCutoffPowerLawSpectralModel()
spectral1.amplitude.value = 5e-12
spectral1.alpha.value = 2.0

model_pks = SkyModel(spectral1, name="model_pks")
dataset_onoff.models = model_pks

fit = Fit()
res_pks = fit.run(dataset_onoff)
print(res_pks.models)


######################################################################
# We see that the parameter `lambda_` is not well constrained. In this
# case, it is helpful to look at the likelihood profile of the parameter.
# We will use the `fit.stat_profile` function, and refit the other free
# parameters in the process by setting `reoptimize=True`
#

parameter = model_pks.parameters["lambda_"]
parameter.scan_n_values = 25
parameter.scan_min = 0.05
parameter.scan_max = 5
parameter.interp = "log"
profile = fit.stat_profile(datasets=dataset_onoff, parameter=parameter, reoptimize=True)


######################################################################
# `profile` is a dict storing the values of `lambda_` and the
# likelihood value. Lets try to visualise it.
#

values = profile["model_pks.spectral.lambda__scan"]
loglike = profile["stat_scan"]

ax = plt.gca()
ax.plot(values, loglike - np.min(loglike))
ax.set_xlabel("Cutoff value (1/TeV)")
ax.set_ylabel(r"$\Delta$TS")
secay = ax.secondary_yaxis("right", functions=(ts_to_sigma, sigma_to_ts))
secay.set_ylabel("Significance [$\sigma$]")
plt.title("Cutoff likelihood profile", fontsize=20, y=1.05)
plt.show()


######################################################################
# Constrain the limits on the parameter
# ---------------
# Thus, this dataset does not yield a significant spectral cutoff in
# PKS2155-304. In particular, we cannot constrain the lower limit from
# this profile. We can compute the limits of the cutoff from the Probability Density Function (PDF).
#

from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid


def likelihood_to_pdf(theta_values, log_likelihoods):
    """
    Convert log-likelihood profile into a normalized PDF.
    """
    log_likelihoods = log_likelihoods
    likelihood = np.exp(1.0 / log_likelihoods)

    # Normalize to make it a proper PDF
    area = np.trapz(likelihood, theta_values)
    pdf = likelihood / area
    return pdf


def compute_upper_limit(theta_values, pdf, confidence_level=0.95):
    """
    Compute the upper limit for a one-sided confidence interval.
    """
    cdf = cumulative_trapezoid(pdf, theta_values, initial=0)
    cdf_interp = CubicSpline(cdf, theta_values)
    upper_limit = cdf_interp(confidence_level)
    return float(upper_limit)


def compute_lower_limit(theta_values, pdf, confidence_level=0.95):
    """
    Compute the lower limit for a one-sided confidence interval.
    """
    cdf = cumulative_trapezoid(pdf, theta_values, initial=0)
    cdf_interp = CubicSpline(cdf, theta_values)
    lower_limit = cdf_interp(1 - confidence_level)
    return float(lower_limit)


pdf = likelihood_to_pdf(values, loglike)
plt.plot(values, pdf)
plt.xlabel("Cutoff value (1/TeV)")
plt.ylabel("pdf")
plt.show()

conf_level = 0.68
# Compute one-sided limits
UL = compute_upper_limit(values, pdf, confidence_level=conf_level)
LL = compute_lower_limit(values, pdf, confidence_level=conf_level)
print("lower limit, upper limit: ", LL, UL)

if LL > parameter.value:
    print("lower limit is not constrained")
if UL < parameter.value:
    print("upper limit is not constrained")

unit = parameter.unit**-1  # Give in units of energy since lambda_ = 1/cutoff energy
print(
    f"{conf_level*100:.2f}% lower limit of energy cutoff: {1/UL:.4f} ({unit.to_string()})"
)


######################################################################
# Since our dataset actually starts from `200 GeV`, all that we did from
# this analysis is rule out any cut-off features!
#
# We could also have used `stat_profile_ul_scipy` for computing the UL, which uses rootfing to obtain the n-sigma limits,
# but this can fail if the limits are not constrained.
#

ax = plt.gca()
ax.plot(1.0 / values, loglike - np.min(loglike))
ax.set_xlabel("Energy cutoff (TeV)")
ax.set_ylabel(r"$\Delta$TS")
ax.axvline(1.0 / UL, ls="dotted", color="brown")
plt.title("Cutoff likelihood profile", fontsize=20, y=1.05)
plt.xscale("log")


######################################################################
# This logic can be extended to any spectral or spatial feature. As an
# exercise, try to compute the 95% spatial extent on the MSH 15 52 dataset
# used for the ring background notebook.
#
