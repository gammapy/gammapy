"""
Event sampling
==============

Explore how to deal with upper limits on parameters for a significant source,  or flux upper limits
for a non-detected source.

Prerequisites
-------------

It is advisable to understand the general Gammapy modelling and fitting
framework before proceeding with this notebook.

Context
-------

In the study of VHE sources, one often encounters no significant
detection even after long exposures. In that case, it may be useful to
compute flux upper limits. Or, given significant detection of a source,
it may still be difficult to contrain some model parameters, and we can
only provide some confidence intervals.

Proposed approach
-----------------

In this section, we will

- Use an empty observation from the H.E.S.S. DL3 DR1 to understand how
  to quantify non-detections
- Use 6 observations of the blazars PKS 2155-304 taken in 2008 by
  H.E.S.S to contrain the curvature in the spectrum

"""


######################################################################
# Setup
# -----
#
# As usual, let’s start with some general imports…
#

# %matplotlib inline
import matplotlib.pyplot as plt

from gammapy.datasets import MapDataset, Datasets, SpectrumDatasetOnOff
from gammapy.estimators import FluxPointsEstimator, ExcessMapEstimator
from gammapy.modeling import Fit, select_nested_models
from gammapy.modeling.models import (
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    ExpCutoffPowerLawSpectralModel,
    create_crab_spectral_model,
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
# When we don’t detect a source
# -----------------------------
#
# 
#


######################################################################
# Now, load and inspect the dataset. For computational purposes, we have
# already created a `~gammapy.datasets.MapDataset` from observation id `20275` from the
# public H.E.S.S. data release and stored it in `$GAMMAPY_DATA`
#

dataset = MapDataset.read("$GAMMAPY_DATA/empty_dl4/empty-dl4.fits.gz")
dataset.peek()
plt.show()


######################################################################
# We will first use the `~gammapy.estimators.ExcessMapEstimator` for a quick check to see if
# there are any hotspots in the field. You may also use the
# `~gammapy.estimators.TSMapEstimator`.
#

estimator = ExcessMapEstimator(sum_over_energy_groups=True, selection_optional="all")
res1 = estimator.run(dataset)
res1.sqrt_ts.plot(add_cbar=True)
plt.show()


######################################################################
# The significance map looks rather flat! Suppose we were expecting a
# source at the centre of our map. Lets try see if we can fit a point
# source there.
# Note that it is necessary to constrain the range of the position, otherwise the fit might not converge. 

spectral = PowerLawSpectralModel()
spatial = PointSpatialModel(frame="icrs")
spatial.lon_0.value = 187.0
spatial.lat_0.value = 2.6
spatial.lat_0.min = 1.0
spatial.lat_0.max = 4.0
spatial.lon_0.min = 185
spatial.lon_0.max = 187

skymodel = SkyModel(spatial_model=spatial, spectral_model=spectral, name="test")

dataset.models = skymodel
fit = Fit()
res = fit.run(dataset)
print(res.models)


######################################################################
# We can see that there is a slight negative excess in the centre, and
# thus, the fitted model has a negative amplitude. We can use the
# `~gammapy.modeling.select_nested_models` function to perform a likelihood ratio test to
# see if this is significant.
#

LLR = select_nested_models(
    Datasets(dataset), parameters=[skymodel.parameters["amplitude"]], null_values=[0]
)
print(LLR)


######################################################################
# You can see that the `ts ~ 4.7`, thus suggesting that the observed
# fluctuations are not significant over the background. Now, we will
# estimate the differential upper limits of the source.
#
# Here, it is important to **set a reasonable model** on the dataset
# before proceeding with the estimator. This model can come from
# measurements from other instruments, be an extrapolation of the flux
# observed at other wavelengths, come from theoretical estimations, etc. A
# model with a negative amplitude as obtained above should not be
# used.
#
# Note that the computed upper limits can depend on the assumed model. The
# values of the amplitude should not matter as long as the values are not
# too absurd. Here, we compute the 3-simga upper limits for assuming a
# spectral index of 2.0
#

model1 = skymodel.copy(name="model1")
model1.parameters["amplitude"].value = 1e-14
model1.parameters["index"].value = 2.0

energy_edges = dataset.geoms["geom"].axes["energy"].edges
fp_est = FluxPointsEstimator(
    selection_optional="all", energy_edges=energy_edges, n_sigma_ul=3
)
fp_est.norm.scan_min = 5
fp_est.norm.scan_max = 100  # set this to a high value for large upper limit

dataset.models = model1
fp1 = fp_est.run(dataset)


fp1.plot(sed_type="dnde")
plt.show()


######################################################################
# We can then ask, would I have seen my source given this irf/ exposure
# time? The `FluxPointsEstimator` can be used to obtain the sensitivity,
# which can be compared to the expected flux. We have the 5-sigma
# sensitivity here, which can be configured using `n_sigma_sensitivity`
# on init. Lets see if we would have seen if a Crab-like source was
# present in the center.
#

crab_model = create_crab_spectral_model()

fp1.flux_sensitivity.plot(label="sensitivity")
crab_model.plot(
    energy_bounds=fp1.geom.axes["energy"], sed_type="flux", label="Crab spectrum"
)
plt.legend()
plt.show()


######################################################################
# Thus, a Crab-like source should have been above our sensitivity till
# around ~ 4 TeV
#


######################################################################
# Constraining model upper limits
# -------------------------------
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

spectral1 = ExpCutoffPowerLawSpectralModel()
spectral1.amplitude.value = 5e-12
spectral1.alpha.value = 2.0

model_pks = SkyModel(spectral1, name="model_pks")
dataset_onoff.models = model_pks

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
# Thus, this dataset does not yield a significant spectral cutoff in
# PKS2155-304. In particular, we cannot constrain the lower limit from
# this profile. We can compute the limits of the cutoff from the pdf.
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
# print(f"{conf_level*100:.2f}% upper limit: {1/LL:.4f} ({unit.to_string()})")


######################################################################
# Since our dataset actually starts from `200 GeV`, all that we did from
# this analysis is rule out any cut-off features!
#

ax = plt.gca()
ax.plot(1.0 / values, loglike - np.min(loglike))
ax.set_xlabel("Energy cutoff (TeV)")
ax.set_ylabel(r"$\Delta$TS")

ax.axhline(9, ls="dashed", color="black")
ax.axvline(1.0 / UL, ls="dotted", color="brown")
plt.title("Cutoff likelihood profile", fontsize=20, y=1.05)
plt.xscale("log")


######################################################################
# This logic can be extended to any spectral or spatial feature. As an
# exercise, try to compute the 95% spatial extent on the MSH 1552 dataset
# used for the ring background notebook.
#
