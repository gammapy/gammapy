# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Computing flux upper limits
============

Explore how to deal with flux upper limits for a non-detected source.

Prerequisites
-------------

It is advisable to understand the general Gammapy modelling and fitting
framework before proceeding with this notebook, eg doc:`docs/user-guide/modeling`.

Context
-------

In the study of VHE sources, one often encounters no significant
detection even after long exposures. In that case, it may be useful to
compute flux upper limits for the said target consistent with the observation.

Proposed approach
-----------------

In this section, we will use an empty observation from the H.E.S.S. DL3 DR1 to understand how
to quantify non-detections. We will
- Compute excess and significance maps
- Perform a source model fit and do a likelihood ratio test
- Compute differential upper limits
- Compute integral upper limits
- Look at the differential sensitivity of our observation/analysis
"""

######################################################################
# Setup
# -----
#
# As usual, let’s start with some general imports…
#

# %matplotlib inline
import matplotlib.pyplot as plt

import astropy.units as u

from gammapy.datasets import MapDataset, Datasets
from gammapy.estimators import FluxPointsEstimator, ExcessMapEstimator
from gammapy.modeling import Fit, select_nested_models
from gammapy.modeling.models import (
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    create_crab_spectral_model,
)


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
# For computational purposes, we have
# already created a `~gammapy.datasets.MapDataset` from observation id `20275` from the
# public H.E.S.S. data release and stored it in `$GAMMAPY_DATA`
#

dataset = MapDataset.read("$GAMMAPY_DATA/datasets/empty-dl4/empty-dl4.fits.gz")
dataset.peek()
plt.show()

######################################################################
# Compute excess maps
# -------------------
#
# We will first use the `~gammapy.estimators.ExcessMapEstimator` for a quick check to see if
# there are any hotspots in the field. You may also use the
# `~gammapy.estimators.TSMapEstimator`.

estimator = ExcessMapEstimator(
    sum_over_energy_groups=True,
    selection_optional="all",
    correlate_off=True,
    correlation_radius=0.1 * u.deg,
)

lima_maps = estimator.run(dataset)

significance_map = lima_maps["sqrt_ts"]
excess_map = lima_maps["npred_excess"]

# We can plot the excess and significance maps
fig, (ax1, ax2) = plt.subplots(
    figsize=(11, 4), subplot_kw={"projection": lima_maps.geom.wcs}, ncols=2
)
ax1.set_title("Significance map")
significance_map.plot(ax=ax1, add_cbar=True)
ax2.set_title("Excess map")
excess_map.plot(ax=ax2, add_cbar=True)
plt.show()

######################################################################
# The significance map looks rather flat! You can plot a histogram of the
# significance distribution to confirm that it is a standard Gaussian. Departure
# from the same can suggest the presence of gamma-ray sources, or can also
# originate from incorrect modeling of the residual hadronic background.
# We will now do a likelihood fit to search for significant emission.
#
#

###############################################################################
# Perform a fit
# -------------
#
# Suppose we were expecting a
# source at the centre of our map. Let's try see if we can fit a point
# source there.
# Note that it is necessary to constrain the range of the position, otherwise the fit might not converge.
#


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
# It is good to ensure that the fit has converged

print(res.minuit)

######################################################################
# We can see that there is a slight negative excess in the centre, and
# thus, the fitted model has a negative amplitude. We can use the
# `~gammapy.modeling.select_nested_models` function to perform a likelihood ratio test to
# see if this number is significant (See :doc:`docs/user-guide/howto.rst`).
#

LLR = select_nested_models(
    Datasets(dataset), parameters=[skymodel.parameters["amplitude"]], null_values=[0]
)
print(LLR)

######################################################################
# You can see that the `ts ~ 4.7`, thus suggesting that the observed
# fluctuations are not significant over the background. Note the here we have
# only 1 free parameter, and thus, significance = sqrt(ts) ~ 2.2.
# Now, we will estimate the differential upper limits of the source.

###############################################################################
# Differential upper limits
# -------------------------
#
# Here, it is important to **set a reasonable model** on the dataset
# before proceeding with the `~gammapy.estimators.FluxPointsEstimator` estimator. This model can come from
# measurements from other instruments, be an extrapolation of the flux
# observed at other wavelengths, come from theoretical estimations, etc. A
# model with a negative amplitude as obtained above should not be
# used.
#
# Note that the computed upper limits can depend on the spectral parameters of the assumed model.
# Here, we compute the 3-simga upper limits for assuming a spectral index of 2.0.
# We also fix the spatial parameters of the model to prevent the minimiser
# from wandering off to different regions in the FoV.
#


model1 = skymodel.copy(name="model1")
model1.parameters["amplitude"].value = 1e-14
model1.parameters["index"].value = 2.0
model1.freeze(model_type="spatial")

energy_edges = dataset.geoms["geom"].axes["energy"].edges
fp_est = FluxPointsEstimator(
    selection_optional="all", energy_edges=energy_edges, n_sigma_ul=3
)
fp_est.norm.scan_min = 5
fp_est.norm.scan_max = 100  # set this to a high value for large upper limits

dataset.models = model1
fp1 = fp_est.run(dataset)


fp1.plot(sed_type="dnde")
plt.show()


######################################################################
# Integral upper limits
# ---------------------
#
# To compute the integral upper limits between certain energies,
# we can simply run  `~gammapy.estimators.FluxPointsEstimator`
# with one bin in energy

emin = energy_edges[0]
emax = energy_edges[-1]
est2 = FluxPointsEstimator(selection_optional=["ul"], energy_edges=[emin, emax])
fp2 = est2.run(dataset)
print(
    f"Integral upper limit between ${emin} and ${emax} is ${fp2.flux_ul.quantity.ravel()[0]}"
)

######################################################################
# Note that this can be different the from correlated upper limits computed with the `ExcessMapEstimator`

lima_maps.flux_ul.plot(add_cbar=True, cmap="viridis")
plt.show()


######################################################################
# Sensitivity estimation
# ----------------------
#
# We can then ask, would I have seen my source given this irf/ exposure
# time? The `~gammapy.estimators.FluxPointsEstimator` can be used to obtain the sensitivity,
# which can be compared to the expected flux. We have the 5-sigma
# sensitivity here, which can be configured using `n_sigma_sensitivity`
# on init. Lets see if we would have seen if a Crab-like source was
# present in the center.
# Note that this computed sensitivity does not take into account the into factors
# like the minimum number of gamma-rays, etc (see :doc:`/tutorials/analysis-1d/cta_sensitivity.py`)
# and is dependent on the analysis configuration.
# We compare this with the know Crab spectrum.

crab_model = create_crab_spectral_model()

fp1.flux_sensitivity.plot(label="sensitivity")
crab_model.plot(
    energy_bounds=fp1.geom.axes["energy"], sed_type="flux", label="Crab spectrum"
)
plt.legend()
plt.show()


######################################################################
# Thus, a Crab-like source should have been above our sensitivity till around ~ 4 TeV
