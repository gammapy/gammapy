"""
Estimators
==========

This tutorial provides an overview of the ``Estimator`` API. All estimators live in the
`gammapy.estimators` sub-module, offering a range of algorithms and classes for high-level flux and
significance estimation. This is accomplished through a common functionality allowing the estimation of
flux points, light curves, flux maps and profiles via a common API.



Key Features
------------

-  **Hypothesis Testing**: Estimations are based on testing a reference model
   against a null hypothesis, deriving flux and significance values.

-  **Estimation via Two Methods**:

   -   **Model Fitting (Forward Folding)**: Refit the flux of a model component
       within specified energy, time, or spatial regions.
   -   **Excess Calculation (Backward Folding)**: Use the analytical solution by Li and Ma
       for significance based on excess counts, currently available in `~gammapy.estimators.ExcessMapEstimator`.

For further information on these details please refer to :doc:`/user-guide/estimators`.

The setup
---------

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from IPython.display import display
from gammapy.datasets import SpectrumDatasetOnOff, Datasets, MapDataset
from gammapy.estimators import (
    FluxPointsEstimator,
    ExcessMapEstimator,
    FluxPoints,
)
from gammapy.modeling import Fit, Parameter
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel
from gammapy.utils.scripts import make_path


######################################################################
# Flux Points Estimation
# ----------------------
#
# We start with a simple example for flux points estimation taking multiple datasets into account.
# In this section we show the steps to estimate the flux points.
# First we read the pre-computed datasets from `$GAMMAPY_DATA`.
#

datasets = Datasets()
path = make_path("$GAMMAPY_DATA/joint-crab/spectra/hess/")

for filename in path.glob("pha_obs*.fits"):
    dataset = SpectrumDatasetOnOff.read(filename)
    datasets.append(dataset)

######################################################################
# Next we define a spectral model and set it on the datasets:
#

pwl = PowerLawSpectralModel(index=2.7, amplitude="5e-11  cm-2 s-1 TeV-1")
datasets.models = SkyModel(spectral_model=pwl, name="crab")

######################################################################
# Before using the estimators, it is necessary to first ensure that the model is properly
# fitted. This applies to all scenarios, including light curve estimation. To optimize the
# model parameters to best fit the data we utilise the following:
#

fit = Fit()
fit_result = fit.optimize(datasets=datasets)
print(fit_result)

######################################################################
# A fully configured Flux Points Estimation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `~gammapy.estimators.FluxPointsEstimator` estimates flux points for a given list of datasets,
# energies and spectral model. The most simple way to call the estimator is by defining both
# the name of the ``source`` and its ``energy_edges``.
# Here we prepare a full configuration of the flux point estimation.
# Firstly we define the ``backend`` for the fit:
#

fit = Fit(
    optimize_opts={"backend": "minuit"},
    confidence_opts={"backend": "scipy"},
)

######################################################################
# Define the fully configured flux points estimator:
#

energy_edges = np.geomspace(0.7, 100, 9) * u.TeV
norm = Parameter(name="norm", value=1.0, interp="log")

fp_estimator = FluxPointsEstimator(
    source="crab",
    energy_edges=energy_edges,
    n_sigma=1,
    n_sigma_ul=2,
    selection_optional="all",
    fit=fit,
    norm=norm,
)

######################################################################
# The ``norm`` parameter can be adjusted in a few different ways. For example, we can change its
# minimum and maximum values that it scans over, as follows.
#

fp_estimator.norm.scan_min = 0.1
fp_estimator.norm.scan_max = 10

######################################################################
# Note: The default scan range of the norm parameter is between 0.2 to 5. In case the upper
# limit values lie outside this range, nan values will be returned. It may thus be useful to
# increase this range, specially for the computation of upper limits from weak sources.
#
# The various quantities utilised in this tutorial are described here:
#
# -  ``source``: which source from the model to compute the flux points for
# -  ``energy_edges``: edges of the flux points energy bins
# -  ``n_sigma``: number of sigma for the flux error
# -  ``n_sigma_ul``: the number of sigma for the flux upper limits
# -  ``selection_optional``: what additional maps to compute
# -  ``fit``: the fit instance (as defined above)
# -  ``reoptimize``: whether to reoptimize the flux points with other model parameters, aside from the ``norm``
# -  ``norm``: normalisation parameter for the fit
#
# **Important note**: the output ``energy_edges`` are taken from the parent dataset energy bins,
# selecting the bins closest to the requested ``energy_edges``. To match the input bins directly,
# specific binning must be defined based on the parent dataset geometry. This could be done in the following way:
# ``energy_edges = datasets[0].geoms["geom"].axes["energy"].downsample(factor=5).edges``
#


fp_result = fp_estimator.run(datasets=datasets)

######################################################################
# Accessing and visualising the results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(fp_result)

######################################################################
# We can specify the SED type to plot:
#
fp_result.plot(sed_type="dnde")
plt.show()

######################################################################
# We can also access
# the quantities names through ``fp_result.available_quantities``.
# Here we show how you can plot a different plot type and define the axes units,
# we also overlay the TS profile.

ax = plt.subplot()
ax.xaxis.set_units(u.eV)
ax.yaxis.set_units(u.Unit("TeV cm-2 s-1"))
fp_result.plot(ax=ax, sed_type="e2dnde", color="tab:orange")
fp_result.plot_ts_profiles(ax=ax, sed_type="e2dnde")
plt.show()

######################################################################
# The actual data members are N-dimensional `~gammapy.maps.region.ndmap.RegionNDMap` objects. So you can
# also plot them:

print(type(fp_result.dnde))

######################################################################
#
fp_result.dnde.plot()
plt.show()

######################################################################
# From the above, we can see that we access to many quantities.


######################################################################
# Access the data:

print(fp_result.e2dnde.quantity.to("TeV cm-2 s-1"))

######################################################################
#
print(fp_result.dnde.quantity.shape)

######################################################################
#
print(fp_result.dnde.quantity[:, 0, 0])

######################################################################
# Or even extract an energy range:

fp_result.dnde.slice_by_idx({"energy": slice(3, 10)})


######################################################################
# A note on the internal representation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The result contains a reference spectral model, which defines the spectral shape.
# Typically, it is the best fit model:

print(fp_result.reference_model)

######################################################################
# `~gammapy.estimators.FluxPoints` are the represented by the "norm" scaling factor with
# respect to the reference model:

fp_result.norm.plot()
plt.show()

######################################################################
# Dataset specific quantities ("counts like")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# While the flux estimate and associated errors are common to all datasets,
# the result also stores some dataset specific quantities, which can be useful
# for debugging.
# Here we remind the user of the meaning of the forthcoming quantities:
#
# -  ``counts``: predicted counts from the null hypothesis,
# -  ``npred``: predicted number of counts from best fit hypothesis,
# -  ``npred_excess``: predicted number of excess counts from best fit hypothesis.
#
# The `~gammapy.maps.region.ndmap.RegionNDMap` allows for plotting of multidimensional data
# as well, by specifying the primary ``axis_name``:


fp_result.counts.plot(axis_name="energy")
plt.show()

######################################################################
#
fp_result.npred.plot(axis_name="energy")
plt.show()

######################################################################
#
fp_result.npred_excess.plot(axis_name="energy")
plt.show()

######################################################################
# Table conversion
# ~~~~~~~~~~~~~~~~
#
# Flux points can be converted to tables:
#

table = fp_result.to_table(sed_type="flux", format="gadf-sed")
display(table)

######################################################################
#
table = fp_result.to_table(sed_type="likelihood", format="gadf-sed", formatted=True)
display(table)


######################################################################
# Common API
# ----------
# In `GAMMAPY_DATA` we have access to other `~gammapy.estimators.FluxPoints` objects
# which have been created utilising the above method. Here we read the PKS 2155-304 light curve
# and create a `~gammapy.estimators.FluxMaps` object and show the data structure of such objects.
# We emphasize that these follow a very similar structure.
#

######################################################################
# Load the light curve for the PKS 2155-304 as a `~gammapy.estimators.FluxPoints` object.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

lightcurve = FluxPoints.read(
    "$GAMMAPY_DATA/estimators/pks2155_hess_lc/pks2155_hess_lc.fits", format="lightcurve"
)

display(lightcurve.available_quantities)


######################################################################
# Create a `~gammapy.estimators.FluxMaps` object through one of the estimators.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
estimator = ExcessMapEstimator(correlation_radius="0.1 deg")
result = estimator.run(dataset)
display(result)

######################################################################
#
display(result.available_quantities)
