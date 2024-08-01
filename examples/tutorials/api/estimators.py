"""
Estimators
==========

This tutorial provides an overview of the `Estimator` API. All estimators live in the
`gammapy.estimators` sub-module offers algorithms and classes for high-level flux and
significance estimation, through a common functionality such as estimation of flux points,
lightcurves, flux maps and profiles via a common API.


Key Features
------------

-  **Hypothesis Testing**: Estimations are based on testing a reference model
   against a null hypothesis, deriving flux and significance values.

-  **Estimation via Two Methods**:

   -   **Model Fitting (Forward Folding)**: Refit the flux of a model component
       within specified energy, time, or spatial regions.
   -   **Excess Calculation (Backward Folding)**: Use the analytical solution by Li and Ma
       for significance based on excess counts, currently available in `~gammapy.estimators.ExcessMapEstimator`.

For further information on these details please refer to :doc:`</user-guide/estimators>`.

The setup
---------

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from IPython.display import display
from gammapy.datasets import SpectrumDatasetOnOff, Datasets
from gammapy.estimators import FluxPointsEstimator
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel
from gammapy.utils.scripts import make_path


######################################################################
# Flux Points Estimation
# ----------------------
#
# We start with a simple example for flux points estimation taking multiple datasets into account.
# First we read the pre-computed datasets from `$GAMMAPY_DATA`.
#

datasets = Datasets()

path = make_path("$GAMMAPY_DATA/joint-crab/spectra/hess/")

for filename in path.glob("pha_obs*.fits"):
    dataset = SpectrumDatasetOnOff.read(filename)
    datasets.append(dataset)

print(datasets)

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
# The `~gammapy.estimators.FluxPointsEstimator` estimates flux points for a given list of datasets,
# energies and spectral model. Now we prepare the flux point estimation:
#

energy_edges = np.geomspace(0.7, 100, 9) * u.TeV

fp_estimator = FluxPointsEstimator(
    source="crab",
    energy_edges=energy_edges,
)

# %%time
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
# From the above we can see that we access to many quantities. We can also access
# the quantities names through `fp_result.available_quantities`.
# Here we show how you can plot a different plot type and define the axes units.

ax = plt.subplot()
ax.xaxis.set_units(u.eV)
ax.yaxis.set_units(u.Unit("TeV cm-2 s-1"))
fp_result.plot(ax=ax, sed_type="e2dnde")
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
# -  ``counts``: predicted counts from the null hypothesis.
# -  ``npred``: predicted number of counts from best fit hypothesis.
# -  ``npred_excess``: predicted number of excess counts from best fit hypothesis.
#
# The `~gammapy.maps.region.ndmap.RegionNDMap` allows for plotting of multidimensional data
# as well, by specifying the primary `axis_name`:


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
# A fully configured estimation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The following code shows fully configured flux points estimation.
# Firstly we define the `backend` for the fit:


fit = Fit(
    optimize_opts={"backend": "minuit"},
    confidence_opts={"backend": "scipy"},
)

######################################################################
# The various quantities utilised in this tutorial are described here:
#
# -  ``source``: which source from the model to compute the flux points for
# -  ``energy_edges``: edges of the flux points energy bins
# -  ``n_sigma``: number of sigma for the flux error
# -  ``n_sigma_ul``: the number of sigma for the flux upper limits
# -  ``selection_optional``: what additional maps to compute
# -  ``fit``: the fit instance (as defined above)
#
# **Important note**: the `energy_edges` are taken from the parent dataset energy bins,
# which may not exactly match the output bins. Specific binning must be defined in the
# parent dataset geometry to achieve that.
#


fp_estimator_config = FluxPointsEstimator(
    source="crab",
    energy_edges=energy_edges,
    n_sigma=1,
    n_sigma_ul=2,
    selection_optional="all",
    fit=fit,
)

print(fp_estimator_config)


######################################################################
#

# %%time
fp_result_config = fp_estimator_config.run(datasets=datasets)

print(fp_result_config)

######################################################################
#
fp_result_config.plot(sed_type="e2dnde", color="tab:orange")
fp_result_config.plot_ts_profiles(sed_type="e2dnde")
plt.show()
