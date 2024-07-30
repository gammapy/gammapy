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

-  **Energy Edges**: Estimators group the parent dataset energy bins based on input energy edges,
   which may not exactly match the output bins. Specific binning must be defined in the parent
   dataset geometry to achieve that.

For further information on these details please refer to :doc:`</user-guide/estimators>`.

The setup
---------

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from IPython.display import display
from gammapy.datasets import MapDataset, SpectrumDatasetOnOff, Datasets
from gammapy.estimators import (
    FluxPointsEstimator,
    FluxProfileEstimator,
    TSMapEstimator,
)
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel, PointSpatialModel
from gammapy.utils.scripts import make_path
from gammapy.utils.regions import make_orthogonal_rectangle_sky_regions
from gammapy.maps import RegionGeom


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
# And optimize the model parameters to best fit the data:
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
# `FluxPoints` are the represented by the "norm" scaling factor with
# respect to the reference model:

fp_result.norm.plot()
plt.show()

######################################################################
# Dataset specific quantities ("counts like")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# While the flux estimate and associated errors are common to all datasets,
# the result also stores some dataset specific quantities, which can be useful
# for debugging. The `~gammapy.maps.region.ndmap.RegionNDMap` allows for plotting of multidimensional data
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
# The following code shows fully configured flux points estimation:


fit = Fit(
    optimize_opts={"backend": "minuit"},
    confidence_opts={"backend": "scipy"},
)

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


######################################################################
# Flux Map Estimation
# -------------------

dataset_cta = MapDataset.read(
    "$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz", name="cta_dataset"
)

plt.figure(figsize=(12, 6))
counts_image = dataset_cta.counts.sum_over_axes()
counts_image.smooth("0.05 deg").plot(stretch="linear", add_cbar=True)
plt.show()

######################################################################
# Estimator configuration
# ~~~~~~~~~~~~~~~~~~~~~~~
#

model = SkyModel(
    spectral_model=PowerLawSpectralModel(), spatial_model=PointSpatialModel()
)

map_estimator = TSMapEstimator(
    model=model,
    energy_edges=[0.1, 0.3, 1, 3, 10] * u.TeV,
    n_sigma=1,
    n_sigma_ul=2,
    selection_optional=None,
    n_jobs=8,
    kernel_width=1 * u.deg,
    sum_over_energy_groups=True,
)

print(map_estimator)

######################################################################
# %%time
map_result = map_estimator.run(dataset_cta)

######################################################################
# Accessing and visualising results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Below we print the result of the `~gammapy.estimators.TSMapEstimator`. We have access to a number of
# different quantities, as shown below. We can also access the quantities names
# through `map_result.available_quantities`.
#

print(map_result)

######################################################################
#
print(type(map_result.dnde))

######################################################################
#
map_result.dnde.plot_grid(stretch="sqrt", ncols=2, add_cbar=True, figsize=(16, 10))
plt.show()

# sphinx_gallery_thumbnail_number = 10


######################################################################
#
map_result.sqrt_ts.plot_grid(add_cbar=True, ncols=2, figsize=(16, 10))
plt.show()

######################################################################
# Again the data is represented internally by a reference model and "norm" factors:

print(map_result.reference_model)

######################################################################
#
map_result.norm.plot_grid(add_cbar=True, ncols=2, stretch="sqrt", figsize=(16, 10))
plt.show()

######################################################################
#
position = SkyCoord("0d", "0d", frame="galactic")
flux_points = map_result.get_flux_points(position=position)

print(flux_points)

######################################################################
#
flux_points.plot(sed_type="e2dnde")
plt.show()

######################################################################
# This is how the maps are serialised to FITS:

hdulist = map_result.to_hdulist(sed_type="dnde")
hdulist.info()


######################################################################
# Flux Map Estimation
# -------------------
#
# Finally we take a quick look at the `~gammapy.estimators.FluxProfileEstimator`.
# For this we first define the profile bins as a list of `~regions.Region` objects:

regions = make_orthogonal_rectangle_sky_regions(
    start_pos=SkyCoord("2d", "0d", frame="galactic"),
    end_pos=SkyCoord("358d", "0d", frame="galactic"),
    wcs=counts_image.geom.wcs,
    height="1 deg",
    nbin=31,
)

geom = RegionGeom.create(region=regions)
ax = counts_image.smooth("0.1 deg").plot()
geom.plot_region(ax=ax, color="w")
plt.show()

######################################################################
#
flux_profile_estimator = FluxProfileEstimator(
    regions=regions,
    spectrum=PowerLawSpectralModel(index=2.3),
    energy_edges=[0.1, 0.3, 1, 3, 10] * u.TeV,
    selection_optional="all",
)

# %%time
profile = flux_profile_estimator.run(datasets=dataset_cta)

print(profile)

######################################################################
#
ax = profile.dnde.plot(axis_name="projected-distance")
ax.set_yscale("log")
plt.show()

######################################################################
#
profile.counts.plot(axis_name="projected-distance")
plt.show()

######################################################################
#
profile_3_10_TeV = profile.slice_by_idx({"energy": slice(2, 3)})
ax = profile_3_10_TeV.plot(sed_type="dnde", color="tab:orange")
profile_3_10_TeV.plot_ts_profiles(sed_type="dnde")
ax.set_yscale("linear")
plt.show()

######################################################################
#
sed = profile.slice_by_idx({"projected-distance": 15})
ax = sed.plot(sed_type="e2dnde", color="tab:orange")
sed.plot_ts_profiles(ax=ax, sed_type="e2dnde")
ax.set_ylim(2e-12, 1e-11)
plt.show()

######################################################################
#
table = profile_3_10_TeV.to_table(sed_type="flux", format="profile")
display(table)
