"""
Flux point fitting
==================

Fit spectral models to combined Fermi-LAT and IACT flux points tables.


Prerequisites
-------------

-  Some knowledge about retrieving information from catalogs, see `the
   catalogs tutorial <../../api/catalog.ipynb>`__

Context
-------

Some high level studies do not rely on reduced datasets with their IRFs
but directly on higher level products such as flux points. This is not
ideal because flux points already contain some hypothesis for the
underlying spectral shape and the uncertainties they carry are usually
simplified (e.g. symmetric gaussian errors). Yet, this is an efficient
way to combine heterogeneous data.

**Objective: fit spectral models to combined Fermi-LAT and IACT flux
points.**

Proposed approach
-----------------

Here we will load, the spectral points from Fermi-LAT and TeV catalogs
and fit them with various spectral models to find the best
representation of the wide band spectrum.

The central class we’re going to use for this example analysis is:

-  `~gammapy.datasets.FluxPointsDataset`

In addition we will work with the following data classes:

-  `~gammapy.estimators.FluxPoints`
-  `~gammapy.catalog.SourceCatalogGammaCat`
-  `~gammapy.catalog.SourceCatalog3FHL`
-  `~gammapy.catalog.SourceCatalog3FGL`

And the following spectral model classes:

-  `~gammapy.modeling.models.PowerLawSpectralModel`
-  `~gammapy.modeling.models.ExpCutoffPowerLawSpectralModel`
-  `~gammapy.modeling.models.LogParabolaSpectralModel`

"""


######################################################################
# Setup
# -----
# 
# Let us start with the usual IPython notebook and Python imports:
# 

# %matplotlib inline
import matplotlib.pyplot as plt

from astropy import units as u
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    LogParabolaSpectralModel,
    SkyModel,
)
from gammapy.datasets import FluxPointsDataset, Datasets
from gammapy.catalog import CATALOG_REGISTRY
from gammapy.modeling import Fit

######################################################################
# Load spectral points
# --------------------
# 
# For this analysis we choose to work with the source ‘HESS J1507-622’ and
# the associated Fermi-LAT sources ‘3FGL J1506.6-6219’ and ‘3FHL
# J1507.9-6228e’. We load the source catalogs, and then access source of
# interest by name:
# 

catalog_3fgl = CATALOG_REGISTRY.get_cls("3fgl")()
catalog_3fhl = CATALOG_REGISTRY.get_cls("3fhl")()
catalog_gammacat = CATALOG_REGISTRY.get_cls("gamma-cat")()

source_fermi_3fgl = catalog_3fgl["3FGL J1506.6-6219"]
source_fermi_3fhl = catalog_3fhl["3FHL J1507.9-6228e"]
source_gammacat = catalog_gammacat["HESS J1507-622"]


######################################################################
# The corresponding flux points data can be accessed with ``.flux_points``
# attribute:
# 

dataset_gammacat = FluxPointsDataset(
    data=source_gammacat.flux_points, name="gammacat"
)
dataset_gammacat.data.to_table(sed_type="dnde", formatted=True)

dataset_3fgl = FluxPointsDataset(
    data=source_fermi_3fgl.flux_points, name="3fgl"
)
dataset_3fgl.data.to_table(sed_type="dnde", formatted=True)

dataset_3fhl = FluxPointsDataset(
    data=source_fermi_3fhl.flux_points, name="3fhl"
)
dataset_3fhl.data.to_table(sed_type="dnde", formatted=True)


######################################################################
# Power Law Fit
# -------------
# 
# First we start with fitting a simple
# `~gammapy.modeling.models.PowerLawSpectralModel`.
# 

pwl = PowerLawSpectralModel(
    index=2, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV"
)
model = SkyModel(spectral_model=pwl, name="j1507-pl")


######################################################################
# After creating the model we run the fit by passing the ``flux_points``
# and ``model`` objects:
# 

datasets = Datasets([dataset_gammacat, dataset_3fgl, dataset_3fhl])
datasets.models = model
print(datasets)

fitter = Fit()
result_pwl = fitter.run(datasets=datasets)


######################################################################
# And print the result:
# 

print(result_pwl)

print(model)


######################################################################
# Finally we plot the data points and the best fit model:
# 

ax = plt.subplot()
ax.yaxis.set_units(u.Unit("TeV cm-2 s-1"))

kwargs = {"ax": ax, "sed_type": "e2dnde"}

for d in datasets:
    d.data.plot(label=d.name, **kwargs)

energy_bounds = [1e-4, 1e2] * u.TeV
pwl.plot(energy_bounds=energy_bounds, color="k", **kwargs)
pwl.plot_error(energy_bounds=energy_bounds, **kwargs)
ax.set_ylim(1e-13, 1e-11)
ax.set_xlim(energy_bounds)
ax.legend()


######################################################################
# Exponential Cut-Off Powerlaw Fit
# --------------------------------
# 
# Next we fit an
# `~gammapy.modeling.models.ExpCutoffPowerLawSpectralModel` law to the
# data.
# 

ecpl = ExpCutoffPowerLawSpectralModel(
    index=1.8,
    amplitude="2e-12 cm-2 s-1 TeV-1",
    reference="1 TeV",
    lambda_="0.1 TeV-1",
)
model = SkyModel(spectral_model=ecpl, name="j1507-ecpl")


######################################################################
# We run the fitter again by passing the flux points and the model
# instance:
# 

datasets.models = model
result_ecpl = fitter.run(datasets=datasets)
print(model)


######################################################################
# We plot the data and best fit model:
# 

ax = plt.subplot()

kwargs = {"ax": ax, "sed_type": "e2dnde"}

ax.yaxis.set_units(u.Unit("TeV cm-2 s-1"))

for d in datasets:
    d.data.plot(label=d.name, **kwargs)

ecpl.plot(energy_bounds=energy_bounds, color="k", **kwargs)
ecpl.plot_error(energy_bounds=energy_bounds, **kwargs)
ax.set_ylim(1e-13, 1e-11)
ax.set_xlim(energy_bounds)
ax.legend()


######################################################################
# Log-Parabola Fit
# ----------------
# 
# Finally we try to fit a
# `~gammapy.modeling.models.LogParabolaSpectralModel` model:
# 

log_parabola = LogParabolaSpectralModel(
    alpha=2, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV", beta=0.1
)
model = SkyModel(spectral_model=log_parabola, name="j1507-lp")

datasets.models = model
result_log_parabola = fitter.run(datasets=datasets)
print(model)

ax = plt.subplot()

kwargs = {"ax": ax, "sed_type": "e2dnde"}

ax.yaxis.set_units(u.Unit("TeV cm-2 s-1"))

for d in datasets:
    d.data.plot(label=d.name, **kwargs)

log_parabola.plot(energy_bounds=energy_bounds, color="k", **kwargs)
log_parabola.plot_error(energy_bounds=energy_bounds, **kwargs)
ax.set_ylim(1e-13, 1e-11)
ax.set_xlim(energy_bounds)
ax.legend()


######################################################################
# Exercises
# ---------
# 
# -  Fit a `~gammapy.modeling.models.PowerLaw2SpectralModel` and
#    `~gammapy.modeling.models.ExpCutoffPowerLaw3FGLSpectralModel` to
#    the same data.
# -  Fit a `~gammapy.modeling.models.ExpCutoffPowerLawSpectralModel`
#    model to Vela X (‘HESS J0835-455’) only and check if the best fit
#    values correspond to the values given in the Gammacat catalog
# 


######################################################################
# What next?
# ----------
# 
# This was an introduction to SED fitting in Gammapy.
# 
# -  If you would like to learn how to perform a full Poisson maximum
#    likelihood spectral fit, please check out the `spectrum
#    analysis <spectral_analysis.ipynb>`__ tutorial.
# -  To learn how to combine heterogeneous datasets to perform a
#    multi-instrument forward-folding fit see the `MWL analysis
#    tutorial <../3D/analysis_mwl.ipynb>`__
# 

