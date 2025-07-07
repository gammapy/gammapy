"""
3D map simulation
=================

Simulate a 3D observation of a source with the CTA 1DC response and fit it with the assumed source model.

Prerequisites
-------------

-  Knowledge of 3D extraction and datasets used in gammapy, see for
   example the :doc:`/tutorials/starting/analysis_1` tutorial.

Context
-------

To simulate a specific observation, it is not always necessary to
simulate the full photon list. For many uses cases, simulating directly
a reduced binned dataset is enough: the IRFs reduced in the correct
geometry are combined with a source model to predict an actual number of
counts per bin. The latter is then used to simulate a reduced dataset
using Poisson probability distribution.

This can be done to check the feasibility of a measurement (performance
/ sensitivity study), to test whether fitted parameters really provide a
good fit to the data etc.

Here we will see how to perform a 3D simulation of a CTA observation,
assuming both the spectral and spatial morphology of an observed source.

**Objective: simulate a 3D observation of a source with CTA using the
CTA 1DC response and fit it with the assumed source model.**

Proposed approach
-----------------

Here we can’t use the regular observation objects that are connected to
a `~gammapy.data.DataStore`. Instead, we will create a fake
`~gammapy.data.Observation` that contain some pointing information and
the CTA 1DC IRFs (that are loaded with `~gammapy.irf.load_irf_dict_from_file`).

Next, we will create a `~gammapy.datasets.MapDataset` geometry through
the `~gammapy.makers.MapDatasetMaker`.

Finally, we will define a model consisting of a
`~gammapy.modeling.models.PowerLawSpectralModel` and a
`~gammapy.modeling.models.GaussianSpatialModel`. This model will be assigned to
the dataset and fake the count data.

"""

######################################################################
# Imports and versions
# --------------------
#

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

# %matplotlib inline
from IPython.display import display
from gammapy.data import FixedPointingInfo, Observation, observatory_locations
from gammapy.datasets import MapDataset
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    FoVBackgroundModel,
    GaussianSpatialModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

######################################################################
# Simulation
# ----------
#


######################################################################
# We will simulate using the CTA-1DC IRFs shipped with gammapy
#

# Loading IRFs
irfs = load_irf_dict_from_file(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)

# Define the observation parameters (typically the observation duration and the pointing position):
livetime = 2.0 * u.hr
pointing_position = SkyCoord(0, 0, unit="deg", frame="galactic")
# We want to simulate an observation pointing at a fixed position in the sky.
# For this, we use the `FixedPointingInfo` class
pointing = FixedPointingInfo(
    fixed_icrs=pointing_position.icrs,
)

# Define map geometry for binned simulation
energy_reco = MapAxis.from_edges(
    np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0),
    binsz=0.02,
    width=(6, 6),
    frame="galactic",
    axes=[energy_reco],
)
# It is usually useful to have a separate binning for the true energy axis
energy_true = MapAxis.from_edges(
    np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy_true", interp="log"
)

empty = MapDataset.create(geom, name="dataset-simu", energy_axis_true=energy_true)

# Define sky model to used simulate the data.
# Here we use a Gaussian spatial model and a Power Law spectral model.
spatial_model = GaussianSpatialModel(
    lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.3 deg", frame="galactic"
)
spectral_model = PowerLawSpectralModel(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model_simu = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    name="model-simu",
)

bkg_model = FoVBackgroundModel(dataset_name="dataset-simu")

models = Models([model_simu, bkg_model])
print(models)


######################################################################
# Now, comes the main part of dataset simulation. We create an in-memory
# observation and an empty dataset. We then predict the number of counts
# for the given model, and Poisson fluctuate it using ``fake()`` to make
# a simulated counts maps. Keep in mind that it is important to specify
# the ``selection`` of the maps that you want to produce
#

# Create an in-memory observation
location = observatory_locations["ctao_south"]
obs = Observation.create(
    pointing=pointing, livetime=livetime, irfs=irfs, location=location
)
print(obs)

# Make the MapDataset
maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])

maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)

dataset = maker.run(empty, obs)
dataset = maker_safe_mask.run(dataset, obs)
print(dataset)

# Add the model on the dataset and Poisson fluctuate
dataset.models = models
dataset.fake()
# Do a print on the dataset - there is now a counts maps
print(dataset)


######################################################################
# Now use this dataset as you would in all standard analysis. You can plot
# the maps, or proceed with your custom analysis. In the next section, we
# show the standard 3D fitting as in :doc:`/tutorials/analysis-3d/analysis_3d`
# tutorial.
#

# To plot, eg, counts:
dataset.counts.smooth(0.05 * u.deg).plot_interactive(add_cbar=True, stretch="linear")
plt.show()


######################################################################
# Fit
# ---
#
# In this section, we do a usual 3D fit with the same model used to
# simulated the data and see the stability of the simulations. Often, it
# is useful to simulate many such datasets and look at the distribution of
# the reconstructed parameters.
#

models_fit = models.copy()

# We do not want to fit the background in this case, so we will freeze the parameters
models_fit["dataset-simu-bkg"].spectral_model.norm.frozen = True
models_fit["dataset-simu-bkg"].spectral_model.tilt.frozen = True

dataset.models = models_fit
print(dataset.models)

# %%time
fit = Fit(optimize_opts={"print_level": 1})
result = fit.run(datasets=[dataset])

dataset.plot_residuals_spatial(method="diff/sqrt(model)", vmin=-0.5, vmax=0.5)
plt.show()


######################################################################
# Compare the injected and fitted models:
#

print(
    "True model: \n",
    model_simu,
    "\n\n Fitted model: \n",
    models_fit["model-simu"],
)


######################################################################
# Get the errors on the fitted parameters from the parameter table
#

display(result.parameters.to_table())
