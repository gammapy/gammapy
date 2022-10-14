"""
Simulating and fitting a time varying source
============================================

Simulate and fit a time decaying light curve of a source using the CTA 1DC response.

Prerequisites
-------------

-  To understand how a single binned simulation works, please refer to
   :doc:`/tutorials/analysis-1d/spectrum_simulation` tutorial and 
   :doc:`/tutorials/analysis-3d/simulate_3d` tutorial for 1D and 3D simulations
   respectively.
-  For details of light curve extraction using gammapy, refer to the two
   tutorials :doc:`/tutorials/analysis-time/light_curve` and
   :doc:`/tutorials/analysis-time/light_curve_flare` tutorial.

Context
-------

Frequently, studies of variable sources (eg: decaying GRB light curves,
AGN flares, etc) require time variable simulations. For most use cases,
generating an event list is an overkill, and it suffices to use binned
simulations using a temporal model.

**Objective: Simulate and fit a time decaying light curve of a source
with CTA using the CTA 1DC response**

Proposed approach
-----------------

We will simulate 10 spectral datasets within given time intervals (Good
Time Intervals) following a given spectral (a power law) and temporal
profile (an exponential decay, with a decay time of 6 hr ). These are
then analysed using the light curve estimator to obtain flux points.

Modelling and fitting of lightcurves can be done either - directly on
the output of the `LighCurveEstimator` (at the DL5 level) - fit the
simulated datasets (at the DL4 level)

In summary, necessary steps are:

-  Choose observation parameters including a list of
   `gammapy.data.GTI`
-  Define temporal and spectral models from :ref:model-gallery as per
   science case
-  Perform the simulation (in 1D or 3D)
-  Extract the light curve from the reduced dataset as shown
   in :doc:`/tutorials/analysis-time/light_curve` tutorial.
-  Optionally, we show here how to fit the simulated datasets using a
   source model

Setup
-----

As usual, we’ll start with some general imports…

"""


######################################################################
# Setup
# -----
#

import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

# %matplotlib inline
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


######################################################################
# And some gammapy specific imports
#

from gammapy.data import Observation, observatory_locations
from gammapy.datasets import Datasets, FluxPointsDataset, SpectrumDataset
from gammapy.estimators import LightCurveEstimator
from gammapy.irf import load_cta_irfs
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom, TimeMapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ExpDecayTemporalModel,
    PowerLawSpectralModel,
    SkyModel,
)

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# We first define our preferred time format:
#

TimeMapAxis.time_format = "iso"


######################################################################
# Simulating a light curve
# ------------------------
#
# We will simulate 10 spectra between 300 GeV and 10 TeV using an
# `PowerLawSpectralModel` and a `ExpDecayTemporalModel`. The important
# thing to note here is how to attach a different `GTI` to each dataset.
# Since we use spectrum datasets here, we will use a `RegionGeom`.
#

# Loading IRFs
irfs = load_cta_irfs(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)

# Reconstructed and true energy axis
energy_axis = MapAxis.from_edges(
    np.logspace(-0.5, 1.0, 10), unit="TeV", name="energy", interp="log"
)
energy_axis_true = MapAxis.from_edges(
    np.logspace(-1.2, 2.0, 31), unit="TeV", name="energy_true", interp="log"
)

geom = RegionGeom.create("galactic;circle(0, 0, 0.11)", axes=[energy_axis])

# Pointing position
pointing = SkyCoord(0.5, 0.5, unit="deg", frame="galactic")


######################################################################
# Note that observations are usually conducted in Wobble mode, in which
# the source is not in the center of the camera. This allows to have a
# symmetrical sky position from which background can be estimated.
#

# Define the source model: A combination of spectral and temporal model

gti_t0 = Time("2020-03-01")
spectral_model = PowerLawSpectralModel(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
temporal_model = ExpDecayTemporalModel(t0="6 h", t_ref=gti_t0.mjd * u.d)

model_simu = SkyModel(
    spectral_model=spectral_model,
    temporal_model=temporal_model,
    name="model-simu",
)

# Look at the model
model_simu.parameters.to_table()


######################################################################
# Now, define the start and observation livetime wrt to the reference
# time, `gti_t0`
#

n_obs = 10

tstart = gti_t0 + [1, 2, 3, 5, 8, 10, 20, 22, 23, 24] * u.h
lvtm = [55, 25, 26, 40, 40, 50, 40, 52, 43, 47] * u.min


######################################################################
# Now perform the simulations
#

datasets = Datasets()

empty = SpectrumDataset.create(
    geom=geom, energy_axis_true=energy_axis_true, name="empty"
)

maker = SpectrumDatasetMaker(selection=["exposure", "background", "edisp"])


for idx in range(n_obs):
    obs = Observation.create(
        pointing=pointing,
        livetime=lvtm[idx],
        tstart=tstart[idx],
        irfs=irfs,
        reference_time=gti_t0,
        obs_id=idx,
        location=observatory_locations["cta_south"],
    )
    empty_i = empty.copy(name=f"dataset-{idx}")
    dataset = maker.run(empty_i, obs)
    dataset.models = model_simu
    dataset.fake()
    datasets.append(dataset)


######################################################################
# The reduced datasets have been successfully simulated. Let’s take a
# quick look into our datasets.
#

datasets.info_table()


######################################################################
# Extract the lightcurve
# ----------------------
#
# This section uses standard light curve estimation tools for a 1D
# extraction. Only a spectral model needs to be defined in this case.
# Since the estimator returns the integrated flux separately for each time
# bin, the temporal model need not be accounted for at this stage. We
# extract the lightcurve in 3 energy binsç
#

# Define the model:
spectral_model = PowerLawSpectralModel(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model_fit = SkyModel(spectral_model=spectral_model, name="model-fit")

# Attach model to all datasets
datasets.models = model_fit

# %%time
lc_maker_1d = LightCurveEstimator(
    energy_edges=[0.3, 0.6, 1.0, 10] * u.TeV,
    source="model-fit",
    selection_optional=["ul"],
)
lc_1d = lc_maker_1d.run(datasets)

ax = lc_1d.plot(marker="o", axis_name="time", sed_type="flux")


######################################################################
# Fitting temporal models
# -----------------------
#
# We have the reconstructed lightcurve at this point. Now we want to fit a
# profile to the obtained light curves, using a joint fitting across the
# different datasets, while simultaneously minimising across the temporal
# model parameters as well. The temporal models can be applied
#
# -  directly on the obtained lightcurve
# -  on the simulated datasets
#


######################################################################
# Fitting the obtained light curve
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The fitting will proceed through a joint fit of the flux points. First,
# we need to obtain a set of `FluxPointDatasets`, one for each time bin
#

# Create the datasets by iterating over the returned lightcurve
datasets = Datasets()

for idx, fp in enumerate(lc_1d.iter_by_axis(axis_name="time")):
    dataset = FluxPointsDataset(data=fp, name=f"time-bin-{idx}")
    datasets.append(dataset)


######################################################################
# We will fit the amplitude, spectral index and the decay time scale. Note
# that `t_ref` should be fixed by default for the
# `ExpDecayTemporalModel`.
#

# Define the model:
spectral_model1 = PowerLawSpectralModel(
    index=2.0, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV"
)
temporal_model1 = ExpDecayTemporalModel(t0="10 h", t_ref=gti_t0.mjd * u.d)

model = SkyModel(
    spectral_model=spectral_model1,
    temporal_model=temporal_model1,
    name="model-test",
)

datasets.models = model

# %%time
# Do a joint fit
fit = Fit()
result = fit.run(datasets=datasets)


######################################################################
# Now let’s plot model and data. We plot only the normalisation of the
# temporal model in relative units for one particular energy range
#

lc_1TeV_10TeV = lc_1d.slice_by_idx({"energy": slice(2, 3)})
ax = lc_1TeV_10TeV.plot(sed_type="norm", axis_name="time")

time_range = lc_1TeV_10TeV.geom.axes["time"].time_bounds
temporal_model1.plot(ax=ax, time_range=time_range, label="Best fit model")

ax.set_yscale("linear")
plt.legend()


######################################################################
# Fit the datasets
# ~~~~~~~~~~~~~~~~
#
# Here, we apply the models directly to the simulated datasets.
#
# For modelling and fitting more complex flares, you should attach the
# relevant model to each group of `datasets`. The parameters of a model
# in a given group of dataset will be tied. For more details on joint
# fitting in Gammapy, see :doc:`/tutorials/analysis-3d/analysis_3d`
#

# Define the model:
spectral_model2 = PowerLawSpectralModel(
    index=2.0, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV"
)
temporal_model2 = ExpDecayTemporalModel(t0="10 h", t_ref=gti_t0.mjd * u.d)

model2 = SkyModel(
    spectral_model=spectral_model2,
    temporal_model=temporal_model2,
    name="model-test2",
)

model2.parameters.to_table()

datasets.models = model2

# %%time
# Do a joint fit
fit = Fit()
result = fit.run(datasets=datasets)

result.parameters.to_table()


######################################################################
# We see that the fitted parameters are consistent between fitting flux
# points and datasets, and match well with the simulated ones
#


######################################################################
# Exercises
# ---------
#
# 1. Re-do the analysis with `MapDataset` instead of `SpectralDataset`
# 2. Model the flare of PKS 2155-304 which you obtained using the
#   :doc:`/tutorials/analysis-time/light_curve_flare` tutorial.
#   Use a combination of a Gaussian and Exponential flare profiles.
# 3. Do a joint fitting of the datasets.
#
