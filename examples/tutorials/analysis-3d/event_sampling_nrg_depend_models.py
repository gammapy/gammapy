"""
Sample a source with energy-dependent temporal evolution
==============
This notebook shows how to sample events of sources whose model evolves in energy and time.

Prerequisites
-------------

To understand how to generate a model and a MapDataset and how to fit the data, please refer to the `~gammapy.modeling.models.SkyModel` and
:doc:`/tutorials/analysis-3d/simulate_3d` tutorial. To know how to sample events for standards sources, we suggest to visit the event sampler :doc:`/tutorials/analysis-3d/event_sampling` tutorial.

Objective
-------------

Describe the process of sampling events of a source having an energy-dependent temporal model, and obtain an output event-list.

Proposed approach
-------------

Here we will show how to create an energy dependent temporal model; then we also create an observation and define a Dataset object. Finally we describe how to sample events from the given model.

We will work with the following functions and classes:

-  `~gammapy.data.Observations`
-  `~gammapy.datasets.Dataset`
-  `~gammapy.modeling.models.SkyModel`
-  `~gammapy.datasets.MapDatasetEventSampler`
-  `~gammapy.data.EventList`
-  `~gammapy.maps.RegionNDMap`
"""

######################################################################
# Setup
# -----
#
# As usual, let’s start with some general imports…
#

from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from regions import CircleSkyRegion, PointSkyRegion
import matplotlib.pyplot as plt
from gammapy.data import Observation, observatory_locations
from gammapy.datasets import MapDataset, MapDatasetEventSampler
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker
from gammapy.maps import Map, MapAxis, RegionNDMap, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ConstantSpectralModel,
    FoVBackgroundModel,
    LightCurveTemplateTemporalModel,
    Models,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# Create the energy-dependent model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The source to simulate has a spectrum that varies as a function of
# the time.
# We define its spatial morphology as `point-like`: we note that this
# is a mandatory condition to simulate energy-dependent sources! Other,
# extended morphologies will raise an error!
# In the following example, the source model will vary in 5 time bins,
# and in each bin the source spectrum is represented as a powerlaw with
# the amplitude and index shown in the following box. The spectral evolution
# is also shown in the plot:
#

ampl_model = (
    np.array([2e-10, 8e-11, 5e-11, 3e-11, 1e-11]) * u.cm**-2 * u.TeV**-1 * u.s**-1
)  # amplitude
index_model = np.array([2.2, 2.0, 1.8, 1.6, 1.4])  # index

for i in np.arange(len(ampl_model)):
    spec = PowerLawSpectralModel(
        index=index_model[i], amplitude=ampl_model[i], reference="1 TeV"
    )
    spec.plot([0.2, 100] * u.TeV, label=f"Time bin {i+1}")
plt.legend()
plt.show()

######################################################################
# Let's now create the temporal model (if you already have this model,
# please go directly to the `Read the energy-dependent model` section),
# that will be defined as a `LightCurveTemplateTemporalModel`. The latter
# can take in input a `RegionNDMap` where the temporal and energy source
# information are stored. To create the map, we firstly need to define the
# time axis with `MapAxis`: here we consider 5 time bins of 720 s (i.e. 1 hr
# in total).
# As a second step, we create an energy axis with 10 bins where the
# powerlaw spectral models will be evaluated.
#

# source position
position = SkyCoord("100 deg", "30 deg", frame="icrs")

# time axis
time_axis = MapAxis.from_bounds(0 * u.s, 3600 * u.s, nbin=5, name="time", interp="lin")

# energy axis
nbin = 10
energy_axis = MapAxis.from_energy_bounds(
    energy_min=0.2 * u.TeV, energy_max=100 * u.TeV, nbin=nbin, name="energy"
)


######################################################################
# Now let's create the `RegionNDMap` and fill it with the expect
# spectral values:
#

# make an array with the time and energy shape
data = (
    np.ones((time_axis.nbin, energy_axis.nbin)) * u.cm**-2 * u.s**-1 * u.TeV**-1
)

# create the RegionNDMap
m = RegionNDMap.create(
    region=PointSkyRegion(center=position),
    axes=[energy_axis, time_axis],
)

# evaluate the spectrum and fill the RegionNDMap
for i in np.arange(time_axis.nbin):
    spec = PowerLawSpectralModel(
        index=index_model[i], amplitude=ampl_model[i], reference="1 TeV"
    )
    data[i, :], _ = spec.evaluate_error(energy_axis.center)

m.data = np.array(data)

######################################################################
# Now, we define the `LightCurveTemplateTemporalModel` passing it the
# map we created above. We set the reference time of the model, and
# this is crucial otherwise the model would be wrongly evaluated.
# We show also how to write the model on disk, noting that we explicitely
# set the `format` to `map`.

t_ref = Time(51544.00074287037, format="mjd", scale="tt")
filename = "./temporal_model_map.fits"
temp = LightCurveTemplateTemporalModel(m, t_ref=t_ref, filename=filename)
temp.write(filename, format="map", overwrite=True)


######################################################################
# Read the energy-dependent model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We read the map written on disc again with `LightCurveTemplateTemporalModel.read`.
# When the model is from a map, the arguments `format="map"` is mandatory.
# The map is `fits` file, with 3 extensions:
#
# 1) `SKYMAP`: a table with a `CHANNEL` and `DATA` column; the number of rows is given
# by the product of the energy and time bins. The `DATA` represent the values of the model
# at each energy;
#
# 2) `SKYMAP_BANDS`: a table with `CHANNEL`, `ENERGY`, `E_MIN`, `E_MAX`, `TIME`,
# `TIME_MIN` and `TIME_MAX`. `ENERGY` is the mean of `E_MIN` and `E_MAX`, as well as
# `TIME` is the mean of `TIME_MIN` and `TIME_MAX`; this extension should contain the
# reference time in the header, through the keywords `MJDREFI` and `MJDREFF`.
#
# 3) `SKYMAP_REGION`: it gives information on the spatial morphology, i.e. `SHAPE`
# (only `point` is accepted), `X` and `Y` (source position), `R` (the radius if
# extended; not used in our case) and `ROTANG` (the angular rotation of the spatial
# model, if extended; not used in our case).
#

temporal_model = LightCurveTemplateTemporalModel.read(filename, format="map")

######################################################################
# We note that an interpolation scheme is also provided when loading
# a map: for a an energy-dependent temporal model, the `method` and
# `values_scale` arguments by default are set to `linear` and `log`.
# We warn the reader to carefully check the interpolation method used
# for the time axis while creating the template model, as different
# methods provide different results.
# By default, we assume `linear` interpolation for the time, `log`
# for the energies and values.
# Users can modify the `method` and `values_scale` arguments but we
# warn that this should be done only when the users knows the consequences
# of the changes. Here, we show how to set them explicitely:
#

temporal_model.method = "linear"  # default
temporal_model.values_scale = "log"  # default

######################################################################
# We can have a visual inspection of the temporal model at different energies:
#

t = Time(temporal_model.reference_time + [-100, 3600] * u.s)

temporal_model.plot(time_range=(t[0], t[-1]), energy=[0.1, 0.5, 1, 5] * u.TeV)
plt.semilogy()
plt.show()

######################################################################
# Now that the temporal model is complete, we create the whole source
# `SkyModel` plus the background model. Note that the source `spectral_model`
# is a `ConstantSpectralModel`: this is necessary and mandatory, as the real
# source spectrum is actually passed through the map.
#

spatial_model = PointSpatialModel.from_position(position)
spectral_model = ConstantSpectralModel(const="1 cm-2 s-1 TeV-1")

model = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    temporal_model=temporal_model,
    name="test-source",
)

bkg_model = FoVBackgroundModel(dataset_name="my-dataset")

models = [model, bkg_model]


######################################################################
# Define an observation and make a dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we define an observation of 1 hr with CTA in the
# alpha-configuration for the south array, and we also create a dataset
# to be passed to the event sampler. The full `SkyModel` created above
# is passed to the dataset.
#

path = Path("$GAMMAPY_DATA/cta-caldb")
irf_filename = "Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz"

pointing = SkyCoord(100.0, 30.0, frame="icrs", unit="deg")
livetime = 1 * u.hr

irfs = load_irf_dict_from_file(path / irf_filename)
location = observatory_locations["cta_south"]

observation = Observation.create(
    obs_id=1001,
    pointing=pointing,
    livetime=livetime,
    irfs=irfs,
    location=location,
)

######################################################################

energy_axis = MapAxis.from_energy_bounds("0.2 TeV", "100 TeV", nbin=5, per_decade=True)
energy_axis_true = MapAxis.from_energy_bounds(
    "0.05 TeV", "150 TeV", nbin=10, per_decade=True, name="energy_true"
)
migra_axis = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")

geom = WcsGeom.create(
    skydir=pointing,
    width=(2, 2),
    binsz=0.02,
    frame="icrs",
    axes=[energy_axis],
)

######################################################################

empty = MapDataset.create(
    geom,
    energy_axis_true=energy_axis_true,
    migra_axis=migra_axis,
    name="my-dataset",
)
maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
dataset = maker.run(empty, observation)

dataset.models = models

print(dataset.models)


######################################################################
# Let's simulate the model
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Initialize and run the `MapDatasetEventSampler` class. We also define
# the `oversample_energy_factor` arguments: this should be carefully
# considered by the user, as a higher `oversample_energy_factor` gives
# a more precise source flux estimate, at the expense of computational
# time. Here we adopt an `oversample_energy_factor` of 10:
#

sampler = MapDatasetEventSampler(random_state=0, oversample_energy_factor=10)
events = sampler.run(dataset, observation)

######################################################################
# Let's inspect the simulated events in the source region:
#

src_position = SkyCoord(100.0, 30.0, frame="icrs", unit="deg")

on_region_radius = Angle("0.15 deg")
on_region = CircleSkyRegion(center=src_position, radius=on_region_radius)

src_events = events.select_region(on_region)

src_events.peek()
plt.show()


######################################################################
# Fit the simulated data
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The energy-dependent temporal model that we used for the simulation
# cannot be used to fit the data, because such a functionality is not
# yet implemented in Gammapy. Therefore, we need to adopt the approach
# of fitting events in a given time range, using a model defined as a
# powerlaw spectral model, with no temporal information and point-like
# morphology. We fix the source coordinates during the fit for simplicity:
#

spec = PowerLawSpectralModel(
    index=2, amplitude=5e-11 * u.cm**-2 * u.s**-1 * u.TeV**-1, reference="1 TeV"
)

spatial_model = PointSpatialModel(lon_0="100 deg", lat_0="30 deg", frame="icrs")

model = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spec,
    name="fit_source",
)

model.spatial_model.parameters[0].frozen = True
model.spatial_model.parameters[1].frozen = True

bkg_model = FoVBackgroundModel(dataset_name="my-dataset")

model_fit = [model, bkg_model]


######################################################################
# Let's create a new observation, with the time bin starting
# at 360s and with a duration of 100s, and a new dataset for
# the fit:
#

livetime = 100 * u.s

tstart = t_ref + (360 * u.s).to("d")
observation = Observation.create(
    obs_id=1001,
    pointing=pointing,
    livetime=livetime,
    irfs=irfs,
    location=location,
    tstart=tstart,
)

dataset_fit = maker.run(empty, observation)

######################################################################
# We select only the events in the time bin of reference, and we
# fill the dataset with them.
#

filt_evt = events.select_time([observation.gti.time_start, observation.gti.time_stop])
counts = Map.from_geom(geom)
counts.fill_events(filt_evt)
dataset_fit.counts = counts

dataset_fit.models = model_fit
print(dataset_fit.models)


######################################################################
# Let's fit the dataset
#

fit = Fit()
result = fit.run(dataset_fit)
print(result)

result.parameters.to_table()

######################################################################
# Please, note that the fitted parameters reflect the time averaged spectra of
# a time varying source. The flux/spectra is not constant within the fitted
# time bin, and depends upon the interpolation scheme of the model within the
# supplied bins
#

######################################################################
# Exercises
# ---------
#
# -  Try to create a temporal model with a more complex energy-dependent
#    evolution;
# -  Read your temporal model in Gammapy and simulate it;
#
