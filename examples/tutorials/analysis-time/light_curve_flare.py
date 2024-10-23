"""
Light curves for flares
=======================

Compute the light curve of a PKS 2155-304 flare on 10 minutes time intervals.

Prerequisites
-------------

-  Understanding of how the light curve estimator works, please refer to
   the :doc:`light curve notebook </tutorials/analysis-time/light_curve>`.

Context
-------

Frequently, especially when studying flares of bright sources, it is
necessary to explore the time behaviour of a source on short time
scales, in particular on time scales shorter than observing runs.

A typical example is given by the flare of PKS 2155-304 during the night
from July 29 to 30 2006. See the `following
article <https://ui.adsabs.harvard.edu/abs/2009A%26A...502..749A/abstract>`__.

**Objective: Compute the light curve of a PKS 2155-304 flare on 5
minutes time intervals, i.e. smaller than the duration of individual
observations.**

Proposed approach
-----------------

We have seen in the general presentation of the light curve estimator,
see the :doc:`light curve notebook </tutorials/analysis-time/light_curve>`, Gammapy produces
datasets in a given time interval, by default that of the parent
observation. To be able to produce datasets on smaller time steps, it is
necessary to split the observations into the required time intervals.

This is easily performed with the `~gammapy.data.Observations.select_time` method of
`~gammapy.data.Observations`. If you pass it a list of time intervals
it will produce a list of time filtered observations in a new
`~gammapy.data.Observations` object. Data reduction can then be
performed and will result in datasets defined on the required time
intervals and light curve estimation can proceed directly.

In summary, we have to:

-  Select relevant `~gammapy.data.Observations` from the
   `~gammapy.data.DataStore`
-  Apply the time selection in our predefined time intervals to obtain a
   new `~gammapy.data.Observations`
-  Perform the data reduction (in 1D or 3D)
-  Define the source model
-  Extract the light curve from the reduced dataset

Here, we will use the PKS 2155-304 observations from the
`H.E.S.S. first public test data release <https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/>`__.
We will use time intervals of 5 minutes
duration. The tutorial is implemented with the intermediate level API.

Setup
-----

As usual, we’ll start with some general imports…

"""

import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from regions import CircleSkyRegion

# %matplotlib inline
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

from gammapy.data import DataStore
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.estimators import LightCurveEstimator
from gammapy.estimators.utils import get_rebinned_axis
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.modeling import Fit

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# Select the data
# ---------------
#
# We first set the datastore.
#

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")


######################################################################
# Now we select observations within 2 degrees of PKS 2155-304.
#

target_position = SkyCoord(329.71693826 * u.deg, -30.2255890 * u.deg, frame="icrs")
selection = dict(
    type="sky_circle",
    frame="icrs",
    lon=target_position.ra,
    lat=target_position.dec,
    radius=2 * u.deg,
)
obs_ids = data_store.obs_table.select_observations(selection)["OBS_ID"]
observations = data_store.get_observations(obs_ids)
print(f"Number of selected observations : {len(observations)}")


######################################################################
# Define time intervals
# ---------------------
#
# We create the list of time intervals, each of duration 10 minutes. Each time interval is an
# `astropy.time.Time` object, containing a start and stop time.

t0 = Time("2006-07-29T20:30")
duration = 10 * u.min
n_time_bins = 35
times = t0 + np.arange(n_time_bins) * duration
time_intervals = [Time([tstart, tstop]) for tstart, tstop in zip(times[:-1], times[1:])]
print(time_intervals[0].mjd)


######################################################################
# Filter the observations list in time intervals
# ----------------------------------------------
#
# Here we apply the list of time intervals to the observations with
# `~gammapy.data.Observations.select_time()`.
#
# This will return a new list of Observations filtered by ``time_intervals``.
# For each time interval, a new observation is created that converts the
# intersection of the GTIs and time interval.
#

short_observations = observations.select_time(time_intervals)
# check that observations have been filtered
print(f"Number of observations after time filtering: {len(short_observations)}\n")
print(short_observations[1].gti)


######################################################################
# As we can see, we have now observations of duration equal to the chosen
# time step.
#
# Now data reduction and light curve extraction can proceed exactly as
# before.
#


######################################################################
# Building 1D datasets from the new observations
# ----------------------------------------------
#
# Here we will perform the data reduction in 1D with reflected regions.
#
# *Beware, with small time intervals the background normalization with OFF
# regions might become problematic.*
#


######################################################################
# Defining the geometry
# ~~~~~~~~~~~~~~~~~~~~~
#
# We define the energy axes. As usual, the true energy axis has to cover a
# wider range to ensure a good coverage of the measured energy range
# chosen.
#
# We need to define the ON extraction region. Its size follows typical
# spectral extraction regions for H.E.S.S. analyses.
#

# Target definition
energy_axis = MapAxis.from_energy_bounds("0.4 TeV", "20 TeV", nbin=10)
energy_axis_true = MapAxis.from_energy_bounds(
    "0.1 TeV", "40 TeV", nbin=20, name="energy_true"
)

on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)

geom = RegionGeom.create(region=on_region, axes=[energy_axis])


######################################################################
# Creation of the data reduction makers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We now create the dataset and background makers for the selected
# geometry.
#

dataset_maker = SpectrumDatasetMaker(
    containment_correction=True, selection=["counts", "exposure", "edisp"]
)
bkg_maker = ReflectedRegionsBackgroundMaker()
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)


######################################################################
# Creation of the datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now we perform the actual data reduction in the ``time_intervals``.
#

# %%time
datasets = Datasets()

dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)

for obs in short_observations:
    dataset = dataset_maker.run(dataset_empty.copy(), obs)

    dataset_on_off = bkg_maker.run(dataset, obs)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, obs)
    datasets.append(dataset_on_off)


######################################################################
# Define underlying model
# -----------------------
#
# Since we use forward folding to obtain the flux points in each bin, exact values will depend on the underlying model. In this example, we use a power law as used in the
# `reference
# paper <https://ui.adsabs.harvard.edu/abs/2009A%26A...502..749A/abstract>`__.
#
# As we have are only using spectral datasets, we do not need any spatial models.
#
# **Note** : All time bins must have the same spectral model. To see how to investigate spectral variability,
# see :doc:`time resolved spectroscopy notebook </tutorials/analysis-time/time_resolved_spectroscopy>`.

spectral_model = PowerLawSpectralModel(amplitude=1e-10 * u.Unit("1 / (cm2 s TeV)"))
sky_model = SkyModel(spatial_model=None, spectral_model=spectral_model, name="pks2155")

######################################################################
# Assign to model to all datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We assign each dataset its spectral model
#

datasets.models = sky_model


# %%time
fit = Fit()
result = fit.run(datasets)
print(result.models.to_parameters_table())

######################################################################
# Extract the light curve
# -----------------------
#
# We first create the `~gammapy.estimators.LightCurveEstimator` for the
# list of datasets we just produced. We give the estimator the name of the
# source component to be fitted. We can directly compute the light curve in multiple energy
# bins by supplying a list of `energy_edges`.
#
# By default the likelihood scan is computed from 0.2 to 5.0.
# Here, we increase the max value to 10.0, because we are
# dealing with a large flare.

lc_maker_1d = LightCurveEstimator(
    energy_edges=[0.7, 1, 20] * u.TeV,
    source="pks2155",
    time_intervals=time_intervals,
    selection_optional="all",
    n_jobs=4,
)
lc_maker_1d.norm.scan_max = 10


######################################################################
# We can now perform the light curve extraction itself. To compare with
# the `reference
# paper <https://ui.adsabs.harvard.edu/abs/2009A%26A...502..749A/abstract>`__,
# we select the 0.7-20 TeV range.
#

# %%time
lc_1d = lc_maker_1d.run(datasets)


######################################################################
# Finally we plot the result for the 1D lightcurve:
#
plt.figure(figsize=(8, 6))
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
lc_1d.plot(marker="o", axis_name="time", sed_type="flux")
plt.show()


######################################################################
# Light curves once obtained can be rebinned using the likelihood profiles.
# Here, we rebin 3 adjacent bins together, to get 30 minute bins.
#
# We will first slice `lc_1d` to obtain the lightcurve in the first energy bin
#

slices = {"energy": slice(0, 1)}
sliced_lc = lc_1d.slice_by_idx(slices)
print(sliced_lc)

axis_new = get_rebinned_axis(
    sliced_lc, method="fixed-bins", group_size=3, axis_name="time"
)
print(axis_new)

lc_new = sliced_lc.resample_axis(axis_new)
plt.figure(figsize=(8, 6))
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
ax = sliced_lc.plot(label="original")
lc_new.plot(ax=ax, label="rebinned")
plt.legend()
plt.show()

#####################################################################
# We can use the sliced lightcurve to understand the variability,
# as shown in the doc:`/tutorials/analysis-time/variability_estimation` tutorial.
