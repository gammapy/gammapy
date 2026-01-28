"""
Time resolved spectroscopy estimator
====================================

Perform spectral fits of a blazar in different time bins to investigate
spectral changes during flares.

Context
-------

The `~gammapy.estimators.LightCurveEstimator` in Gammapy (see
:doc:`light curve notebook </tutorials/analysis-time/light_curve>`,
and
:doc:`light curve for flares notebook </tutorials/analysis-time/light_curve_flare>`.)
fits the amplitude in each time/energy bin, keeping the spectral shape
frozen. However, in the analysis of flaring sources, it is often
interesting to study not only how the flux changes with time but how the
spectral shape varies with time.

Proposed approach
-----------------

The main idea behind doing a time resolved spectroscopy is to

-  Select relevant `~gammapy.data.Observations` from the
   `~gammapy.data.DataStore`
-  Define time intervals in which to fit the spectral model
-  Apply the above time selections on the data to obtain new
   `~gammapy.data.Observations`
-  Perform standard data reduction on the above data
-  Define a source model
-  Fit the reduced data in each time bin with the source model
-  Extract relevant information in a table

Here, we will use the PKS 2155-304 observations from the
`H.E.S.S. first public test data release <https://hess-experiment.eu/releases/>`__.

We use time intervals of 15 minutes duration to explore spectral
variability.

Setup
-----

As usual, we’ll start with some general imports…

"""

import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import QTable
from astropy.time import Time
from regions import CircleSkyRegion

# %matplotlib inline
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

from gammapy.data import DataStore
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import MapAxis, RegionGeom, TimeMapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
)

log = logging.getLogger(__name__)


######################################################################
# Data selection
# ~~~~~~~~~~~~~~
#
# We select all runs pointing within 2 degrees of PKS 2155-304.
#

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
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
# The flaring observations were taken during July 2006. We define
# 15-minute time intervals as lists of `~astropy.time.Time` start and stop
# objects, and apply the intervals to the observations by using
# `~gammapy.data.Observations.select_time`
#

t0 = Time("2006-07-29T20:30")
duration = 15 * u.min
n_time_bins = 25
times = t0 + np.arange(n_time_bins) * duration

time_intervals = [Time([tstart, tstop]) for tstart, tstop in zip(times[:-1], times[1:])]
print(time_intervals[-1].mjd)
short_observations = observations.select_time(time_intervals)

# check that observations have been filtered
print(f"Number of observations after time filtering: {len(short_observations)}\n")
print(short_observations[1].gti)


######################################################################
# Data reduction
# --------------
#
# In this example, we perform a 1D analysis with a reflected regions
# background estimation. For details, see the
# :doc:`/tutorials/analysis-1d/spectral_analysis` tutorial.
#

energy_axis = MapAxis.from_energy_bounds("0.4 TeV", "20 TeV", nbin=10)
energy_axis_true = MapAxis.from_energy_bounds(
    "0.1 TeV", "40 TeV", nbin=20, name="energy_true"
)

on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)

geom = RegionGeom.create(region=on_region, axes=[energy_axis])

dataset_maker = SpectrumDatasetMaker(
    containment_correction=True, selection=["counts", "exposure", "edisp"]
)
bkg_maker = ReflectedRegionsBackgroundMaker()
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

datasets = Datasets()

dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)

for obs in short_observations:
    dataset = dataset_maker.run(dataset_empty.copy(), obs)

    dataset_on_off = bkg_maker.run(dataset, obs)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, obs)
    datasets.append(dataset_on_off)


######################################################################
# This gives us list of `~gammapy.datasets.SpectrumDatasetOnOff` which can now be
# modelled.
#

print(datasets)


######################################################################
# Modeling
# --------
#
# We will first fit a simple power law model in each time bin. Note that
# since we are using an on-off analysis here, no background model is
# required. If you are doing a 3D FoV analysis, you will need to model the
# background appropriately as well.
#
# The index and amplitude of the spectral model is kept free. You can
# configure the quantities you want to freeze.
#

spectral_model = PowerLawSpectralModel(
    index=3.0, amplitude=2e-11 * u.Unit("1 / (cm2 s TeV)"), reference=1 * u.TeV
)
spectral_model.parameters["index"].frozen = False


sky_model = SkyModel(spatial_model=None, spectral_model=spectral_model, name="pks2155")
print(sky_model)


######################################################################
# Time resolved spectroscopy algorithm
# ------------------------------------
#
# The following function is the crux of this tutorial. The ``sky_model``
# is fit in each bin and a list of ``fit_results`` stores the fit
# information in each bin.
#
# If time bins are present without any available observations, those bins
# are discarded and a new list of valid time intervals and fit results are
# created.
#


def time_resolved_spectroscopy(datasets, model, time_intervals):
    fit = Fit()
    valid_intervals = []
    fit_results = []
    index = 0
    for t_min, t_max in time_intervals:
        datasets_to_fit = datasets.select_time(time_min=t_min, time_max=t_max)

        if len(datasets_to_fit) == 0:
            log.info(
                f"No Dataset for the time interval {t_min} to {t_max}. Skipping interval."
            )
            continue

        model_in_bin = model.copy(name="Model_bin_" + str(index))
        datasets_to_fit.models = model_in_bin
        result = fit.run(datasets_to_fit)
        fit_results.append(result)
        valid_intervals.append([t_min, t_max])
        index += 1

    return valid_intervals, fit_results


######################################################################
# We now apply it to our data
#

valid_times, results = time_resolved_spectroscopy(datasets, sky_model, time_intervals)


######################################################################
# To view the results of the fit,
#

print(results[0])


######################################################################
# Or, to access the fitted models,
#

print(results[0].models)


######################################################################
# To better visualise the data, we can create a table by extracting some
# relevant information. In the following, we extract the time intervals,
# information on the fit convergence and the free parameters. You can
# extract more information if required, eg, the ``total_stat`` in each
# bin, etc.
#


def create_table(time_intervals, fit_result):
    t = QTable()

    t["tstart"] = np.array(time_intervals).T[0]
    t["tstop"] = np.array(time_intervals).T[1]
    t["convergence"] = [result.success for result in fit_result]
    for par in fit_result[0].models.parameters.free_parameters:
        t[par.name] = [
            result.models.parameters[par.name].value * par.unit for result in fit_result
        ]
        t[par.name + "_err"] = [
            result.models.parameters[par.name].error * par.unit for result in fit_result
        ]

    return t


table = create_table(valid_times, results)
print(table)


######################################################################
# Visualising the results
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can plot the spectral index and the amplitude as a function of time.
# For convenience, we will convert the times into a `~gammapy.maps.TimeMapAxis`.
#

time_axis = TimeMapAxis.from_time_edges(
    time_min=table["tstart"], time_max=table["tstop"]
)

fix, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].errorbar(
    x=time_axis.as_plot_center, y=table["index"], yerr=table["index_err"], fmt="o"
)
axes[1].errorbar(
    x=time_axis.as_plot_center,
    y=table["amplitude"],
    yerr=table["amplitude_err"],
    fmt="o",
)

axes[0].set_ylabel("index")
axes[1].set_ylabel("amplitude")
axes[1].set_xlabel("time")
plt.show()


######################################################################
# To get the integrated flux, we can access the model stored in the fit
# result object, eg
#

integral_flux = (
    results[0]
    .models[0]
    .spectral_model.integral_error(energy_min=1 * u.TeV, energy_max=10 * u.TeV)
)
print("Integral flux in the first bin:", integral_flux)


######################################################################
# To plot hysteresis curves, ie the spectral index as a function of
# amplitude
#

plt.errorbar(
    table["amplitude"],
    table["index"],
    xerr=table["amplitude_err"],
    yerr=table["index_err"],
    linestyle=":",
    linewidth=0.5,
)
plt.scatter(table["amplitude"], table["index"], c=time_axis.center.value)
plt.xlabel("amplitude")
plt.ylabel("index")
plt.colorbar().set_label("time")
plt.show()


######################################################################
# Exercises
# ---------
#
# 1. Quantify the variability in the spectral index
# 2. Rerun the algorithm using a different spectral shape, such as a
#    broken power law.
# 3. Compare the significance of the new model with the simple power law.
#    Take note of any fit non-convergence in the bins.
#
