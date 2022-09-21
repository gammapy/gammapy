"""
Spectral analysis with the HLI
==============================

Introduction to 1D analysis using the Gammapy high level interface.

Prerequisites
-------------

-  Understanding the gammapy data workflow, in particular what are DL3
   events and instrument response functions (IRF).

Context
-------

This notebook is an introduction to gammapy analysis using the high
level interface.

Gammapy analysis consists in two main steps.

The first one is data reduction: user selected observations are reduced
to a geometry defined by the user. It can be 1D (spectrum from a given
extraction region) or 3D (with a sky projection and an energy axis). The
resulting reduced data and instrument response functions (IRF) are
called datasets in Gammapy.

The second step consists in setting a physical model on the datasets and
fitting it to obtain relevant physical information.

**Objective: Create a 1D dataset of the Crab using the H.E.S.S. DL3 data
release 1 and perform a simple model fitting of the Crab nebula.**

Proposed approach
-----------------

This notebook uses the high level `~gammapy.analysis.Analysis` class to orchestrate data
reduction and run the data fits. In its current state, `Analysis`
supports the standard analysis cases of joint or stacked 3D and 1D
analyses. It is instantiated with an `~gammapy.analysis.AnalysisConfig` object that
gives access to analysis parameters either directly or via a YAML config
file.

To see what is happening under-the-hood and to get an idea of the
internal API, a second notebook performs the same analysis without using
the `~gammapy.analysis.Analysis` class.

In summary, we have to:

-  Create an `~gammapy.analysis.AnalysisConfig` object and the
   analysis configuration:

   -  Define what observations to use
   -  Define the geometry of the dataset (data and IRFs)
   -  Define the model we want to fit on the dataset.

-  Instantiate a `~gammapy.analysis.Analysis` from this configuration
   and run the different analysis steps

   -  Observation selection
   -  Data reduction
   -  Model fitting
   -  Estimating flux points

"""


######################################################################
# Setup
# -----
# 

# %matplotlib inline
import matplotlib.pyplot as plt

from pathlib import Path
from astropy import units as u
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import Models, SkyModel

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()

######################################################################
# Analysis configuration
# ----------------------
# 
# For configuration of the analysis we use the
# `YAML <https://en.wikipedia.org/wiki/YAML>`__ data format. YAML is a
# machine readable serialisation format, that is also friendly for humans
# to read. In this tutorial we will write the configuration file just
# using Python strings, but of course the file can be created and modified
# with any text editor of your choice.
# 
# Here is what the configuration for our analysis looks like:
# 

yaml_str = """
observations:
    datastore: $GAMMAPY_DATA/hess-dl3-dr1
    obs_cone: {frame: icrs, lon: 83.633 deg, lat: 22.014 deg, radius: 5 deg}

datasets:
    type: 1d
    stack: true
    geom:
        axes:
            energy: {min: 0.5 TeV, max: 30 TeV, nbins: 20}
            energy_true: {min: 0.1 TeV, max: 50 TeV, nbins: 40}
    on_region: {frame: icrs, lon: 83.633 deg, lat: 22.014 deg, radius: 0.11 deg}
    containment_correction: true
    safe_mask:
       methods: ['offset-max']
       parameters: {offset_max: 2.0 deg}
    background:
        method: reflected
fit:
    fit_range: {min: 1 TeV, max: 20 TeV}

flux_points:
    energy: {min: 1 TeV, max: 20 TeV, nbins: 8}
    source: 'crab'
"""

config = AnalysisConfig.from_yaml(yaml_str)
print(config)


######################################################################
# Note that you can save this string into a yaml file and load it as
# follow:
# 

# config = AnalysisConfig.read("config-1d.yaml")
# # the AnalysisConfig gives access to the various parameters used from logging to reduced dataset geometries
# print(config)


######################################################################
# Using data stored into your computer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# Here, we want to use Crab runs from the H.E.S.S. DL3-DR1. We have
# defined the datastore and a cone search of observations pointing with 5
# degrees of the Crab nebula. Parameters can be set directly or as a
# python dict.
# 
# PS: do not forget to setup your environment variable *$GAMMAPY_DATA* to
# your local directory containing the H.E.S.S. DL3-DR1 as described in
# :ref:`quickstart-setup`.
# 


######################################################################
# Setting the exclusion mask
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# In order to properly adjust the background normalisation on regions
# without gamma-ray signal, one needs to define an exclusion mask for the
# background normalisation. For this tutorial, we use the following one
# `$GAMMAPY_DATA/joint-crab/exclusion/exclusion_mask_crab.fits.gz`
# 

config.datasets.background.exclusion = (
    "$GAMMAPY_DATA/joint-crab/exclusion/exclusion_mask_crab.fits.gz"
)


######################################################################
# We’re all set. But before we go on let’s see how to save or import
# `~gammapy.analysis.AnalysisConfig` objects though YAML files.
# 


######################################################################
# Using YAML configuration files for setting/writing the Data Reduction parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# One can export/import the `~gammapy.analysis.AnalysisConfig` to/from a YAML file.
# 

config.write("config.yaml", overwrite=True)

config = AnalysisConfig.read("config.yaml")
print(config)


######################################################################
# Running the first step of the analysis: the Data Reduction
# ----------------------------------------------------------
# 


######################################################################
# Configuration of the analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We first create an `~gammapy.analysis.Analysis` object from our
# configuration.
# 

analysis = Analysis(config)


######################################################################
# Observation selection
# ~~~~~~~~~~~~~~~~~~~~~
# 
# We can directly select and load the observations from disk using
# `~gammapy.analysis.Analysis.get_observations()`:
# 

analysis.get_observations()


######################################################################
# The observations are now available on the `Analysis` object. The
# selection corresponds to the following ids:
# 

analysis.observations.ids


######################################################################
# To see how to explore observations, please refer to the following
# notebook: `CTA with Gammapy <../data/cta.ipynb>`__ or `HESS with
# Gammapy <../data/hess.ipynb>`__
# 


######################################################################
# Running the Data Reduction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now we proceed to the data reduction. In the config file we have chosen
# a WCS map geometry, energy axis and decided to stack the maps. We can
# run the reduction using `.get_datasets()`:
# 

# %%time
analysis.get_datasets()


######################################################################
# Results exploration
# ~~~~~~~~~~~~~~~~~~~
# 
# As we have chosen to stack the data, one can print what contains the
# unique entry of the datasets:
# 

print(analysis.datasets[0])


######################################################################
# As you can see the dataset uses WStat with the background computed with
# the Reflected Background method during the data reduction, but no source
# model has been set yet.
# 
# The counts, exposure and background, etc are directly available on the
# dataset and can be printed:
# 

info_table = analysis.datasets.info_table()
info_table

print(
    f"Tobs={info_table['livetime'].to('h')[0]:.1f} Excess={info_table['excess'].value[0]:.1f} \
Significance={info_table['sqrt_ts'][0]:.2f}"
)


######################################################################
# Save dataset to disk
# ~~~~~~~~~~~~~~~~~~~~
# 
# It is common to run the preparation step independent of the likelihood
# fit, because often the preparation of counts, collection are and energy
# dispersion is slow if you have a lot of data. We first create a folder:
# 

path = Path("hli_spectrum_analysis")
path.mkdir(exist_ok=True)


######################################################################
# And then write the stacked dataset to disk by calling the dedicated
# `write()` method:
# 

filename = path / "crab-stacked-dataset.fits.gz"
analysis.datasets.write(filename, overwrite=True)


######################################################################
# Model fitting
# -------------
# 


######################################################################
# Creation of the model
# ~~~~~~~~~~~~~~~~~~~~~
# 
# First, let’s create a model to be adjusted. As we are performing a 1D
# Analysis, only a spectral model is needed within the `SkyModel`
# object. Here is a pre-defined YAML configuration file created for this
# 1D analysis:
# 

model_str = """
components:
- name: crab
  type: SkyModel
  spectral:
    type: PowerLawSpectralModel
    parameters:
      - name: index
        frozen: false
        scale: 1.0
        unit: ''
        value: 2.6
      - name: amplitude
        frozen: false
        scale: 1.0
        unit: cm-2 s-1 TeV-1
        value: 5.0e-11
      - name: reference
        frozen: true
        scale: 1.0
        unit: TeV
        value: 1.0
"""
model_1d = Models.from_yaml(model_str)
print(model_1d)


######################################################################
# Or from a yaml file, e.g. 
# 

# model_1d = Models.read("model-1d.yaml")
# print(model_1d)


######################################################################
# Now we set the model on the analysis object:
# 

analysis.set_models(model_1d)


######################################################################
# Setting fitting parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# `Analysis` can perform a few modeling and fitting tasks besides data
# reduction. Parameters have then to be passed to the configuration
# object.
# 


######################################################################
# Running the fit
# ~~~~~~~~~~~~~~~
# 

# %%time
analysis.run_fit()


######################################################################
# Exploration of the fit results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

print(analysis.fit_result)

model_1d.to_parameters_table()


######################################################################
# To check the fit is correct, we compute the excess spectrum with the
# predicted counts.
# 

ax_spectrum, ax_residuals = analysis.datasets[0].plot_fit()
ax_spectrum.set_ylim(0.1, 200)
ax_spectrum.set_xlim(0.2, 60)
analysis.datasets[0].plot_masks(ax=ax_spectrum);


######################################################################
# Serialisation of the fit result
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# This is how we can write the model back to file again:
# 

filename = path / "model-best-fit.yaml"
analysis.models.write(filename, overwrite=True)

with filename.open("r") as f:
    print(f.read())


######################################################################
# Creation of the Flux points
# ---------------------------
# 


######################################################################
# Running the estimation
# ~~~~~~~~~~~~~~~~~~~~~~
# 

analysis.get_flux_points()

crab_fp = analysis.flux_points.data
crab_fp.to_table(sed_type="dnde", formatted=True)


######################################################################
# Let’s plot the flux points with their likelihood profile
# 

plt.figure(figsize=(10, 8))
ax_sed = crab_fp.plot(sed_type="e2dnde", color="darkorange")
ax_sed.set_ylim(1.0e-12, 2.0e-10)
ax_sed.set_xlim(0.5, 40)
crab_fp.plot_ts_profiles(ax=ax_sed, sed_type="e2dnde");


######################################################################
# Serialisation of the results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The flux points can be exported to a fits table following the format
# defined
# `here <https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/flux_points/index.html>`__
# 

filename = path / "flux-points.fits"
analysis.flux_points.write(filename, overwrite=True)


######################################################################
# Plotting the final results of the 1D Analysis
# ---------------------------------------------
# 


######################################################################
# We can plot of the spectral fit with its error band overlaid with the
# flux points:
# 

ax_sed, ax_residuals = analysis.flux_points.plot_fit()
ax_sed.set_ylim(1.0e-12, 1.0e-9)
ax_sed.set_xlim(0.5, 40)


######################################################################
# What’s next?
# ------------
# 
# You can look at the same analysis without the high level interface in
# `spectral analysis <../analysis/1D/spectral_analysis.ipynb>`__.
# 
# As we can store the best model fit, you can overlaid the fit results of
# both methods on an unique plot.
# 

