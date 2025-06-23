"""
VERITAS with gammapy
====================

Explore VERITAS point-like DL3 files, including event lists and IRFs and
calculate Li & Ma significance, spectra, and fluxes.

"""


######################################################################
# Introduction
# ------------
# 
# `VERITAS <https://veritas.sao.arizona.edu/>`__ (Very Energetic Radiation
# Imaging Telescope Array System) is a ground-based gamma-ray instrument
# operating at the Fred Lawrence Whipple Observatory (FLWO) in southern
# Arizona, USA. It is an array of four 12m optical reflectors for
# gamma-ray astronomy in the ~ 100 GeV to > 30 TeV energy range.
# 
# VERITAS data are private and lower level analysis is done using either
# the
# `Eventdisplay <https://github.com/VERITAS-Observatory/EventDisplay_v4>`__
# or `VEGAS (internal access
# only) <https://github.com/VERITAS-Observatory/VEGAS>`__ analysis
# packages to produce DL3 files (using
# `V2DL3 <https://github.com/VERITAS-Observatory/V2DL3>`__), which can be
# used in Gammapy to produce high-level analysis products. A small sub-set
# of archival Crab nebula data has been publically released to accompany
# this tutorial, which provide an introduction to VERITAS data analysis
# using gammapy for VERITAS members and external users alike.
# 
# This notebook is only intended for use with these publically released
# Crab nebula files and the use of other sources or datasets may require
# modifications to this notebook.
# 

import numpy as np
from matplotlib import pyplot as plt

import astropy.units as u

import logging
from gammapy.maps import MapAxis, WcsGeom

from regions import CircleSkyRegion
import matplotlib.pyplot as plt
from gammapy.data import DataStore
from gammapy.modeling.models import SkyModel, LogParabolaSpectralModel
from gammapy.modeling import Fit
from gammapy.datasets import Datasets, SpectrumDataset, FluxPointsDataset
from gammapy.estimators import FluxPointsEstimator, LightCurveEstimator
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
    ReflectedRegionsFinder,
)
from astropy.table import Table
from astropy.coordinates import Angle, SkyCoord
from gammapy.maps import MapAxis, RegionGeom
from gammapy.visualization import plot_spectrum_datasets_off_regions
from astropy.time import Time


######################################################################
# Load in files
# -------------
# 
# First, we select and load VERITAS observations of the Crab Nebula. These
# files are processed with **EventDisplay**, but VEGAS analysis should be
# identical apart from the integration region size, which is specified in
# the relevant section.
# 

data_store = DataStore.from_dir("$GAMMAPY_DATA/veritas/crab-point-like-ED")
data_store.info()


######################################################################
# We filter our data by only taking observations within
# :math:`5 \deg` of the Crab Nebula. See 
# 
# 

target_position = SkyCoord(83.6333,22.0145,unit='deg')

selection = dict(
    type="sky_circle",
    frame="icrs",
    lon=f"{target_position.ra.value} deg",
    lat=f"{target_position.dec.value} deg",
    radius="5 deg",
)
obs_table = data_store.obs_table.select_observations(selection)
obs_ids = obs_table["OBS_ID"]
observations = data_store.get_observations(obs_id=obs_ids,required_irf="point-like")

obs_ids = observations.ids


######################################################################
# Part I: Data Exploration
# ========================
# 


######################################################################
# Look at the information contained in the DL3 file for a single observation
# --------------------------------------------------------------------------
# 


######################################################################
# Peek at the IRFs included : point source files will contain effective
# areas, energy dispersion matrices, and events for both VEGAS and
# Eventdisplay files. You should verify that the IRFs are filled correctly and
# that there are no values set to zero within your analysis range.
# 
# Here we peek at the first run in the data release: 64080. The Crab
# should be visible in the events plot.
# 
# You can peek other runs by changing the index 0 to the appropriate
# index.
# 

observations[0].peek(figsize=(25,5))


######################################################################
# Peek at the events and their time/energy/spatial distributions for run
# 64080. We can also peek at the effective area (``aeff``) or energy migration
# matrics (``edisp``) with the ``peek()`` method. 

observations[0].events.peek()


######################################################################
# Part II: Estimate counts and significance
# =========================================
# 


######################################################################
# Set the energy binning
# ----------------------
# 
# The energy axis will determine the bins in which energy is calculated,
# while the true energy axis defines the binning of the energy dispersion
# matrix and effective area. Generally, the true energy axis should be more 
# finely binned than the energy axis and span a larger range of 
# energies, and the energy axis should be binned to match the needs of spectral
# reconstruction.
# 
# Note that if the ``~gammapy.makers.SafeMaskMaker`` (which we will define later) is set
# to exclude events below a given percentage of the effective area, it will
# remove the entire bin containing the energy that corresponds to that
# percentage (which is why the energy axis below extends to broader
# energies than the VERITAS energy sensitivity, in addition to catching
# events for significance calculations that were mis-reconstructed to
# low/high energies). Additionally, spectral bins are determined based on
# the energy axis and cannot be finer or offset from the energy axis bin
# edges.
# 
# Depending on your analysis requirements and your safe mask maker
# definition, ``energy_axis`` may need to be rebinned to ensure that the
# required energies are included. Note that finer binning will result in
# slower spectral/light curve calculations. See :doc:`/tutorials/api/makers.html#safe-data-range-handling`
# for more information on how the safe mask maker works. 
# 

energy_axis = MapAxis.from_energy_bounds("0.01 TeV", "100 TeV", nbin=100)
energy_axis_true = MapAxis.from_energy_bounds(
    "0.01 TeV", "100 TeV", nbin=200, name="energy_true"
)


######################################################################
# Create an exclusion mask
# ------------------------
# 
# Here, we create a spatial mask and append exclusion regions for the
# source region, stars < 8th magnitude, and any other sources that we wish
# to exclude from background calculations.
# 
# By default, stars are excluded with a radius of 0.3 deg and sources with
# a radius of 0.35 deg. This can be increased for brighter sources, as
# necessary.
# 
# To exclude additional sources, more ``regions`` (of any geometry) can be
# appended to the ``all_ex`` list.
# 
# Here, we use the Hipparcos catalog to search for bright stars within
# :math:`1.75\degree` of our source - this radius can be changed in
# ``star_mask``, additionally, to change from the default cut of 8th
# magnitude, an additional masking condition can be used to decrease this
# threshold - here we use 6th magnitude.
# 

exclusion_geom = WcsGeom.create(
    skydir=(target_position.ra.value, target_position.dec.value),
    binsz=0.01,
    width=(4, 4),
    frame="icrs",
    proj="CAR",
)

regions = CircleSkyRegion(center=target_position, radius=0.35 * u.deg)
all_ex = [regions]

star_data = np.loadtxt("$GAMMAPY_DATA/veritas/crab-point-like-ED/Hipparcos_MAG8_1997.dat",usecols=(0, 1, 2, 3))
star_cat = Table(
    {
        "ra": star_data[:, 0],
        "dec": star_data[:, 1],
        "id": star_data[:, 2],
        "mag": star_data[:, 3],
    }
)
star_mask = (
    np.sqrt(
        (star_cat["ra"] - target_position.ra.deg) ** 2
        + (star_cat["dec"] - target_position.dec.deg) ** 2
    ) < 1.75
)

for src in star_cat[(star_mask) & (star_cat["mag"] < 6)]:
    all_ex.append(
        CircleSkyRegion(
            center=SkyCoord(src["ra"], src["dec"], unit="deg", frame="icrs"),
            radius=0.3 * u.deg,
        )
    )
    
exclusion_geom_image = exclusion_geom.to_image() # flatten the energy axis so that the exclusion mask is not energy dependent
exclusion_mask = ~exclusion_geom_image.region_mask(all_ex)


######################################################################
# Define the integration region
# -----------------------------
# 
# Point-like DL3 files can only be analyzed using the reflected regions
# background method and for a pre-determined integration region (which is
# the :math:`\sqrt{\theta^2}` used in IRF simulations), where the number
# of ON counts are determined.
# 
# The default values for moderate/medium cuts are as follows: \* **For
# Eventdisplay files (which applies to the files found in gammapy-data),
# this ON region radius is :math:`\sqrt{0.008}\degree`** \* For VEGAS ITM
# files, this ON region radius is :math:`\sqrt{0.005}\degree` \* For VEGAS
# GEO files, this ON region radius is :math:`0.1 \degree`
# 
# *Note that full-enclosure files are required to use any other
# integration radius size!*
# 

on_region_radius = Angle(f"{np.sqrt(0.008)} deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)
geom = RegionGeom.create(region=on_region, axes=[energy_axis])

######################################################################
# SafeMaskMaker
# -------------
# 
# The ``SafeMaskMaker`` sets the boundaries of our analysis based on the
# uncertainties contained in the instrument response functions (IRFs).
# 
# For VERITAS point-like analysis (both ED and VEGAS), the following
# methods are strongly recommended: \* ``offset-max``: Sets the maximum
# radial offset from the camera center within which we accept events. This
# is set to slightly below the edge of the VERITAS FoV to reduce artifacts
# at the edge of the FoV and events with poor angular reconstruction. \*
# ``edisp-bias``: Removes events which are reconstructed with energies
# that have :math:`>5\%` energy bias. \* ``aeff-max``: Removes events
# which are reconstructed to :math:`<10\%` of the maximum value of the
# effective area. These are important to remove for spectral analysis,
# since they have large uncertainties on their reconstructed energies.
# 

safe_mask_maker = SafeMaskMaker(methods=["offset-max","aeff-max","edisp-bias"], aeff_percent=5,bias_percent=5,offset_max=1.70*u.deg)



######################################################################
# We will now run the data reduction chain to calculate our ON and OFF
# counts. To get a significance for the whole energy range (to match VERITAS packages), 
# remove the ``SafeMaskMaker`` from being applied to `dataset_on_off`. 
# 
# You need to add ``containment_correction=True`` as an argument to
# ``dataset_maker`` if you are using full-enclosure DL3 files.
# 
# The parameters of the reflected background regions can be changed using
# the
# ```ReflectedRegionsFinder`` :doc:`/tutorials/api/gammapy.makers.ReflectedRegionsFinder`
# which is passed as an argument to the
# ``ReflectedRegionsBackgroundMaker``). To use the default values, do not
# pass a region_finder argument.
# 

dataset_maker = SpectrumDatasetMaker(
    selection=["counts", "exposure", "edisp"]
)
dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)
bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)

datasets = Datasets()

for obs_id, observation in zip(obs_ids, observations):
    dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation)
    dataset_on_off = safe_mask_maker.run(dataset_on_off, observation)
    dataset_on_off = bkg_maker.run(dataset, observation)
    datasets.append(dataset_on_off)


######################################################################
# The plot below will show your exclusion regions in black and your
# background regions with coloured circles. You should check to make sure
# these regions are sensible and that none of your background regions
# overlap with your exclusion regions.
# 

plt.figure(figsize=(7,7))
ax = exclusion_mask.plot()
on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="k")
plot_spectrum_datasets_off_regions(ax=ax, datasets=datasets)
plt.show()


######################################################################
# Significance analysis results
# -----------------------------
# 


######################################################################
# Here, we display the results of the significance analysis.
# ``info_table`` can be modified with ``cumulative = False`` to display a
# table with rows that correspond to the values for each run separately.
# 
# However, ``cumulative = True`` is needed to produce the combined values
# in the next cell.
# 

info_table = datasets.info_table(cumulative = True)
info_table

print(f"ON: {info_table['counts'][-1]}")
print(f"OFF: {info_table['counts_off'][-1]}")
print(f"Significance: {info_table['sqrt_ts'][-1]:.2f} sigma")
print(f"Alpha: {info_table['alpha'][-1]:.2f}")


######################################################################
# We can also plot the cumulative excess counts and significance over
# time. For a steady source, we generally expect excess to increase
# linearly with time and for significance to increase as
# :math:`\sqrt{\textrm{time}}`.
# 

fig, (ax_excess, ax_sqrt_ts) = plt.subplots(figsize=(10, 4), ncols=2, nrows=1)
ax_excess.plot(
    info_table["livetime"].to("h"),
    info_table["excess"],
    marker="o",
)

ax_excess.set_title("Excess")
ax_excess.set_xlabel("Livetime [h]")
ax_excess.set_ylabel("Excess events")

ax_sqrt_ts.plot(
    info_table["livetime"].to("h"),
    info_table["sqrt_ts"],
    marker="o",
)

ax_sqrt_ts.set_title("Significance")
ax_sqrt_ts.set_xlabel("Livetime [h]")
ax_sqrt_ts.set_ylabel("Significance [sigma]")
plt.show()


######################################################################
# Part III: Make a spectrum
# =========================
# 

######################################################################
# Now, we’ll calculate the source spectrum. This uses a forward-folding
# approach that will assume a given spectrum and fit the counts calculated
# above to that spectrum in each energy bin specified by the
# ``energy_axis``.
# 
# For this reason, it’s important that ``spectral_model`` be set as
# closely as possible to the expected spectrum - for the Crab nebula, this
# is a log parabola. If you don’t know the spectrum a priori, this can be
# adjusted iteratively to get the best fit. Here, we are doing a 1D fit,
# so we assign only the spectral model to the datasets’ ``SkyModel``
# before running the fit on our datasets.
# 
# See 


spectral_model = LogParabolaSpectralModel(
    amplitude=3.75e-11 * u.Unit("cm-2 s-1 TeV-1"),
    alpha=2.45,
    beta=0.15,
    reference=1 * u.TeV,
)

model = SkyModel(spectral_model=spectral_model, name="crab")

datasets.models = [model]

fit_joint = Fit()
result_joint = fit_joint.run(datasets=datasets)


######################################################################
# The best-fit spectral parameters are shown in this table.
# 

display(datasets.models.to_parameters_table())


######################################################################
# We can inspect how well our data fit the model’s predicted counts in
# each ``energy_axis`` bin.
# 

ax_spectrum, ax_residuals = datasets[0].plot_fit()
ax_spectrum.set_ylim(0.1, 40)
plt.show()


######################################################################
# We can now calculate flux points to get a spectrum by fitting the
# ``result_joint`` model’s amplitude in selected energy bands (defined by
# ``energy_edges``). We set ``selection_optional = "all"`` in
# ``FluxPointsEstimator``, which will include a calcuation for the upper
# limits in bins with a significance :math:`< 2\sigma`.
# 
# Upper limit and/or flux uncertainty calculations may not work for bins
# without a sufficient number of excess counts. To improve this, we can
# add ``fpe.norm.min``, ``fpe.norm.max``, and ``fpe.norm.scan_values`` to
# increase the range for which the norm (the normalization of the
# predicted signal counts; see orange curve above) is being fit. This will
# help avoid situations where the norm profiles are insufficient to
# calculate a 2 sigma point or 95% C.L. upper limit.
# 
# Flux points values can be viewed with ``flux_points.to_table()`` and/or
# saved as an ascii or ecsv file with ``flux_points.write()``.
# 

fpe = FluxPointsEstimator(
    energy_edges = np.logspace(-0.7,1.5,12)*u.TeV, 
    source = "crab", 
    selection_optional = "all",
)
fpe.norm.min=-1e2
fpe.norm.max=1e2
fpe.norm.scan_values=np.array(np.linspace(-10,10,10))
flux_points = fpe.run(datasets=datasets)


######################################################################
# Now, we can plot our flux points along with the best-fit spectral model.
# For the Crab, curvature is clearly present in the spectrum and we can
# see that the flux points closely follow the
# ``LogParabolaSpectralModel``.
# 

flux_points_dataset = FluxPointsDataset(
    data=flux_points, models=datasets.models)

flux_points_dataset.plot_fit()
plt.ylim(1e-20,)
plt.show()


######################################################################
# Part IV: Make a lightcurve and caluclate integral flux
# ======================================================
# 


######################################################################
# Integral flux can be calculated by integrating the spectral model we fit
# earlier. This will be a model-dependent flux estimate, so the choice of
# spectral model should match the data as closely as possible.
# Additionally, sources with poor spectral fits due to low statistics may
# have inaccurate flux estimations.
# 
# ``e_min`` and ``e_max`` should be adjusted depending on the analysis
# requirements - different gamma/hadron cuts in Eventdisplay/VEGAS and/or
# observing conditions will lead to different energy thresholds. Note that
# the ``energy_axis`` lower bin edge containing ``e_min`` is the value
# used in the integral flux calculation, which is *not* necessarily
# identical to the energy threshold the user defines.
# 

e_min = 0.25 * u.TeV
e_max = 30 * u.TeV

flux,flux_err = result_joint.models["crab"].spectral_model.integral_error(e_min,e_max)
print(f"Integral flux > {e_min}: {flux.value:.2} +/- {flux_err.value:.2} {flux.unit}")


######################################################################
# Finally, we’ll create a run-wise binned light curve. See the `light
# curves for
# flares <https://docs.gammapy.org/1.3/tutorials/analysis-time/light_curve_flare.html>`__
# for instructions on how to set up sub-run binning. Here, we set our
# energy edges so that the light curve has an energy threshold of 0.25 TeV
# and will plot upper limits for time bins with significance
# :math:`<2 \sigma`.
# 

lc_maker = LightCurveEstimator(
    energy_edges=[0.25, 30] * u.TeV, source="crab", reoptimize=False
)
lc_maker.n_sigma_ul = 2 
lc_maker.selection_optional = ["ul"] 
lc = lc_maker.run(datasets)

fig, ax = plt.subplots(
    figsize=(8, 6),
)
lc.sqrt_ts_threshold_ul = 2
lc.plot(ax=ax, axis_name="time",sed_type='flux')

# these lines will print out a table with values for individual light curve bins
table = lc.to_table(format="lightcurve", sed_type="flux")
display(table["time_min", "time_max", "flux", "flux_err"])

