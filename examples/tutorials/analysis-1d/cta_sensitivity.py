"""
Point source sensitivity
========================

Estimate the CTAO sensitivity for a point-like IRF at a fixed zenith angle and fixed offset.

Introduction
------------

This notebook explains how to estimate the CTAO sensitivity for a
point-like IRF at a fixed zenith angle and fixed offset, using the full
containment IRFs distributed for the CTA 1DC. The significance is
computed for a 1D analysis (ON-OFF regions) with the
`Li & Ma formula <https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract>`__.

We use here an approximate approach with an energy dependent integration
radius to take into account the variation of the PSF. We will first
determine the 1D IRFs including a containment correction.

We will be using the following Gammapy class:

-  `~gammapy.estimators.SensitivityEstimator`

"""

from cycler import cycler
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

# %matplotlib inline
from regions import CircleSkyRegion
import matplotlib.pyplot as plt

######################################################################
# Setup
# -----
#
# As usual, we’ll start with some setup …
#
from IPython.display import display
from gammapy.data import FixedPointingInfo, Observation, observatory_locations
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import FluxPoints, SensitivityEstimator
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom
from gammapy.maps.axes import UNIT_STRING_FORMAT

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()

######################################################################
# Define analysis region and energy binning
# -----------------------------------------
#
# Here we assume a source at 0.5 degree from pointing position. We perform
# a simple energy independent extraction for now with a radius of 0.1
# degree.
#

energy_axis = MapAxis.from_energy_bounds(0.03 * u.TeV, 30 * u.TeV, nbin=20)
energy_axis_true = MapAxis.from_energy_bounds(
    0.01 * u.TeV, 100 * u.TeV, nbin=100, name="energy_true"
)

pointing = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)
pointing_info = FixedPointingInfo(fixed_icrs=pointing)
offset = 0.5 * u.deg

source_position = pointing.directional_offset_by(0 * u.deg, offset)
on_region_radius = 0.1 * u.deg
on_region = CircleSkyRegion(source_position, radius=on_region_radius)

geom = RegionGeom.create(on_region, axes=[energy_axis])
empty_dataset = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)

######################################################################
# Load IRFs and prepare dataset
# -----------------------------
#
# We extract the 1D IRFs from the full 3D IRFs provided by CTAO.
#

irfs = load_irf_dict_from_file(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)
location = observatory_locations["ctao_south"]
livetime = 50.0 * u.h
obs = Observation.create(
    pointing=pointing_info, irfs=irfs, livetime=livetime, location=location
)


######################################################################
# Initiate and run the `~gammapy.makers.SpectrumDatasetMaker`.
#
# Note that here we ensure ``containment_correction=False`` which allows us to
# apply our own containment correction in the next part of the tutorial.
#

spectrum_maker = SpectrumDatasetMaker(
    selection=["exposure", "edisp", "background"],
    containment_correction=False,
)
dataset = spectrum_maker.run(empty_dataset, obs)

######################################################################
# Now we correct for the energy dependent region size.
#
# **Note**: In the calculation of the containment radius, we use the point spread function
# which is defined dependent on true energy to compute the correction we apply in reconstructed
# energy, thus neglecting the energy dispersion in this step.
#
# Start by correcting the exposure:
#

containment = 0.68
dataset.exposure *= containment

######################################################################
# Next, correct the background estimation.
#
# Warning: this neglects the energy dispersion by computing the containment
# radius from the PSF in true energy but using the reco energy axis.
#

on_radii = obs.psf.containment_radius(
    energy_true=energy_axis.center, offset=offset, fraction=containment
)
factor = (1 - np.cos(on_radii)) / (1 - np.cos(on_region_radius))
dataset.background *= factor.value.reshape((-1, 1, 1))


######################################################################
# Finally, define a `~gammapy.datasets.SpectrumDatasetOnOff` with an alpha of 0.2.
# The off counts are created from the background model:
#

dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
    dataset=dataset, acceptance=1, acceptance_off=5
)


######################################################################
# Compute sensitivity
# -------------------
#
# We impose a minimal number of expected signal counts of 10 per bin and a
# minimal significance of 5 per bin. The excess must also be larger than 5% of the background.
#
# We assume an alpha of 0.2 (ratio between ON and OFF area). We then run the sensitivity estimator.
#
# These are the conditions imposed in standard CTAO sensitivity computations.

sensitivity_estimator = SensitivityEstimator(
    gamma_min=10,
    n_sigma=5,
    bkg_syst_fraction=0.05,
)
sensitivity_table = sensitivity_estimator.run(dataset_on_off)

######################################################################
# Results
# -------
#
# The results are given as a `~astropy.table.Table`, which can be written to
# disk utilising the usual `~astropy.table.Table.write` method.
# A column criterion allows us
# to distinguish bins where the significance is limited by the signal
# statistical significance from bins where the sensitivity is limited by
# the number of signal counts. This is visible in the plot below.
#

display(sensitivity_table)


######################################################################
# Plot the sensitivity curve
#


fig, ax = plt.subplots()

ax.set_prop_cycle(cycler("marker", "s*v") + cycler("color", "rgb"))

for criterion in ("significance", "gamma", "bkg"):
    mask = sensitivity_table["criterion"] == criterion
    t = sensitivity_table[mask]

    ax.errorbar(
        t["e_ref"],
        t["e2dnde"],
        xerr=0.5 * (t["e_max"] - t["e_min"]),
        label=criterion,
        linestyle="",
    )

ax.loglog()

ax.set_xlabel(f"Energy [{t['e_ref'].unit.to_string(UNIT_STRING_FORMAT)}]")
ax.set_ylabel(f"Sensitivity [{t['e2dnde'].unit.to_string(UNIT_STRING_FORMAT)}]")

ax.legend()

plt.show()

######################################################################
# We add some control plots showing the expected number of background
# counts per bin and the ON region size cut (here the 68% containment
# radius of the PSF).
#
# Plot expected number of counts for signal and background.
#

fig, ax1 = plt.subplots()
ax1.plot(
    sensitivity_table["e_ref"],
    sensitivity_table["background"],
    "o-",
    color="black",
    label="background",
)

ax1.loglog()
ax1.set_xlabel(f"Energy [{t['e_ref'].unit.to_string(UNIT_STRING_FORMAT)}]")
ax1.set_ylabel("Expected number of bkg counts")

ax2 = ax1.twinx()
ax2.set_ylabel(
    f"ON region radius [{on_radii.unit.to_string(UNIT_STRING_FORMAT)}]", color="red"
)
ax2.semilogy(sensitivity_table["e_ref"], on_radii, color="red", label="PSF68")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_ylim(0.01, 0.5)
plt.show()

######################################################################
# Obtaining an integral flux sensitivity
# --------------------------------------
#
# It is often useful to obtain the integral sensitivity above a certain
# threshold. In this case, it is simplest to use a dataset with one energy bin
# while setting the high energy edge to a very large value.
# Here, we simply squash the previously created dataset into one with a single
# energy
#

dataset_on_off1 = dataset_on_off.to_image()
sensitivity_estimator = SensitivityEstimator(
    gamma_min=5, n_sigma=3, bkg_syst_fraction=0.10
)
sensitivity_table = sensitivity_estimator.run(dataset_on_off1)
print(sensitivity_table)


######################################################################
# To get the integral flux, we convert to a `~gammapy.estimators.FluxPoints` object
# that does the conversion internally.
#

flux_points = FluxPoints.from_table(
    sensitivity_table,
    sed_type="e2dnde",
    reference_model=sensitivity_estimator.spectral_model,
)
print(
    f"Integral sensitivity in {livetime:.2f} above {energy_axis.edges[0]:.2e} "
    f"is {np.squeeze(flux_points.flux.quantity):.2e}"
)

######################################################################
# Exercises
# ---------
#
# -  Compute the sensitivity for a 20 hour observation
# -  Compare how the sensitivity differs between 5 and 20 hours by
#    plotting the ratio as a function of energy.
#
