"""
Point source sensitivity
========================

Estimate the CTA sensitivity for a point-like IRF at a fixed zenith angle and fixed offset.

Introduction
------------

This notebook explains how to estimate the CTA sensitivity for a
point-like IRF at a fixed zenith angle and fixed offset using the full
containment IRFs distributed for the CTA 1DC. The significance is
computed for a 1D analysis (On-OFF regions) and the LiMa formula.

We use here an approximate approach with an energy dependent integration
radius to take into account the variation of the PSF. We will first
determine the 1D IRFs including a containment correction.

We will be using the following Gammapy class:

-  `~gammapy.estimators.SensitivityEstimator`

"""


######################################################################
# Setup
# -----
# 
# As usual, we’ll start with some setup …
# 

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord

from gammapy.irf import load_cta_irfs
from gammapy.makers import SpectrumDatasetMaker
from gammapy.data import Observation, observatory_locations
from gammapy.estimators import SensitivityEstimator
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.maps import MapAxis, RegionGeom

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

energy_axis = MapAxis.from_energy_bounds("0.03 TeV", "30 TeV", nbin=20)
energy_axis_true = MapAxis.from_energy_bounds(
    "0.01 TeV", "100 TeV", nbin=100, name="energy_true"
)

geom = RegionGeom.create("icrs;circle(0, 0.5, 0.1)", axes=[energy_axis])

empty_dataset = SpectrumDataset.create(
    geom=geom, energy_axis_true=energy_axis_true
)


######################################################################
# Load IRFs and prepare dataset
# -----------------------------
# 
# We extract the 1D IRFs from the full 3D IRFs provided by CTA.
# 

irfs = load_cta_irfs(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)
location = observatory_locations["cta_south"]
pointing = SkyCoord("0 deg", "0 deg")
obs = Observation.create(
    pointing=pointing, irfs=irfs, livetime="5 h", location=location
)

spectrum_maker = SpectrumDatasetMaker(
    selection=["exposure", "edisp", "background"]
)
dataset = spectrum_maker.run(empty_dataset, obs)


######################################################################
# Now we correct for the energy dependent region size:
# 

containment = 0.68

# correct exposure
dataset.exposure *= containment

# correct background estimation
on_radii = obs.psf.containment_radius(
    energy_true=energy_axis.center, offset=0.5 * u.deg, fraction=containment
)
factor = (1 - np.cos(on_radii)) / (1 - np.cos(geom.region.radius))
dataset.background *= factor.value.reshape((-1, 1, 1))


######################################################################
# And finally define a `SpectrumDatasetOnOff` with an alpha of ``0.2``.
# The off counts are created from the background model:
# 

dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
    dataset=dataset, acceptance=1, acceptance_off=5
)


######################################################################
# Compute sensitivity
# -------------------
# 
# We impose a minimal number of expected signal counts of 5 per bin and a
# minimal significance of 3 per bin. We assume an alpha of 0.2 (ratio
# between ON and OFF area). We then run the sensitivity estimator.
# 

sensitivity_estimator = SensitivityEstimator(
    gamma_min=5, n_sigma=3, bkg_syst_fraction=0.10
)
sensitivity_table = sensitivity_estimator.run(dataset_on_off)


######################################################################
# Results
# -------
# 
# The results are given as an Astropy table. A column criterion allows to
# distinguish bins where the significance is limited by the signal
# statistical significance from bins where the sensitivity is limited by
# the number of signal counts. This is visible in the plot below.
# 

# Show the results table
sensitivity_table

# Save it to file (could use e.g. format of CSV or ECSV or FITS)
# sensitivity_table.write('sensitivity.ecsv', format='ascii.ecsv')

# Plot the sensitivity curve
t = sensitivity_table

is_s = t["criterion"] == "significance"
plt.plot(
    t["energy"][is_s],
    t["e2dnde"][is_s],
    "s-",
    color="red",
    label="significance",
)

is_g = t["criterion"] == "gamma"
plt.plot(
    t["energy"][is_g], t["e2dnde"][is_g], "*-", color="blue", label="gamma"
)
is_bkg_syst = t["criterion"] == "bkg"
plt.plot(
    t["energy"][is_bkg_syst],
    t["e2dnde"][is_bkg_syst],
    "v-",
    color="green",
    label="bkg syst",
)

plt.loglog()
plt.xlabel(f"Energy ({t['energy'].unit})")
plt.ylabel(f"Sensitivity ({t['e2dnde'].unit})")
plt.legend();


######################################################################
# We add some control plots showing the expected number of background
# counts per bin and the ON region size cut (here the 68% containment
# radius of the PSF).
# 

# Plot expected number of counts for signal and background
fig, ax1 = plt.subplots()
# ax1.plot( t["energy"], t["excess"],"o-", color="red", label="signal")
ax1.plot(
    t["energy"], t["background"], "o-", color="black", label="blackground"
)

ax1.loglog()
ax1.set_xlabel(f"Energy ({t['energy'].unit})")
ax1.set_ylabel("Expected number of bkg counts")

ax2 = ax1.twinx()
ax2.set_ylabel(f"ON region radius ({on_radii.unit})", color="red")
ax2.semilogy(t["energy"], on_radii, color="red", label="PSF68")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_ylim(0.01, 0.5)


######################################################################
# Exercises
# ---------
# 
# -  Also compute the sensitivity for a 20 hour observation
# -  Compare how the sensitivity differs between 5 and 20 hours by
#    plotting the ratio as a function of energy.
# 



