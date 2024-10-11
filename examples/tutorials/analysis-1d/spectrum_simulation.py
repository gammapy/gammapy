"""
1D spectrum simulation
======================

Simulate a number of spectral on-off observations of a source with a power-law spectral
model using the CTA 1DC response and fit them with the assumed spectral model.

Prerequisites
-------------

-  Knowledge of spectral extraction and datasets used in gammapy, see
   for instance the :doc:`spectral analysis
   tutorial </tutorials/analysis-1d/spectral_analysis>`

Context
-------

To simulate a specific observation, it is not always necessary to
simulate the full photon list. For many uses cases, simulating directly
a reduced binned dataset is enough: the IRFs reduced in the correct
geometry are combined with a source model to predict an actual number of
counts per bin. The latter is then used to simulate a reduced dataset
using Poisson probability distribution.

This can be done to check the feasibility of a measurement, to test
whether fitted parameters really provide a good fit to the data etc.

Here we will see how to perform a 1D spectral simulation of a CTAO
observation, in particular, we will generate OFF observations following
the template background stored in the CTAO IRFs.

**Objective: simulate a number of spectral ON-OFF observations of a
source with a power-law spectral model with CTAO using the CTA 1DC
response, fit them with the assumed spectral model and check that the
distribution of fitted parameters is consistent with the input values.**

Proposed approach
-----------------

We will use the following classes and functions:

-  `~gammapy.datasets.SpectrumDatasetOnOff`
-  `~gammapy.datasets.SpectrumDataset`
-  `~gammapy.irf.load_irf_dict_from_file`
-  `~gammapy.modeling.models.PowerLawSpectralModel`

"""

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion

# %matplotlib inline
import matplotlib.pyplot as plt

######################################################################
# Setup
# -----
#
from IPython.display import display
from gammapy.data import FixedPointingInfo, Observation, observatory_locations
from gammapy.datasets import Datasets, SpectrumDataset, SpectrumDatasetOnOff
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# Simulation of a single spectrum
# -------------------------------
#
# To do a simulation, we need to define the observational parameters like
# the livetime, the offset, the assumed integration radius, the energy
# range to perform the simulation for and the choice of spectral model. We
# then use an in-memory observation which is convolved with the IRFs to
# get the predicted number of counts. This is Poisson fluctuated using
# the `fake()` to get the simulated counts for each observation.
#

# Define simulation parameters parameters
livetime = 1 * u.h

pointing_position = SkyCoord(0, 0, unit="deg", frame="galactic")
# We want to simulate an observation pointing at a fixed position in the sky.
# For this, we use the `FixedPointingInfo` class
pointing = FixedPointingInfo(
    fixed_icrs=pointing_position.icrs,
)
offset = 0.5 * u.deg

# Reconstructed and true energy axis
energy_axis = MapAxis.from_edges(
    np.logspace(-0.5, 1.0, 10), unit="TeV", name="energy", interp="log"
)
energy_axis_true = MapAxis.from_edges(
    np.logspace(-1.2, 2.0, 31), unit="TeV", name="energy_true", interp="log"
)

on_region_radius = Angle("0.11 deg")

center = pointing_position.directional_offset_by(
    position_angle=0 * u.deg, separation=offset
)
on_region = CircleSkyRegion(center=center, radius=on_region_radius)

# Define spectral model - a simple Power Law in this case
model_simu = PowerLawSpectralModel(
    index=3.0,
    amplitude=2.5e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1 * u.TeV,
)
print(model_simu)
# we set the sky model used in the dataset
model = SkyModel(spectral_model=model_simu, name="source")

######################################################################
# Load the IRFs
# In this simulation, we use the CTA-1DC IRFs shipped with Gammapy.
irfs = load_irf_dict_from_file(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)

location = observatory_locations["cta_south"]
obs = Observation.create(
    pointing=pointing,
    livetime=livetime,
    irfs=irfs,
    location=location,
)
print(obs)

######################################################################
# Simulate a spectra
#

# Make the SpectrumDataset
geom = RegionGeom.create(region=on_region, axes=[energy_axis])

dataset_empty = SpectrumDataset.create(
    geom=geom, energy_axis_true=energy_axis_true, name="obs-0"
)
maker = SpectrumDatasetMaker(selection=["exposure", "edisp", "background"])

dataset = maker.run(dataset_empty, obs)

# Set the model on the dataset, and fake
dataset.models = model
dataset.fake(random_state=42)
print(dataset)


######################################################################
# You can see that background counts are now simulated
#


######################################################################
# On-Off analysis
# ~~~~~~~~~~~~~~~
#
# To do an on off spectral analysis, which is the usual science case, the
# standard would be to use `SpectrumDatasetOnOff`, which uses the
# acceptance to fake off-counts. Please also refer to the `Dataset simulations`
# section in the :doc:`/tutorials/analysis-1d/spectral_analysis_rad_max` tutorial,
# dealing with simulations based on observations of real off counts.
#

dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
    dataset=dataset, acceptance=1, acceptance_off=5
)
dataset_on_off.fake(npred_background=dataset.npred_background())
print(dataset_on_off)


######################################################################
# You can see that off counts are now simulated as well. We now simulate
# several spectra using the same set of observation conditions.
#

# %%time

n_obs = 100
datasets = Datasets()

for idx in range(n_obs):
    dataset_on_off.fake(random_state=idx, npred_background=dataset.npred_background())
    dataset_fake = dataset_on_off.copy(name=f"obs-{idx}")
    dataset_fake.meta_table["OBS_ID"] = [idx]
    datasets.append(dataset_fake)

table = datasets.info_table()
display(table)


######################################################################
# Before moving on to the fit let’s have a look at the simulated
# observations.
#

fix, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].hist(table["counts"])
axes[0].set_xlabel("Counts")
axes[1].hist(table["counts_off"])
axes[1].set_xlabel("Counts Off")
axes[2].hist(table["excess"])
axes[2].set_xlabel("excess")
plt.show()


######################################################################
# Now, we fit each simulated spectrum individually
#

# %%time
results = []

fit = Fit()

for dataset in datasets:
    dataset.models = model.copy()
    result = fit.optimize(dataset)
    results.append(
        {
            "index": result.parameters["index"].value,
            "amplitude": result.parameters["amplitude"].value,
        }
    )


######################################################################
# We take a look at the distribution of the fitted indices. This matches
# very well with the spectrum that we initially injected.
#

fig, ax = plt.subplots()
index = np.array([_["index"] for _ in results])
ax.hist(index, bins=10, alpha=0.5)
ax.axvline(x=model_simu.parameters["index"].value, color="red")
ax.set_xlabel("Reconstructed spectral index")
print(f"index: {index.mean()} += {index.std()}")
plt.show()


######################################################################
# Exercises
# ---------
#
# -  Change the observation time to something longer or shorter. Does the
#    observation and spectrum results change as you expected?
# -  Change the spectral model, e.g. add a cutoff at 5 TeV, or put a
#    steep-spectrum source with spectral index of 4.0
# -  Simulate spectra with the spectral model we just defined. How much
#    observation duration do you need to get back the injected parameters?
#
