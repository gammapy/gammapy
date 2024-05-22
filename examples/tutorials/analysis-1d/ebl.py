"""
Account for spectral absorption due to the EBL
==============================================

Gamma rays emitted extra-galactic objects, eg blazars, interact with the
photons of the Extragalactic Background Light (EBL) through pair
production and are attenuated, thus modifying the intrinsic spectrum.

Various models of the EBL are supplied in `GAMMAPY_DATA`. This
notebook shows how to use these models to correct for this interaction.

"""

######################################################################
# Setup
# -----
#
# As usual, we’ll start with the standard imports …
#

from copy import deepcopy
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from regions import CircleSkyRegion
import matplotlib.pyplot as plt
from gammapy.data import DataStore
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.estimators import FluxPointsEstimator
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    EBLAbsorptionNormSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
)

######################################################################
# Select the data
# ---------------
#
# We will analyse 6 observations of the blazars PKS~2155-304 taken in 2008
# by H.E.S.S. when it was in a steady state.
#

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")

# spatial selection
target_position = SkyCoord(329.71693826 * u.deg, -30.2255890 * u.deg, frame="icrs")
selection_pos = dict(
    type="sky_circle",
    frame="icrs",
    lon=target_position.ra,
    lat=target_position.dec,
    radius=2 * u.deg,
)

# time selection
selection_time = dict(type="time_box", time_range=Time([54705, 54709], format="mjd"))

# apply the selections
obs_ids = data_store.obs_table.select_observations([selection_pos, selection_time])[
    "OBS_ID"
]
observations = data_store.get_observations(obs_ids)
print(f"Number of selected observations : {len(observations)}")


######################################################################
# Data reduction
# --------------
#
# Here, we do a 1D spectral analysis (see:
# :doc:`/tutorials/analysis-1d/spectral_analysis` for details). You may
# also choose to do a 3D analysis.
#

# define the axes and the geom
energy_axis = MapAxis.from_energy_bounds("0.2 TeV", "20 TeV", nbin=5, per_decade=True)
energy_axis_true = MapAxis.from_energy_bounds(
    "0.1 TeV", "40 TeV", nbin=20, name="energy_true", per_decade=True
)

on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)

geom = RegionGeom.create(region=on_region, axes=[energy_axis])

dataset_maker = SpectrumDatasetMaker(
    containment_correction=True, selection=["counts", "exposure", "edisp"]
)
bkg_maker = ReflectedRegionsBackgroundMaker()
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

# Do the data reduction
datasets = Datasets()
dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)

for obs in observations:
    dataset = dataset_maker.run(dataset_empty.copy(), obs)
    dataset_on_off = bkg_maker.run(dataset, obs)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, obs)
    datasets.append(dataset_on_off)

# Here, we stack the datasets to save time during the fitting. You can also do a joint fitting
stacked = datasets.stack_reduce(name="pks2155-304")

print(stacked)


######################################################################
# Model the observed spectrum
# ---------------------------
#
# The observed spectrum is already attenuated due to the EBL. Assuming
# that the intrinsic spectrum is a power law, the observed spectrum is a
# `CompoundSpectrumModel` given by the product of an EBL model with the
# intrinsic model.
#

# define the power law
index = 3.53
amplitude = 1.81 * 1e-12 * u.Unit("cm-2 s-1 TeV-1")
reference = 1 * u.TeV
pwl = PowerLawSpectralModel(index=index, amplitude=amplitude, reference=reference)

# Specify the redshift of the source
redshift = 0.116
# Load the EBL model. Here we use the model from Dominguez 2011
absorption = EBLAbsorptionNormSpectralModel.read_builtin("dominguez", redshift=redshift)


######################################################################
# We keep the paramters of the EBL model frozen in this example. For a
# list of the other available models, see
# :doc:`/api/gammapy.modeling.models.EBL_DATA_BUILTIN`.
#

# The power-law model is multiplied by the EBL norm spectral model
spectral_model = pwl * absorption
print(spectral_model)

# Now, create a sky model and proceed with the fit
sky_model = SkyModel(spatial_model=None, spectral_model=spectral_model, name="pks2155")

stacked.models = sky_model

fit = Fit()
result = fit.run(datasets=[stacked])

# we make a copy here to compare it later
model_best = sky_model.copy()

print(result.models.to_parameters_table())

# To see the covariance,
model_best.covariance.plot_correlation()
plt.show()


######################################################################
# Get the flux points
# ===================
#
# To get the observed flux points, just run the `FluxPointsEstimator`
# normally
#

energy_edges = energy_axis.edges
fpe = FluxPointsEstimator(
    energy_edges=energy_edges, source="pks2155", selection_optional="all"
)
flux_points_obs = fpe.run(datasets=[stacked])


######################################################################
# To get the deabsorbed flux points (ie, intrinsic points), we simply need
# to set the reference model to the best fit power law instead of the
# compound model. We first make a copy of the computed flux points
#

flux_points_intrinsic = deepcopy(flux_points_obs)
flux_points_intrinsic._reference_model = SkyModel(spectral_model=pwl)

print(flux_points_obs._reference_model)

print(flux_points_intrinsic._reference_model)


######################################################################
# Set the covariance on the power law model correctly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To set the covariance on the powerlaw, we must extract the relevant
# values from the full covariance
#


pwl.covariance = spectral_model.covariance.get_subcovariance(pwl.covariance.parameters)


######################################################################
# Plot the observed and intrinsic fluxes
# --------------------------------------
#

# sphinx_gallery_thumbnail_number = 2
plt.figure()
sed_type = "e2dnde"
energy_bounds = [0.2, 20] * u.TeV
ax = flux_points_obs.plot(sed_type=sed_type, label="observed", color="navy")
flux_points_intrinsic.plot(ax=ax, sed_type=sed_type, label="intrinsic", color="red")

model_best.spectral_model.plot(
    ax=ax, energy_bounds=energy_bounds, sed_type=sed_type, color="blue"
)
model_best.spectral_model.plot_error(
    ax=ax, energy_bounds=energy_bounds, sed_type="e2dnde", facecolor="blue"
)

pwl.plot(ax=ax, energy_bounds=energy_bounds, sed_type=sed_type, color="tomato")
pwl.plot_error(
    ax=ax, energy_bounds=energy_bounds, sed_type=sed_type, facecolor="tomato"
)
plt.ylim(bottom=1e-13)
plt.legend()
plt.show()
