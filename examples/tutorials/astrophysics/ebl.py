"""
Account for spectral absorption due to the EBL
==============================================

Gamma rays emitted from extra-galactic objects, eg blazars, interact
with the photons of the Extragalactic Background Light (EBL) through
pair production and are attenuated, thus modifying the intrinsic
spectrum.

Various models of the EBL are supplied in `GAMMAPY_DATA`. This
notebook shows how to use these models to correct for this interaction.

"""

######################################################################
# Setup
# -----
#
# As usual, we’ll start with the standard imports …
#
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from gammapy.catalog import SourceCatalog4FGL
from gammapy.datasets import SpectrumDatasetOnOff
from gammapy.estimators import FluxPointsEstimator
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    EBL_DATA_BUILTIN,
    EBLAbsorptionNormSpectralModel,
    GaussianPrior,
    PowerLawSpectralModel,
    SkyModel,
    TemplateNDSpectralModel,
)
from gammapy.maps import MapAxis, RegionNDMap, RegionGeom

######################################################################
# Load the data
# -------------
#
# We will use 6 observations of the blazars PKS 2155-304 taken in 2008 by
# H.E.S.S. when it was in a steady state. The data have already been
# reduced to OGIP format `~gammapy.datasets.SpectrumDatasetOnOff` following the procedure
# :doc:`/tutorials/analysis-1d/spectral_analysis` tutorial using a
# reflected regions background estimation. The spectra and IRFs from the
# 6 observations have been stacked together.
#
# We will load this dataset as a `~gammapy.datasets.SpectrumDatasetOnOff` and proceed with
# the modeling. You can do a 3D analysis as well.
#

dataset = SpectrumDatasetOnOff.read(
    "$GAMMAPY_DATA/PKS2155-steady/pks2155-304_steady.fits.gz"
)

print(dataset)


######################################################################
# Model the observed spectrum
# ---------------------------
#
# The observed spectrum is already attenuated due to the EBL. Assuming
# that the intrinsic spectrum is a power law, the observed spectrum is a
# `~gammapy.modeling.models.CompoundSpectralModel` given by the product of an
# EBL model with the intrinsic model.
#


######################################################################
# For a list of available models, see
# :doc:`/api/gammapy.modeling.models.EBL_DATA_BUILTIN`.
#

print(EBL_DATA_BUILTIN.keys())

######################################################################
# To use other EBL models, you need to save the optical depth as a
# function of energy and redshift as an XSPEC model.
# Alternatively, you can use packages like `ebltable <https://github.com/me-manu/ebltable>`_
# which shows how to interface other EBL models with Gammapy.
#

######################################################################
# Define the power law
#
index = 2.3
amplitude = 1.81 * 1e-12 * u.Unit("cm-2 s-1 TeV-1")
reference = 1 * u.TeV
pwl = PowerLawSpectralModel(index=index, amplitude=amplitude, reference=reference)
pwl.index.frozen = False
# Specify the redshift of the source
redshift = 0.116

# Load the EBL model. Here we use the model from Dominguez, 2011
absorption = EBLAbsorptionNormSpectralModel.read_builtin("dominguez", redshift=redshift)


# The power-law model is multiplied by the EBL to get the final model
spectral_model = pwl * absorption
print(spectral_model)

######################################################################
# Now, create a sky model and proceed with the fit
#
sky_model = SkyModel(spatial_model=None, spectral_model=spectral_model, name="pks2155")

dataset.models = sky_model

######################################################################
# Note that since this dataset has been produced
# by a reflected region analysis, it uses ON-OFF statistic
# and does not require a background model.
#

fit = Fit()
result = fit.run(datasets=[dataset])

# we make a copy here to compare it later
model_best = sky_model.copy()

print(result.models.to_parameters_table())


######################################################################
# Get the flux points
# ===================
#
# To get the observed flux points, just run the `~gammapy.estimators.FluxPointsEstimator`
# normally
#

energy_edges = dataset.counts.geom.axes["energy"].edges
fpe = FluxPointsEstimator(
    energy_edges=energy_edges, source="pks2155", selection_optional="all"
)
flux_points_obs = fpe.run(datasets=[dataset])


######################################################################
# To get the deabsorbed flux points (ie, intrinsic points), we simply need
# to set the reference model to the best fit power law instead of the
# compound model.
#

flux_points_intrinsic = flux_points_obs.copy(
    reference_model=SkyModel(spectral_model=pwl)
)

######################################################################
#
print(flux_points_obs.reference_model)

######################################################################
#
print(flux_points_intrinsic.reference_model)


######################################################################
# Plot the observed and intrinsic fluxes
# --------------------------------------
#

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
# sphinx_gallery_thumbnail_number = 2


######################################################################
# Use a custom EBL model
# ----------------------
#
# To use a custom EBL model, you have different options:
#
# You can save the optical depth as a function of energy and redshift
# in the format proposed by the XSPEC table models (see `here <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/summary/ogip_92_009_summary.html>`_).
# You can also check the model fits files in `$GAMMAPY_DATA/ebl/ <https://github.com/gammapy/gammapy-data/tree/main/ebl>`_ for examples.
#
# To read in a custom EBL model, use the `~gammapy.modeling.models.EBLAbsorptionNormSpectralModel.read()` method.

filename = "$GAMMAPY_DATA/ebl/ebl_dominguez11.fits.gz"
absorption_custom = EBLAbsorptionNormSpectralModel.read(filename, redshift=redshift)
print(absorption_custom)

######################################################################
# Or you can create `~gammapy.modeling.models.TemplateNDSpectralModel` from your data:
# To create your own models, you must have the optical depth tabulated as a function of gamma-ray energy and redshift.
# In this example, we create a toy model with 10 bins in energy (`energies_toymodel`) and 3 bins in redshift (`redshifts_toymodel`).

ebl_abs_dict_toymodel = {
    0.1: [
        0.98033076,
        0.89441662,
        0.63278539,
        0.34641156,
        0.22098954,
        0.12568629,
        0.024528508,
        3.6312218e-8,
        1.175492e-38,
        1.1754907e-38,
    ],
    0.3: [
        0.91439042,
        0.62265147,
        0.17481028,
        0.025712425,
        0.0064894828,
        0.00065273359,
        1.0083145e-7,
        1.1754939e-38,
        1.1754907e-38,
        1.1754907e-38,
    ],
    0.5: [
        0.81180396,
        0.35624504,
        0.032063942,
        0.0013128846,
        0.00010244388,
        6.1055654e-7,
        5.5373134e-19,
        1.1754907e-38,
        1.1754907e-38,
        1.1754907e-38,
    ],
}
redshifts_toymodel = [key for key in ebl_abs_dict_toymodel]
energies_toymodel = np.logspace(-1, 2, 10) * u.TeV


######################################################################
# Use energy and redshift as axes to define a `~gammapy.maps.RegionGeom`:

axes_energy_toymodel = MapAxis.from_nodes(
    energies_toymodel, name="energy_true", interp="log"
)
axes_redshift_toymodel = MapAxis.from_nodes(
    redshifts_toymodel, name="redshift", interp="linear"
)
geom_toymodel = RegionGeom(
    region=None, axes=[axes_energy_toymodel, axes_redshift_toymodel]
)


######################################################################
# Reshape your data and use it together with the above define geometry to create a
# `~gammapy.modeling.models.TemplateNDSpectralModel` from a `~gammapy.maps.RegionNDMap`:
data_reshaped_toymodel = np.array(
    [ebl_abs_dict_toymodel[key] for key in ebl_abs_dict_toymodel]
)
print(np.shape(data_reshaped_toymodel))

######################################################################
ndmap_toymodel = RegionNDMap(geom=geom_toymodel, data=data_reshaped_toymodel, unit="")
absorption_toymodel = TemplateNDSpectralModel(
    map=ndmap_toymodel,
    filename="$GAMMAPY_DATA/ebl/ebl_toymodel.fits.gz",
    interp_kwargs={"extrapolate": False},
)


######################################################################
# Set your redshift value and freeze it before using the model
absorption_toymodel.parameters["redshift"].value = redshift
absorption_toymodel.parameters["redshift"].frozen = True
print(absorption_toymodel)


######################################################################
# This `absorption_toymodel` can now be used in the same way as the `absorption` model defined above.

######################################################################
# To write and read your toymodel, you can use the `~gammapy.modeling.models.TemplateNDSpectralModel.write()`
# and read it again with `~gammapy.maps.RegionNDMap.read()` as a  `~gammapy.maps.RegionNDMap`, which can
# then again be assigned to a `~gammapy.modeling.models.TemplateNDSpectralModel` as shown above.
absorption_toymodel.write(overwrite=True)
ndmap_toymodel_read = RegionNDMap.read("$GAMMAPY_DATA/ebl/ebl_toymodel.fits.gz")


######################################################################
# Further extensions
# ------------------
#
# In this notebook, we have kept the parameters of the EBL model, the
# `alpha_norm` and the `redshift` frozen. Under reasonable assumptions
# on the intrinsic spectrum, it can be possible to constrain these
# parameters.
#
# Example: We now assume that the FermiLAT 4FGL catalog spectrum of the
# source is a good assumption of the intrinsic spectrum.
#
# *NOTE*: This is a very simplified assumption and in reality, EBL
# absorption can affect the Fermi spectrum significantly. Also, blazar
# spectra vary with time and long term averaged states may not be
# representative of a specific steady state
#

catalog = SourceCatalog4FGL()

src = catalog["PKS 2155-304"]

# Get the intrinsic model
intrinsic_model = src.spectral_model()
print(intrinsic_model)


######################################################################
# We add Gaussian priors on the `alpha` and `beta` parameters based on the 4FGL
# measurements and the associated errors. For more details on using priors, see
# :doc:`/tutorials/details/priors`
#

intrinsic_model.alpha.prior = GaussianPrior(
    mu=intrinsic_model.alpha.value, sigma=intrinsic_model.alpha.error
)
intrinsic_model.beta.prior = GaussianPrior(
    mu=intrinsic_model.beta.value, sigma=intrinsic_model.beta.error
)


######################################################################
# As before, multiply the intrinsic model with the EBL model
#

obs_model = intrinsic_model * absorption


######################################################################
# Now, free the redshift of the source
#

obs_model.parameters["redshift"].frozen = False

print(obs_model.parameters.to_table())

sky_model = SkyModel(spectral_model=obs_model, name="observed")
dataset.models = sky_model

result1 = fit.run([dataset])

print(result1.parameters.to_table())


######################################################################
# Get a fit stat profile for the redshift
# ---------------------------------------
#
# For more information about stat profiles, see
# :doc:`/tutorials/details/fitting`
#

total_stat = result1.total_stat

par = sky_model.parameters["redshift"]
par.scan_max = par.value + 5.0 * par.error
par.scan_min = max(0, par.value - 5.0 * par.error)
par.scan_n_values = 31

# %time
profile = fit.stat_profile(
    datasets=[dataset], parameter=sky_model.parameters["redshift"], reoptimize=True
)

plt.figure()
ax = plt.gca()
ax.plot(
    profile["observed.spectral.model2.redshift_scan"], profile["stat_scan"] - total_stat
)
ax.set_title("TS profile")
ax.set_xlabel("Redshift")
ax.set_ylabel("$\Delta$ TS")
plt.show()


######################################################################
# We see that the redshift is well constrained.
#
