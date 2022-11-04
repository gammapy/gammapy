"""
Modelling
=========

Multiple datasets and models interaction in Gammapy.

Aim
---

The main aim of this tutorial is to illustrate model management in
Gammapy, specially how to distribute multiple models across multiple
datasets. We also show some convenience functions built in gammapy for
handling multiple model components.

**Note: Since gammapy v0.18, the responsibility of model management is
left totally upon the user. All models, including background models,
have to be explicitly defined.** To keep track of the used models, we
define a global `Models` object (which is a collection of `SkyModel`
objects) to which we append and delete models.

Prerequisites
-------------

-  Knowledge of 3D analysis, dataset reduction and fitting see the :doc:`/tutorials/starting/analysis_2`
   tutorial.
-  Understanding of gammapy models, see the :doc:`/tutorials/api/models` tutorial.
-  Analysis of the Galactic Center with Fermi-LAT, shown in the  :doc:`/tutorials/data/fermi_lat` tutorial.
-  Analysis of the Galactic Center with CTA-DC1 , shown in the  :doc:`/tutorials/analysis-3d/analysis_3d` tutorial.

Proposed approach
-----------------

To show how datasets interact with models, we use two pre-computed
datasets on the galactic center, one from Fermi-LAT and the other from
simulated CTA (DC1) data. We demonstrate

-  Adding background models for each dataset
-  Sharing a model between multiple datasets

We then load models from the Fermi 3FHL catalog to show some convenience
handling for multiple `Models` together

-  accessing models from a catalog
-  selecting models contributing to a given region
-  adding and removing models
-  freezing and thawing multiple model parameters together
-  serialising models

For computational purposes, we do not perform any fitting in this
notebook.

Setup
-----

"""

from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion

# %matplotlib inline
import matplotlib.pyplot as plt
from gammapy.datasets import Datasets, MapDataset
from gammapy.maps import Map
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    PowerLawNormSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# Read the datasets
# -----------------
#
# First, we read some precomputed Fermi and CTA datasets, and create a
# `Datasets` object containing the two.
#

fermi_dataset = MapDataset.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc.fits.gz", name="fermi_dataset"
)
cta_dataset = MapDataset.read(
    "$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz", name="cta_dataset"
)
datasets = Datasets([fermi_dataset, cta_dataset])


######################################################################
# Plot the counts maps to see the region
#

plt.figure(figsize=(15, 5))
ax1 = plt.subplot(121, projection=fermi_dataset.counts.geom.wcs)
ax2 = plt.subplot(122, projection=cta_dataset.counts.geom.wcs)


datasets[0].counts.sum_over_axes().smooth(0.05 * u.deg).plot(
    ax=ax1, stretch="sqrt", add_cbar=True
)
datasets[1].counts.sum_over_axes().smooth(0.05 * u.deg).plot(
    ax=ax2, stretch="sqrt", add_cbar=True
)
ax1.set_title("Fermi counts")
ax2.set_title("CTA counts")

######################################################################
#

print(datasets.info_table(cumulative=False))

######################################################################
#

print(datasets)


######################################################################
# Note that while the datasets have an associated background map, they
# currently do not have any associated background model. This will be
# added in the following section
#


######################################################################
# Assigning background models to datasets
# ---------------------------------------
#
# For any IACT dataset (in this case `cta_dataset`) , we have to create
# a `FoVBackgroundModel`. Note that `FoVBackgroundModel` must be
# specified to one dataset only
#
# For Fermi-LAT, the background contribution is taken from a diffuse
# isotropic template. To convert this into a gammapy `SkyModel`, use the
# helper function `create_fermi_isotropic_diffuse_model()`
#
# To attach a model on a particular dataset it is necessary to specify the
# `datasets_names`. Otherwise, by default, the model will be applied to
# all the datasets in `datasets`
#


######################################################################
# First, we must create a global `Models` object which acts as the
# container for all models used in a particular analysis
#

models = Models()  # global models object

# Create the FoV background model for CTA data

bkg_model = FoVBackgroundModel(dataset_name=cta_dataset.name)
models.append(bkg_model)  # Add the bkg_model to models()

# Read the fermi isotropic diffuse background model

diffuse_iso = create_fermi_isotropic_diffuse_model(
    filename="$GAMMAPY_DATA/fermi_3fhl/iso_P8R2_SOURCE_V6_v06.txt",
)
diffuse_iso.datasets_names = fermi_dataset.name  # specifying the dataset name

models.append(diffuse_iso)  # Add the fermi_bkg_model to models()

# Now, add the models to datasets
datasets.models = models

# You can see that each dataset lists the correct associated models
print(datasets)


######################################################################
# Add a model on multiple datasets
# --------------------------------
#
# In this section, we show how to add a model to multiple datasets. For
# this, we specify a list of `datasets_names` to the model.
# Alternatively, not specifying any `datasets_names` will add it to all
# the datasets.
#
# For this example, we use a template model of the galactic diffuse
# emission to be shared between the two datasets.
#

# Create the diffuse model
diffuse_galactic_fermi = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz")

template_diffuse = TemplateSpatialModel(
    diffuse_galactic_fermi, normalize=False
)  # the template model in this case is already a full 3D model, it should not be normalised

diffuse_iem = SkyModel(
    spectral_model=PowerLawNormSpectralModel(),
    spatial_model=template_diffuse,
    name="diffuse-iem",
    datasets_names=[
        cta_dataset.name,
        fermi_dataset.name,
    ],  # specifying list of dataset names
)  # A power law spectral correction is applied in this case

# Now, add the diffuse model to the global models list
models.append(diffuse_iem)

# add it to the datasets, and inspect
datasets.models = models
print(datasets)


######################################################################
# The `diffuse-iem` model is correctly present on both. Now, you can
# proceed with the fit. For computational purposes, we skip it in this
# notebook
#

# %%time
# fit2 = Fit()
# result2 = fit2.run(datasets)
# print(result2.success)


######################################################################
# Loading models from a catalog
# -----------------------------
#
# We now load the Fermi 3FHL catalog and demonstrate some convenience
# functions. For more details on using Gammapy catalog, see the
# :doc:`/tutorials/api/catalog` tutorial.
#

from gammapy.catalog import SourceCatalog3FHL

catalog = SourceCatalog3FHL()


######################################################################
# We first choose some relevant models from the catalog and create a new
# `Models` object.
#

gc_sep = catalog.positions.separation(SkyCoord(0, 0, unit="deg", frame="galactic"))
models_3fhl = [_.sky_model() for k, _ in enumerate(catalog) if gc_sep[k].value < 8]
models_3fhl = Models(models_3fhl)

print(len(models_3fhl))


######################################################################
# Selecting models contributing to a given region
# -----------------------------------------------
#
# We now use `Models.select_region()` to get a subset of models
# contributing to a particular region. You can also use
# `Models.select_mask()` to get models lying inside the `True` region
# of a mask map\`
#

region = CircleSkyRegion(
    center=SkyCoord(0, 0, unit="deg", frame="galactic"), radius=3.0 * u.deg
)

models_selected = models_3fhl.select_region(region)
print(len(models_selected))


######################################################################
# We now want to assign `models_3fhl` to the Fermi dataset, and
# `models_selected` to both the CTA and Fermi datasets. For this, we
# explicitlty mention the `datasets_names` to the former, and leave it
# `None` (default) for the latter.
#

for model in models_3fhl:
    if model not in models_selected:
        model.datasets_names = fermi_dataset.name

# assign the models to datasets
datasets.models = models_3fhl


######################################################################
# To see the models on a particular dataset, you can simply see
#

print("Fermi dataset models: ", datasets[0].models.names)
print("\n CTA dataset models: ", datasets[1].models.names)


######################################################################
# Combining two `Models`
# ------------------------
#


######################################################################
# `Models` can be extended simply as as python lists
#

models.extend(models_selected)
print(len(models))


######################################################################
# Selecting models from a list
# ----------------------------
#
# A `Model` can be selected from a list of `Models` by specifying its
# index or its name.
#

model = models_3fhl[0]
print(model)

# Alternatively
model = models_3fhl["3FHL J1731.7-3003"]
print(model)


######################################################################
# `Models.select` can be used to select all models satisfying a list of
# conditions. To select all models applied on the cta_dataset with the
# characters `1748` in the name
#

models = models_3fhl.select(datasets_names=cta_dataset.name, name_substring="1748")
print(models)


######################################################################
# Note that `Models.select()` combines the different conditions with an
# `AND` operator. If one needs to combine conditions with a `OR`
# operator, the `Models.selection_mask()` method can generate a boolean
# array that can be used for selection. For ex:
#

selection_mask = models_3fhl.selection_mask(
    name_substring="1748"
) | models_3fhl.selection_mask(name_substring="1731")

models_OR = models_3fhl[selection_mask]
print(models_OR)


######################################################################
# Removing a model from a dataset
# -------------------------------
#


######################################################################
# Any addition or removal of a model must happen through the global models
# object, which must then be re-applied on the dataset(s). Note that
# operations **cannot** be directly performed on `dataset.models()`.
#

# cta_dataset.models.remove()
# * this is forbidden *

# Remove the model '3FHL J1744.5-2609'
models_3fhl.remove("3FHL J1744.5-2609")
len(models_3fhl)

# After any operation on models, it must be re-applied on the datasets
datasets.models = models_3fhl


######################################################################
# To see the models applied on a dataset, you can simply
#

print(datasets.models.names)


######################################################################
# Plotting models on a (counts) map
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The spatial regions of `Models` can be plotted on a given geom using
# `Models.plot_regions()`. You can also use `Models.plot_positions()`
# to plot the centers of each model.
#

plt.figure(figsize=(16, 5))
ax1 = plt.subplot(121, projection=fermi_dataset.counts.geom.wcs)
ax2 = plt.subplot(122, projection=cta_dataset.counts.geom.wcs)

for ax, dataset in zip([ax1, ax2], datasets):
    dataset.counts.sum_over_axes().smooth(0.05 * u.deg).plot(
        ax=ax, stretch="sqrt", add_cbar=True, cmap="afmhot"
    )
    dataset.models.plot_regions(ax=ax, color="white")
    ax.set_title(dataset.name)


######################################################################
# Freezing and unfreezing model parameters
# ----------------------------------------
#
# For a given model, any parameter can be (un)frozen individually.
# Additionally, `model.freeze` and `model.unfreeze` can be used to
# freeze and unfreeze all parameters in one go.
#

model = models_3fhl[0]
print(model)

# To freeze a single parameter
model.spectral_model.index.frozen = True
print(model)  # index is now frozen

# To unfreeze a parameter
model.spectral_model.index.frozen = False

# To freeze all parameters of a model
model.freeze()
print(model)

# To unfreeze all parameters (except parameters which must remain frozen)
model.unfreeze()
print(model)


######################################################################
# Only spectral or spatial or temporal components of a model can also be
# frozen
#

# To freeze spatial components
model.freeze("spatial")
print(model)


######################################################################
# To check if all the parameters of a model are frozen,
#

print(model.frozen)  # False because spectral components are not frozen

print(model.spatial_model.frozen)  # all spatial components are frozen


######################################################################
# The same operations can be performed on `Models` directly - to perform
# on a list of models at once, eg
#

models_selected.freeze()  # freeze all parameters of all models

models_selected.unfreeze()  # unfreeze all parameters of all models

# print the free parameters in the models
print(models_selected.parameters.free_parameters.names)


######################################################################
# There are more functionalities which you can explore. In general, using
# `help()` on any function is a quick and useful way to access the
# documentation. For ex, `Models.unfreeze_all` will unfreeze all
# parameters, even those which are fixed by default. To see its usage, you
# can simply type
#

help(models_selected.unfreeze)


######################################################################
# Serialising models
# ------------------
#


######################################################################
# `Models` can be (independently of `Datasets`) written to/ read from
# a disk as yaml files. Datasets are always serialised along with their
# associated models, ie, with yaml and fits files. eg:
#

# To save only the models
models_3fhl.write("3fhl_models.yaml", overwrite=True)

# To save datasets and models
datasets.write(
    filename="datasets-gc.yaml", filename_models="models_gc.yaml", overwrite=True
)

# To read only models
models = Models.read("3fhl_models.yaml")
print(models)

# To read datasets with models
datasets_read = Datasets.read("datasets-gc.yaml", filename_models="models_gc.yaml")
print(datasets_read)
