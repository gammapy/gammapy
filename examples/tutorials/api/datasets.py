"""
Datasets - Reduced data, IRFs, models
=====================================

Learn how to work with datasets

Introduction
------------

``gammapy.datasets`` are a crucial part of the gammapy API. ``datasets``
constitute ``DL4`` data - binned counts, IRFs, models and the associated
likelihoods. ``Datasets`` from the end product of the ``makers`` stage,
see `makers notebook <makers.ipynb>`__, and are passed on to the ``Fit``
or estimator classes for modelling and fitting purposes.

To find the different types of ``Dataset`` that are supported see
`Datasets
home <../../user-guide/datasets/index.rst#Types-of-supported-datasets>`__

Setup
-----

"""

import numpy as np
import astropy.units as u
from astropy.time import Time
from regions import CircleSkyRegion
from astropy.coordinates import SkyCoord
from gammapy.datasets import (
    MapDataset,
    SpectrumDataset,
    SpectrumDatasetOnOff,
    Datasets,
    FluxPointsDataset,
)
from gammapy.data import DataStore, GTI
from gammapy.maps import WcsGeom, RegionGeom, MapAxes, MapAxis, Map
from gammapy.modeling.models import (
    SkyModel,
    PowerLawSpectralModel,
    FoVBackgroundModel,
)
from gammapy.estimators import FluxPoints
from gammapy.utils.scripts import make_path

# %matplotlib inline


######################################################################
# MapDataset
# ----------
# 
# The counts, exposure, background, masks, and IRF maps are bundled
# together in a data structure named ``MapDataset``. While the ``counts``,
# and ``background`` maps are binned in reconstructed energy and must have
# the same geometry, the IRF maps can have a different spatial (coarsely
# binned and larger) geometry and spectral range (binned in true
# energies). It is usually recommended that the true energy bin should be
# larger and more finely sampled and the reco energy bin.
# 
# Creating an empty dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# An empty ``MapDataset`` can be directly instantiated from any
# ``WcsGeom`` object:
# 

energy_axis = MapAxis.from_energy_bounds(
    1, 10, nbin=11, name="energy", unit="TeV"
)

geom = WcsGeom.create(
    skydir=(83.63, 22.01),
    axes=[energy_axis],
    width=5 * u.deg,
    binsz=0.05 * u.deg,
    frame="icrs",
)

dataset_empty = MapDataset.create(geom=geom, name="my-dataset")


######################################################################
# It is good practice to define a name for the dataset, such that you can
# identify it later by name. However if you define a name it **must** be
# unique. Now we can already print the dataset:
# 

print(dataset_empty)


######################################################################
# The printout shows the key summary information of the dataset, such as
# total counts, fit statistics, model information etc.
# 
# ``MapDataset.from_geom`` has additional keywords, that can be used to
# define the binning of the IRF related maps:
# 

# choose a different true energy binning for the exposure, psf and edisp
energy_axis_true = MapAxis.from_energy_bounds(
    0.1, 100, nbin=11, name="energy_true", unit="TeV", per_decade=True
)

# choose a different rad axis binning for the psf
rad_axis = MapAxis.from_bounds(0, 5, nbin=50, unit="deg", name="rad")

gti = GTI.create(0 * u.s, 1000 * u.s)

dataset_empty = MapDataset.create(
    geom=geom,
    energy_axis_true=energy_axis_true,
    rad_axis=rad_axis,
    binsz_irf=0.1,
    gti=gti,
    name="dataset-empty",
)


######################################################################
# To see the geometry of each map, we can use:
# 

dataset_empty.geoms


######################################################################
# Another way to create a ``MapDataset`` is to just read an existing one
# from a FITS file:
# 

dataset_cta = MapDataset.read(
    "$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz", name="dataset-cta"
)

print(dataset_cta)


######################################################################
# Accessing contents of a dataset
# -------------------------------
# 


######################################################################
# To further explore the contents of a ``Dataset``, you can use
# e.g. ``.info_dict()``
# 

# For a quick info, use
dataset_cta.info_dict()

# For a quick view, use
dataset_cta.peek()


######################################################################
# And access individual maps like:
# 

counts_image = dataset_cta.counts.sum_over_axes()
counts_image.smooth("0.1 deg").plot()


######################################################################
# Of course you can also access IRF related maps, e.g. the psf as
# ``PSFMap``:
# 

dataset_cta.psf


######################################################################
# And use any method on the ``PSFMap`` object:
# 

dataset_cta.psf.plot_containment_radius_vs_energy()

edisp_kernel = dataset_cta.edisp.get_edisp_kernel()
edisp_kernel.plot_matrix()


######################################################################
# The ``MapDataset`` typically also contains the information on the
# residual hadronic background, stored in ``MapDataset.background`` as a
# map:
# 

dataset_cta.background


######################################################################
# As a next step we define a minimal model on the dataset using the
# ``.models`` setter:
# 

model = SkyModel.create("pl", "point", name="gc")
model.spatial_model.position = SkyCoord("0d", "0d", frame="galactic")
model_bkg = FoVBackgroundModel(dataset_name="dataset-cta")

dataset_cta.models = [model, model_bkg]


######################################################################
# Assigning models to datasets is covered in more detail in the `Modeling
# notebook <model_management.ipynb>`__. Printing the dataset will now show
# the mode components:
# 

print(dataset_cta)


######################################################################
# Now we can use ``.npred()`` to get a map of the total predicted counts
# of the model:
# 

npred = dataset_cta.npred()
npred.sum_over_axes().plot()


######################################################################
# To get the predicted counts from an individual model component we can
# use:
# 

npred_source = dataset_cta.npred_signal(model_name="gc")
npred_source.sum_over_axes().plot()


######################################################################
# ``MapDataset.background`` contains the background map computed from the
# IRF. Internally it will be combined with a ``FoVBackgroundModel``, to
# allow for adjusting the backgroun model during a fit. To get the model
# corrected background, one can use ``dataset.npred_background()``.
# 

npred_background = dataset_cta.npred_background()
npred_background.sum_over_axes().plot()


######################################################################
# Using masks
# ~~~~~~~~~~~
# 
# There are two masks that can be set on a ``Dataset``, ``mask_safe`` and
# ``mask_fit``.
# 
# -  The ``mask_safe`` is computed during the data reduction process
#    according to the specified selection cuts, and should not be changed
#    by the user.
# -  During modelling and fitting, the user might want to additionally
#    ignore some parts of a reduced dataset, e.g. to restrict the fit to a
#    specific energy range or to ignore parts of the region of interest.
#    This should be done by applying the ``mask_fit``. To see details of
#    applying masks, please refer to
#    `Masks-for-fitting <mask_maps.ipynb#Masks-for-fitting:-mask_fit>`__
# 
# Both the ``mask_fit`` and ``mask_safe`` must have the safe ``geom`` as
# the ``counts`` and ``background`` maps.
# 

# eg: to see the safe data range
dataset_cta.mask_safe.plot_grid();


######################################################################
# In addition it is possible to define a custom ``mask_fit``:
# 

# To apply a mask fit - in enegy and space
region = CircleSkyRegion(SkyCoord("0d", "0d", frame="galactic"), 1.5 * u.deg)

geom = dataset_cta.counts.geom

mask_space = geom.region_mask([region])
mask_energy = geom.energy_mask(0.3 * u.TeV, 8 * u.TeV)
dataset_cta.mask_fit = mask_space & mask_energy
dataset_cta.mask_fit.plot_grid(vmin=0, vmax=1, add_cbar=True);


######################################################################
# To access the energy range defined by the mask you can use: -
# ``dataset.energy_range_safe`` : energy range definedby the ``mask_safe``
# - ``dataset.energy_range_fit`` : energy range defined by the
# ``mask_fit`` - ``dataset.energy_range`` : the final energy range used in
# likelihood computation
# 
# These methods return two maps, with the ``min`` and ``max`` energy
# values at each spatial pixel
# 

e_min, e_max = dataset_cta.energy_range

# To see the lower energy threshold at each point
e_min.plot(add_cbar=True)

# To see the lower energy threshold at each point
e_max.plot(add_cbar=True)


######################################################################
# Just as for ``Map`` objects it is possible to cutout a whole
# ``MapDataset``, which will perform the cutout for all maps in
# parallel.Optionally one can provide a new name to the resulting dataset:
# 

cutout = dataset_cta.cutout(
    position=SkyCoord("0d", "0d", frame="galactic"),
    width=2 * u.deg,
    name="cta-cutout",
)

cutout.counts.sum_over_axes().plot()


######################################################################
# It is also possible to slice a ``MapDataset`` in energy:
# 

sliced = dataset_cta.slice_by_energy(
    energy_min=1 * u.TeV, energy_max=5 * u.TeV, name="slice-energy"
)
sliced.counts.plot_grid();


######################################################################
# The same operation will be applied to all other maps contained in the
# datasets such as ``mask_fit``:
# 

sliced.mask_fit.plot_grid();


######################################################################
# Resampling datasets
# ~~~~~~~~~~~~~~~~~~~
# 
# It can often be useful to coarsely rebin an initially computed datasets
# by a specified factor. This can be done in either spatial or energy
# axes:
# 

downsampled = dataset_cta.downsample(factor=8)
downsampled.counts.sum_over_axes().plot()


######################################################################
# And the same downsampling process is possible along the energy axis:
# 

downsampled_energy = dataset_cta.downsample(
    factor=5, axis_name="energy", name="downsampled-energy"
)
downsampled_energy.counts.plot_grid();


######################################################################
# In the printout one can see that the actual number of counts is
# preserved during the downsampling:
# 

print(downsampled_energy, dataset_cta)


######################################################################
# We can also resample the finer binned datasets to an arbitrary coarser
# energy binning using:
# 

energy_axis_new = MapAxis.from_energy_edges([0.1, 0.3, 1, 3, 10] * u.TeV)
resampled = dataset_cta.resample_energy_axis(energy_axis=energy_axis_new)
resampled.counts.plot_grid(ncols=2);


######################################################################
# To squash the whole dataset into a single energy bin there is the
# ``.to_image()`` convenience method:
# 

dataset_image = dataset_cta.to_image()
dataset_image.counts.plot()


######################################################################
# SpectrumDataset
# ---------------
# 
# ``SpectrumDataset`` inherits from a ``MapDataset``, and is specially
# adapted for 1D spectral analysis, and uses a ``RegionGeom`` instead of a
# ``WcsGeom``. A ``MapDatset`` can be converted to a ``SpectrumDataset``,
# by summing the ``counts`` and ``background`` inside the ``on_region``,
# which can then be used for classical spectral analysis. Containment
# correction is feasible only for circular regions.
# 

region = CircleSkyRegion(
    SkyCoord(0, 0, unit="deg", frame="galactic"), 0.5 * u.deg
)
spectrum_dataset = dataset_cta.to_spectrum_dataset(
    region, containment_correction=True, name="spectrum-dataset"
)

# For a quick look
spectrum_dataset.peek();


######################################################################
# A ``MapDataset`` can also be integrated over the ``on_region`` to create
# a ``MapDataset`` with a ``RegionGeom``. Complex regions can be handled
# and since the full IRFs are used, containment correction is not
# required.
# 

reg_dataset = dataset_cta.to_region_map_dataset(
    region, name="region-map-dataset"
)
print(reg_dataset)


######################################################################
# FluxPointsDataset
# -----------------
# 
# ``FluxPointsDataset`` is a ``Dataset`` container for precomputed flux
# points, which can be then used in fitting. ``FluxPointsDataset`` cannot
# be read directly, but should be read through ``FluxPoints``, with an
# additional ``SkyModel``. Similarly, ``FluxPointsDataset.write`` only
# saves the ``data`` component to disk.
# 

flux_points = FluxPoints.read(
    "$GAMMAPY_DATA/tests/spectrum/flux_points/diff_flux_points.fits"
)
model = SkyModel(spectral_model=PowerLawSpectralModel(index=2.3))
fp_dataset = FluxPointsDataset(data=flux_points, models=model)

fp_dataset.plot_spectrum()


######################################################################
# The masks on ``FluxPointsDataset`` are ``np.array`` and the data is a
# ``FluxPoints`` object. The ``mask_safe``, by default, masks the upper
# limit points
# 

fp_dataset.mask_safe  # Note: the mask here is simply a numpy array

fp_dataset.data  # is a `FluxPoints` object

fp_dataset.data_shape()  # number of data points


######################################################################
# For an example of fitting ``FluxPoints``, see `flux point
# fitting <../analysis/1D/sed_fitting.ipynb>`__, and can be used for
# catalog objects, eg see `catalog notebook <catalog.ipynb>`__
# 


######################################################################
# Datasets
# --------
# 


######################################################################
# ``Datasets`` are a collection of ``Dataset`` objects. They can be of the
# same type, or of different types, eg: mix of ``FluxPointDataset``,
# ``MapDataset`` and ``SpectrumDataset``.
# 
# For modelling and fitting of a list of ``Dataset`` objects, you can
# either - Do a joint fitting of all the datasets together - Stack the
# datasets together, and then fit them.
# 
# ``Datasets`` is a convenient tool to handle joint fitting of
# simultaneous datasets. As an example, please see the `joint fitting
# tutorial <../analysis/3D/analysis_mwl.ipynb>`__
# 
# To see how stacking is performed, please see `Implementation of
# stacking <../../user-guide/datasets/index.rst#stacking-multiple-datasets>`__
# 
# To create a ``Datasets`` object, pass a list of ``Dataset`` on init, eg
# 

datasets = Datasets([dataset_empty, dataset_cta])

print(datasets)


######################################################################
# If all the datasets have the same type we can also print an info table,
# collectiong all the information from the individual casll to
# ``Dataset.info_dict()``:
# 

datasets.info_table()  # quick info of all datasets

datasets.names  # unique name of each dataset


######################################################################
# We can access individual datasets in ``Datasets`` object by name:
# 

datasets["dataset-empty"]  # extracts the first dataset


######################################################################
# Or by index:
# 

datasets[0]


######################################################################
# Other list type operations work as well such as:
# 

# Use python list convention to remove/add datasets, eg:
datasets.remove("dataset-empty")
datasets.names


######################################################################
# Or
# 

datasets.append(spectrum_dataset)
datasets.names


######################################################################
# Let’s create a list of spectrum datasets to illustrate some more
# functionality:
# 

datasets = Datasets()

path = make_path("$GAMMAPY_DATA/joint-crab/spectra/hess")

for filename in path.glob("pha_*.fits"):
    dataset = SpectrumDatasetOnOff.read(filename)
    datasets.append(dataset)

print(datasets)


######################################################################
# Now we can stack all datasets using ``.stack_reduce()``:
# 

stacked = datasets.stack_reduce(name="stacked")
print(stacked)


######################################################################
# Or slice all datasets by a given energy range:
# 

datasets_sliced = datasets.slice_by_energy(
    energy_min="1 TeV", energy_max="10 TeV"
)
print(datasets_sliced.energy_ranges)