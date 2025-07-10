"""
Source catalogs
===============

Access and explore thew most common gamma-ray source catalogs.

Introduction
------------

`~gammapy.catalog` provides convenient access to common gamma-ray
source catalogs. This module is mostly independent of the rest of
Gammapy. Typically, you use it to compare new analyses against catalog
results, e.g. overplot the spectral model, or compare the source
position.

Moreover, as creating a source model and flux points for a given catalog
from the FITS table is tedious, `~gammapy.catalog` has this already
implemented. So you can create initial source models for your analyses.
This is very common for Fermi-LAT, to start with a catalog model. For
TeV analysis, especially in crowded Galactic regions, using the HGPS,
gamma-cat or 2HWC catalog in this way can also be useful.

In this tutorial you will learn how to:

-  List available catalogs
-  Load a catalog
-  Access the source catalog table data
-  Select a catalog subset or a single source
-  Get source spectral and spatial models
-  Get flux points (if available)
-  Get lightcurves (if available)
-  Access the source catalog table data
-  Pretty-print the source information

In this tutorial we will show examples using the following catalogs:

-  `~gammapy.catalog.SourceCatalogHGPS`
-  `~gammapy.catalog.SourceCatalogGammaCat`
-  `~gammapy.catalog.SourceCatalog3FHL`
-  `~gammapy.catalog.SourceCatalog4FGL`

All catalog and source classes work the same, as long as some
information is available. E.g. trying to access a lightcurve from a
catalog and source that does not have that information will return
`None`.

Further information is available at `~gammapy.catalog`.

"""

import numpy as np
import astropy.units as u

# %matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display
from gammapy.catalog import SourceCatalog4FGL
from gammapy.catalog import CATALOG_REGISTRY

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# List available catalogs
# -----------------------
#
# `~gammapy.catalog` contains a catalog registry `~gammapy.catalog.CATALOG_REGISTRY`,
# which maps catalog names (e.g. “3fhl”) to catalog classes
# (e.g. `~gammapy.catalog.SourceCatalog3FHL`).
#

print(CATALOG_REGISTRY)


######################################################################
# Load catalogs
# -------------
#
# If you have run ``gammapy download datasets`` or
# ``gammapy download tutorials``, you have a copy of the catalogs as FITS
# files in ``$GAMMAPY_DATA/catalogs``, and that is the default location
# where `~gammapy.catalog` loads from.
#

# # # !ls -1 $GAMMAPY_DATA/catalogs

# %%

# # # !ls -1 $GAMMAPY_DATA/catalogs/fermi


######################################################################
# So a catalog can be loaded directly from its corresponding class
#


catalog = SourceCatalog4FGL()
print("Number of sources :", len(catalog.table))


######################################################################
# Note that it loads the default catalog from `$GAMMAPY_DATA/catalogs`,
# you could pass a different ``filename`` when creating the catalog. For
# example here we load an older version of 4FGL catalog:
#

catalog = SourceCatalog4FGL("$GAMMAPY_DATA/catalogs/fermi/gll_psc_v20.fit.gz")
print("Number of sources :", len(catalog.table))


######################################################################
# Alternatively you can load a catalog by name via
# ``CATALOG_REGISTRY.get_cls(name)()`` (note the ``()`` to instantiate a
# catalog object from the catalog class - only this will load the catalog
# and be useful), or by importing the catalog class
# (e.g. `~gammapy.catalog.SourceCatalog3FGL`) directly. The two ways are equivalent, the
# result will be the same.
#

# FITS file is loaded
catalog = CATALOG_REGISTRY.get_cls("3fgl")()
print(catalog)

# %%
# Let's load the source catalogs we will use throughout this tutorial
catalog_gammacat = CATALOG_REGISTRY.get_cls("gamma-cat")()
catalog_3fhl = CATALOG_REGISTRY.get_cls("3fhl")()
catalog_4fgl = CATALOG_REGISTRY.get_cls("4fgl")()
catalog_hgps = CATALOG_REGISTRY.get_cls("hgps")()


######################################################################
# Catalog table
# -------------
#
# Source catalogs are given as ``FITS`` files that contain one or multiple
# tables.
#
# However, you can also access the underlying `astropy.table.Table` for
# a catalog, and the row data as a Python `dict`. This can be useful if
# you want to do something that is not pre-scripted by the
# `~gammapy.catalog.SourceCatalog` classes, such as e.g. selecting sources by sky
# position or association class, or accessing special source information.
#

print(type(catalog_3fhl.table))

# %%
print(len(catalog_3fhl.table))

# %%
display(catalog_3fhl.table[:3][["Source_Name", "RAJ2000", "DEJ2000"]])


######################################################################
# Note that the catalogs object include a helper property that gives
# directly the sources positions as a `~astropy.coordinates.SkyCoord` object (we will show an
# usage example in the following).
#

print(catalog_3fhl.positions[:3])


######################################################################
# Source object
# -------------
#
# Select a source
# ~~~~~~~~~~~~~~~
#
# The catalog entries for a single source are represented by a
# `~gammapy.catalog.SourceCatalogObject`. In order to select a source object index into
# the catalog using ``[]``, with a catalog table row index (zero-based,
# first row is ``[0]``), or a source name. If a name is given, catalog
# table columns with source names and association names (“ASSOC1” in the
# example below) are searched top to bottom. There is no name resolution
# web query.
#

source = catalog_4fgl[49]
print(source)

# %%
print(source.row_index, source.name)

# %%
source = catalog_4fgl["4FGL J0010.8-2154"]
print(source.row_index, source.name)

# %%
print(source.data["ASSOC1"])

# %%
source = catalog_4fgl["PKS 0008-222"]
print(source.row_index, source.name)


######################################################################
# Note that you can also do a ``for source in catalog`` loop, to find or
# process sources of interest.
#
# Source information
# ~~~~~~~~~~~~~~~~~~
#
# The source objects have a ``data`` property that contains the
# information of the catalog row corresponding to the source.
#

print(source.data["Npred"])

# %%
print(source.data["GLON"], source.data["GLAT"])


######################################################################
# As for the catalog object, the source object has a ``position``
# property.
#

print(source.position.galactic)


######################################################################
# Select a catalog subset
# -----------------------
#
# The catalog objects support selection using boolean arrays (of the same
# length), so one can create a new catalog as a subset of the main catalog
# that verify a set of conditions.
#
# In the next example we select only few of the brightest sources
# in the 100 to 200 GeV energy band.
#

mask_bright = np.zeros(len(catalog_3fhl.table), dtype=bool)
for k, source in enumerate(catalog_3fhl):
    flux = source.spectral_model().integral(100 * u.GeV, 200 * u.GeV).to("cm-2 s-1")
    if flux > 1e-10 * u.Unit("cm-2 s-1"):
        mask_bright[k] = True
        print(f"{source.row_index:<7d} {source.name:20s} {flux:.3g}")

# %%
catalog_3fhl_bright = catalog_3fhl[mask_bright]
print(catalog_3fhl_bright)

# %%
print(catalog_3fhl_bright.table["Source_Name"])


######################################################################
# Similarly we can select only sources within a region of interest. Here
# for example we use the ``position`` property of the catalog object to
# select sources within 5 degrees from “PKS 0008-222”:
#

source = catalog_4fgl["PKS 0008-222"]
mask_roi = source.position.separation(catalog_4fgl.positions) < 5 * u.deg

catalog_4fgl_roi = catalog_4fgl[mask_roi]
print("Number of sources :", len(catalog_4fgl_roi.table))


######################################################################
# Source models
# -------------
#
# The `~gammapy.catalog.SourceCatalogObject` classes have a
# `~gammapy.catalog.SourceCatalogObject.sky_model()` model which creates a
# `~gammapy.modeling.models.SkyModel` object, with model parameter values
# and parameter errors from the catalog filled in.
#
# In most cases, the `~gammapy.catalog.SourceCatalogObject.spectral_model()` method provides the
# `~gammapy.modeling.models.SpectralModel` part of the sky model, and the
# `~gammapy.catalog.SourceCatalogObject.spatial_model()` method the `~gammapy.modeling.models.SpatialModel`
# part individually.
#
# We use the `~gammapy.catalog.SourceCatalog3FHL` for the examples in
# this section.
#

source = catalog_4fgl["PKS 2155-304"]

model = source.sky_model()
print(model)

# %%
print(model)

# %%
print(model.spatial_model)

# %%
print(model.spectral_model)

# %%
energy_bounds = (100 * u.MeV, 100 * u.GeV)
opts = dict(sed_type="e2dnde", yunits=u.Unit("TeV cm-2 s-1"))
model.spectral_model.plot(energy_bounds, **opts)
model.spectral_model.plot_error(energy_bounds, **opts)
plt.show()


######################################################################
# You can create initial source models for your analyses using the
# `~gammapy.catalog.SourceCatalog.to_models()` method of the catalog objects. Here for example we
# create a `~gammapy.modeling.models.Models` object from the 4FGL catalog subset we previously
# defined:
#

models_4fgl_roi = catalog_4fgl_roi.to_models()
print(models_4fgl_roi)


######################################################################
# Specificities of the HGPS catalog
# ---------------------------------
#
# Using the `~gammapy.catalog.SourceCatalog.to_models()` method for the
# `~gammapy.catalog.SourceCatalogHGPS` will return only the models
# components of the sources retained in the main catalog, several
# candidate objects appears only in the Gaussian components table (see
# section 4.9 of the HGPS paper, https://arxiv.org/abs/1804.02432). To
# access these components you can do the following:
#

discarded_ind = np.where(
    ["Discarded" in _ for _ in catalog_hgps.table_components["Component_Class"]]
)[0]
discarded_table = catalog_hgps.table_components[discarded_ind]


######################################################################
# There is no spectral model available for these components but you can
# access their spatial models:
#

discarded_spatial = [
    catalog_hgps.gaussian_component(idx).spatial_model() for idx in discarded_ind
]


######################################################################
# In addition to the source components the HGPS catalog include a large
# scale diffuse component built by fitting a gaussian model in a sliding
# window along the Galactic plane. Information on this model can be
# accessed via the properties `~gammapy.catalog.SourceCatalogHGPS.table_large_scale_component` and
# `~gammapy.catalog.SourceCatalogHGPS.large_scale_component` of `~gammapy.catalog.SourceCatalogHGPS`.
#

# here we show the 5 first elements of the table
display(catalog_hgps.table_large_scale_component[:5])
# you can also try :
# help(catalog_hgps.large_scale_component)


######################################################################
# Flux points
# -----------
#
# The flux points are available via the ``flux_points`` property as a
# `~gammapy.estimators.FluxPoints` object.
#

source = catalog_4fgl["PKS 2155-304"]
flux_points = source.flux_points


print(flux_points)

# %%
display(flux_points.to_table(sed_type="flux"))

# %%
flux_points.plot(sed_type="e2dnde")
plt.show()


######################################################################
# Lightcurves
# -----------
#
# The Fermi catalogs contain lightcurves for each source. It is available
# via the ``source.lightcurve`` method as a
# `~gammapy.estimators.FluxPoints` object with a time axis.
#

lightcurve = catalog_4fgl["4FGL J0349.8-2103"].lightcurve()

print(lightcurve)

# %%
display(lightcurve.to_table(format="lightcurve", sed_type="flux"))

# %%
plt.figure(figsize=(8, 6))
plt.subplots_adjust(bottom=0.2, left=0.2)
lightcurve.plot()
plt.show()


######################################################################
# Pretty-print source information
# -------------------------------
#
# A source object has a nice string representation that you can print.
#

source = catalog_hgps["MSH 15-52"]
print(source)


######################################################################
# You can also call ``source.info()`` instead and pass as an option what
# information to print. The options available depend on the catalog, you
# can learn about them using ``help()``
#

help(source.info)

# %%
print(source.info("associations"))
