.. _catalog:

*************************
catalog - Source catalogs
*************************

.. currentmodule:: gammapy.catalog

Introduction
============

`gammapy.catalog` provides convenient access to common gamma-ray source catalogs.

* ``hgps`` / `SourceCatalogHGPS` - H.E.S.S. Galactic plane survey (HGPS)
* ``gamma-cat`` /  `SourceCatalogGammaCat` - An open catalog of gamma-ray sources
* ``3fgl`` / `SourceCatalog3FGL` - LAT 4-year point source catalog
* ``4fgl`` / `SourceCatalog4FGL` - LAT 8-year point source catalog
* ``2fhl`` / `SourceCatalog2FHL` - LAT second high-energy source catalog
* ``3fhl`` / `SourceCatalog3FHL` - LAT third high-energy source catalog
* ``2hwc`` / `SourceCatalog2HWC` - 2HWC catalog from the HAWC observatory

For each catalog, a `SourceCatalog` class is provided to represent the catalog table,
and a matching `SourceCatalogObject` class to represent one catalog source and table row.

The main functionality provided is methods that map catalog information to
`~gammapy.modeling.models.SkyModel`, `~gammapy.modeling.models.SpectralModel`,
`~gammapy.modeling.models.SpatialModel`, `~gammapy.estimators.FluxPoints` and `~gammapy.estimators.LightCurve` objects.

`gammapy.catalog` is independent from the rest of Gammapy. The typical use cases
are to compare your results against previous results in the catalogs (e.g. overplot a spectral model),
or to create initial source models for certain energy bands and sky regions.

Using `gammapy.catalog`
=======================

Gammapy tutorial notebooks that show examples using ``gammapy.catalog``:

.. nbgallery::

   ../tutorials/api/catalog.ipynb
   ../tutorials/starting/overview.ipynb
   ../tutorials/analysis/1D/sed_fitting.ipynb


Reference/API
=============

.. automodapi:: gammapy.catalog
    :no-inheritance-diagram:
    :include-all-objects:
