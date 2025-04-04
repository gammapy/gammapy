.. _catalog:

Source catalogs
===============

`gammapy.catalog` provides convenient access to common gamma-ray source catalogs.

* ``hgps`` / `~gammapy.catalog.SourceCatalogHGPS` - H.E.S.S. Galactic plane survey (HGPS)
* ``gamma-cat`` /  `~gammapy.catalog.SourceCatalogGammaCat` - An open catalog of gamma-ray sources
* ``3fgl`` / `~gammapy.catalog.SourceCatalog3FGL` - LAT 4-year point source catalog
* ``4fgl`` / `~gammapy.catalog.SourceCatalog4FGL` - LAT 8-year point source catalog
* ``2fhl`` / `~gammapy.catalog.SourceCatalog2FHL` - LAT second high-energy source catalog
* ``3fhl`` / `~gammapy.catalog.SourceCatalog3FHL` - LAT third high-energy source catalog
* ``2hwc`` / `~gammapy.catalog.SourceCatalog2HWC` - 2HWC catalog from the HAWC observatory
* ``3hwc`` / `~gammapy.catalog.SourceCatalog3HWC` - 3HWC catalog from the HAWC observatory

For each catalog, a `~gammapy.catalog.SourceCatalog` class is provided to represent the catalog table,
and a matching `~gammapy.catalog.SourceCatalogObject` class to represent one catalog source and table row.

The main functionality provided is methods that map catalog information to
`~gammapy.modeling.models.SkyModel`, `~gammapy.modeling.models.SpectralModel`,
`~gammapy.modeling.models.SpatialModel`, `~gammapy.estimators.FluxPoints` and `~gammapy.estimators.LightCurve` objects.

`gammapy.catalog` is independent from the rest of Gammapy. The typical use cases
are to compare your results against previous results in the catalogs (e.g. overplot a spectral model),
or to create initial source models for certain energy bands and sky regions.


Using gammapy.catalog
---------------------

.. minigallery::
    :add-heading: Examples using `~gammapy.catalog`

    ../examples/tutorials/tutorials/api/catalog.py
    ../examples/tutorials/starting/overview.py
    ../examples/tutorials/analysis-1d/sed_fitting
    ../examples/tutorials/api/model_management.py

