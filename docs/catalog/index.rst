.. _catalog:

*************************
catalog - Source catalogs
*************************

.. currentmodule:: gammapy.catalog

Introduction
============

`gammapy.catalog` provides utilities to work with source catalogs in general,
and catalogs relevant for gamma-ray astronomy specifically.

If you just want to browse catalog information, you can visit
http://gamma-sky.net .

Available catalogs
------------------

Support for the following catalogs is available::

    Source catalog registry:
       name                       description                      sources
    --------- ---------------------------------------------------- -------
         hgps H.E.S.S. Galactic plane survey (HGPS) source catalog      78
    gamma-cat                 An open catalog of gamma-ray sources     166
         3fgl                      LAT 4-year point source catalog    3034
         1fhl      First Fermi-LAT Catalog of Sources above 10 GeV     514
         2fhl                LAT second high-energy source catalog     360
         3fhl                 LAT third high-energy source catalog    1556
         2hwc               2HWC catalog from the HAWC observatory      40

More catalogs can be added to ``gammapy.catalog``, and users can also add
support for their favourite catalog in their Python script or package, by
following the examples how the built-in catalogs are implemented.

How it works
------------

This section provides some information how ``gammapy.catalog`` works. In
principle, to use it, you don't have to know how it's implemented, and just
follow the examples in the sections below. In practice, if you want to work with
catalogs via ``gammapy.catalog`` or directly, it really helps if you spend a
little time and learn about Astropy tables.

Catalog data is stored as an `astropy.table.Table` object, with the information
for each source in a `astropy.table.Row`. In ``gammapy.catalog`` we have
implemented a base class `gammapy.catalog.SourceCatalog` that stores the
`astropy.table.Table` in the ``table`` attribute, and a base
`gammapy.catalog.SourceCatalogObject` class that can extract the
`astropy.table.Row` data into a Python dictionary in the ``data`` attribute and
then provides conveniences to work with the data for a given source.

For every given concrete catalog, two classes ("catalog" and "source/object")
are needed. E.g. for the Fermi-LAT 3FGL catalog, there are the
`gammapy.catalog.SourceCatalog3FGL` and the
`gammapy.catalog.SourceCatalogObject3FGL` classes.

The ``SourceCatalog`` class mostly handles data file loading, as well as source
access by integer row index or source name. The ``SourceCatalogObject`` class
implements in ``__str__`` a pretty-printed version of ``source.data``, so that
you can ``print(source)``, as well as factory methods to create Gammapy objects
such as `gammapy.modeling.models.SpectralModel` or
`gammapy.modeling.models.SpatialModel` or `gammapy.spectrum.FluxPoints`
representing spatial and spectral models, or spectral points, which you can then
print or plot or use for simulation and analysis.

Getting Started
===============

Let's start by checking which catalogs are available::

    >>> from gammapy.catalog import source_catalogs
    >>> source_catalogs.info()
    Source catalog registry:
    Name Description Loaded
    ---- ----------- ------
    3fgl description     no
    2fhl description     no

You can access ``source_catalogs`` like a dict, i.e. load catalogs by name::

    >>> catalog = source_catalogs['3fgl']
    >>> catalog.table # To access the underlying astropy.table.Table

Note that importing ``source_catalogs`` did not load catalogs from disk, they
are lazy-loaded on access via ``[name]`` and then cached for the duration of
your Python session.

You can get an object representing one source of interest by source name or by
row index::

    >>> source = catalog['3FGL J0004.7-4740']  # access by source name
    >>> source = catalog[15]  # access by row index

The ``source`` object contains all of the information in the ``data``
attribute::

    >>> source.data['RAJ2000']
    1.1806999
    >>> source.data['CLASS1']
    'fsrq '
    >>> source.pprint()
    # print all info on this source in a readable format

TODO: continue here describing how to access spectra, finder charts, ... once
that's implemented.

Using `gammapy.catalog`
=======================

For more advanced use cases please go to the tutorial notebooks:

* :gp-notebook:`hgps`
* :gp-notebook:`first_steps`
* :gp-notebook:`sed_fitting_gammacat_fermi`

The following pages describe ``gammapy.catalog`` in more detail:

.. toctree::
    :maxdepth: 1

    gammacat

Reference/API
=============

.. automodapi:: gammapy.catalog
    :no-inheritance-diagram:
    :include-all-objects:
