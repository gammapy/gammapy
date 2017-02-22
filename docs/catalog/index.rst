.. _catalog:

***********************************************
Source catalogs and objects (`gammapy.catalog`)
***********************************************

.. currentmodule:: gammapy.catalog

Introduction
============

`gammapy.catalog` provides utilities to work with source catalogs in general,
and catalogs relevant for gamma-ray astronomy specifically.

A tutorial introduction is available here: :gp-extra-notebook:`source_catalogs`.

Available catalogs
------------------

Support for the following catalogs is available::

       Name                       Description                      Sources
    --------- ---------------------------------------------------- -------
         3fgl                      LAT 4-year point source catalog    3034
         1fhl      First Fermi-LAT Catalog of Sources above 10 GeV     514
         2fhl                LAT second high-energy source catalog     360
         3fhl                 LAT third high-energy source catalog    1558
         hgps H.E.S.S. Galactic plane survey (HGPS) source catalog      78
    gamma-cat                 An open catalog of gamma-ray sources     162

More catalogs can be added to ``gammapy.catalog``, and users can also add
support for their favourite catalog in their Python script or package,
by following the examples how the built-in catalogs are implemented.

How it works
------------

This section provides some information how ``gammapy.catalog`` works. In principle,
to use it, you don't have to know how it's implemented, and just follow the examples
in the sections below. In practice, if you want to work with catalogs via ``gammapy.catalog``
or directly, it really helps if you spend a little time and learn about Astropy tables.

Catalog data is stored as an `astropy.table.Table` object, with the information for
each source in a `astropy.table.Row`. In ``gammapy.catalog`` we have implemented
a base class `gammapy.catalog.SourceCatalog` that stores the `astropy.table.Table`
in the ``table`` attribute, and a base `gammapy.catalog.SourceCatalogObject` class
that can extract the `astropy.table.Row` data into a Python dictionary in the ``data`` attribute
and then provides conveniences to work with the data for a given source.

For every given concrete catalog, two classes ("catalog" and "source/object") are needed.
E.g. for the Fermi-LAT 3FGL catalog, there are the `gammapy.catalog.SourceCatalog3FGL`
and the `gammapy.catalog.SourceCatalogObject3FGL` classes.

The ``SourceCatalog`` class mostly handles data file loading, as well as source access by integer
row index or source name. The ``SourceCatalogObject`` class implements in ``__str__`` a
pretty-printed version of ``source.data``, so that you can ``print(source)``, as well as
factory methods to create Gammapy objects such as `gammapy.spectrum.models.SpectralModel`
or ``gammapy.image.models.SpatialModel`` or `gammapy.spectrum.FluxPoints` representing
spatial and spectral models, or spectral points, which you can then print or plot
or use for simulation and analysis.

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

Note that importing ``source_catalogs`` did not load catalogs from disk,
they are lazy-loaded on access via ``[name]`` and then cached for the duration
of your Python session.

You can get an object representing one source of interest by source name or by row index::

    >>> source = catalog['3FGL J0004.7-4740']  # access by source name
    >>> source = catalog[15]  # access by row index

The ``source`` object contains all of the information in the ``data`` attribute::

    >>> source.data['RAJ2000']
    1.1806999
    >>> source.data['CLASS1']
    'fsrq '
    >>> source.pprint()
    # print all info on this source in a readable format

TODO: continue here describing how to access spectra, finder charts, ...
once that's implemented.


Catalog analysis
----------------

TODO: explain about the catalog "analysis" classes and functions (see API docs below)

TODO: give one example, e.g. how to reproduce a log(N)-log(S) plot from a Fermi catalog paper.

Command line tool
=================

Sometimes you just want to look up the information for a give source, and it's a little
inconvenient to have to start ``python`` and type the imports to access the info via ``gammapy.catalog``.

In this case we recommend you go to http://gamma-sky.net/cat ,
a website we are building for this use case (still very preliminary and incomplete).

Another option is to use the command line tool ``gammapy-catalog-query``::

    $ gammapy-catalog-query --help
    Usage: gammapy-catalog-query [OPTIONS] COMMAND [ARGS]...

      Gammapy catalog query command line tool.

      Examples
      --------

      gammapy-catalog-query -h
      gammapy-catalog-query catalogs
      gammapy-catalog-query sources 2fhl
      gammapy-catalog-query info 2fhl "2FHL J0534.5+2201"
      gammapy-catalog-query info 3fgl "3FGL J0534.5+2201"
      gammapy-catalog-query info hgps "HESS J1825-137"

      gammapy-catalog-query table-info 2fhl
      gammapy-catalog-query table-web 2fhl

    Options:
      -h, --help  Show this message and exit.

    Commands:
      catalogs         List available catalogs
      info             Print info for CATALOG and SOURCE
      plot-lightcurve  Plot lightcurve for CATALOG and SOURCE
      plot-spectrum    Plot spectrum for CATALOG and SOURCE
      sources          List sources for CATALOG
      table-info       Summarise table info for CATALOG
      table-web        Open table in web browser for CATALOG

We also started to implement a local web app: ``gammapy-catalog-browse``.
It isn't working well at the moment, and probably now that we started http://gamma-sky.net we'll probably remove it.
But if anyone is interested to fix and improve ``gammapy-catalog-browse``, we could also keep it.


Reference/API
=============

.. automodapi:: gammapy.catalog
    :no-inheritance-diagram:
