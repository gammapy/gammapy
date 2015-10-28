.. _catalog:

***************************
Catalog (`gammapy.catalog`)
***************************

.. currentmodule:: gammapy.catalog

Introduction
============

`gammapy.catalog` provides utilities to work with source catalogs in general,
and catalogs relevant for gamma-ray astronomy specifically.

TODO: auto-generate that list of built-in catalogs during the Sphinx build.

The `gammapy.catalog.SourceCatalog` class is a thin wrapper around
`astropy.table.Table` (stored as a ``table`` attribute) that adds
source lookup by name or row index, returning a `gammapy.catalog.SourceCatalogObject`
object that contains the `astropy.table.Row` information in a ``data``
attribute of type `collections.OrderedDict`, but unlike Row is decoupled from
the original table (a ``catalog_row_index`` entry has been added to the ``data`` dict).

Each specific catalog (e.g. `gammapy.catalog.SourceCatalog3FGL`) is implemented as a
sub-class and can add functionality that's specific to that catalog, e.g. access
to a spectrum object that can be plotted.

Two command-line tools are provided to quickly query and browse catalog information:

* ``gammapy-catalog-query`` -- Print catalog source info to the terminal and make plots.
* ``gammapy-catalog-browse`` - A web app to browse catalog source info and make plots.

`gammapy.catalog` also contains some code to analyse source catalogs
(e.g. via `gammapy.catalog.FluxDistribution`), but that's very limited at this time.
Contributions welcome!

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

TODO: give one example, e.g. how to reproduce a log(N)-log(S) plot from a Fermi catalog paper.

Reference/API
=============

.. automodapi:: gammapy.catalog
    :no-inheritance-diagram:
