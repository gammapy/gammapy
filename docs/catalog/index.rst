.. _catalog:

*************************
catalog - Source catalogs
*************************

.. currentmodule:: gammapy.catalog

Introduction
============

`gammapy.catalog` provides utilities to work with source catalogs in general,
and catalogs relevant for gamma-ray astronomy specifically.

Support for the following catalogs is available:

* ``hgps`` / `SourceCatalogHGPS` - H.E.S.S. Galactic plane survey (HGPS)
* ``gamma-cat`` /  / `SourceCatalogGammaCat` - An open catalog of gamma-ray sources
* ``3fgl`` / `SourceCatalog3FGL` - LAT 4-year point source catalog
* ``4fgl`` / `SourceCatalog4FGL` - LAT 8-year point source catalog
* ``2fhl`` / `SourceCatalog2FHL` - LAT second high-energy source catalog
* ``3fhl`` / `SourceCatalog3FHL` - LAT third high-energy source catalog
* ``2hwc`` / `SourceCatalog2HWC` - 2HWC catalog from the HAWC observatory

More catalogs can be added to ``gammapy.catalog``, and users can also add
support for their favourite catalog in their Python script or package, by
following the examples how the built-in catalogs are implemented.

Some catalogs can be interactively explored at http://gamma-sky.net .

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

    >>> from gammapy.catalog import SOURCE_CATALOGS
    >>> list(SOURCE_CATALOGS)
    ['gamma-cat', 'hgps', '2hwc', '3fgl', '4fgl', '2fhl', '3fhl']

The ``SOURCE_CATALOG`` dict maps names to classes, so you have to call the class
to instantiate a catalog object::

    >>> SOURCE_CATALOGS['3fgl']
    <class 'gammapy.catalog.fermi.SourceCatalog3FGL'>
    >>> SOURCE_CATALOGS['3fgl']()
    <gammapy.catalog.fermi.SourceCatalog3FGL object at 0x1006fa198>

Alternatively, you can also import and call the class directly::

    >>> from gammapy.catalog import SourceCatalog3FGL
    >>> catalog = SourceCatalog3FGL()

Usually you create a catalog object, which loads the catalog from disk once,
and then index into it to get source objects representing a given source::

    >>> catalog = SOURCE_CATALOGS["3fgl"]()
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

.. code-block:: python

    from gammapy.catalog import SOURCE_CATALOGS
    source = SOURCE_CATALOGS['3fgl']()['3FGL J0349.9-2102']
    lc = source.lightcurve
    lc.plot()

Using `gammapy.catalog`
=======================

For more advanced use cases please go to the tutorial notebooks:

* `hgps.html <../notebooks/hgps.html>`__
* `first_steps.html <../notebooks/first_steps.html>`__
* `sed_fitting_gammacat_fermi.html <../notebooks/sed_fitting_gammacat_fermi.html>`__

The following pages describe ``gammapy.catalog`` in more detail:

.. toctree::
    :maxdepth: 1

    gammacat

Reference/API
=============

.. automodapi:: gammapy.catalog
    :no-inheritance-diagram:
    :include-all-objects:
