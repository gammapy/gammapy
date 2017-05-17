.. include:: ../references.txt

.. _gammacat:

*********
gamma-cat
*********

``gamma-cat`` (https://github.com/gammapy/gamma-cat) is an
open data collection and source catalog for TeV gamma-ray astronomy.

As explained further in the ``gamma-cat`` docs, it provides two data products:

1. the full data collection
2. a source catalog with part of the data

Catalog
-------

The gamma-cat catalog is available here:

* ``$GAMMA_CAT/docs/data/gammacat.fits.gz`` -- latest version
* ``$GAMMAPY_EXTRA/datasets/catalogs/gammacat/gammacat.fits.gz``
  - a stable version copied to the ``gammapy-extra`` repo,
  used for the ``gammapy.catalog.gammacat`` tests.

To work with the gamma-cat catalog from Gammapy,
pick a version and create a `~gammapy.catalog.SourceCatalogGammaCat`::

    from gammapy.catalog import SourceCatalogGammaCat
    filename = '$GAMMA_CAT/docs/data/gammacat.fits.gz'
    filename = '$GAMMAPY_EXTRA/datasets/catalogs/gammacat/gammacat.fits.gz'
    cat = SourceCatalogGammaCat(filename)

TODO: add examples how to use it and links to notebooks.

Data collection
---------------

The gamma-cat data collection consists of a bunch of files in JSON and ECSV format, and there's a single
JSON index file summarising all available data and containing pointers to the other files.
(we plan to make a bundled version with all info in one JSON file soon)

It is available here:

* ``$GAMMA_CAT/docs/data/gammacat-datasets.json`` -- latest version
* ``$GAMMAPY_EXTRA/datasets/catalogs/gammacat/gammacat-datasets.json``
  - a stable version copied to the ``gammapy-extra`` repo,
  used for the ``gammapy.catalog.gammacat`` tests.

To work with the gamma-cat data collection from Gammapy,
pick a version and create a `~gammapy.catalog.GammaCatDataCollection` class.

::

    from gammapy.catalog import GammaCatDataCollection
    filename = '$GAMMA_CAT/docs/data/gammacat-datasets.json'
    filename = '$GAMMAPY_EXTRA/datasets/catalogs/gammacat/gammacat-datasets.json'
    gammacat = GammaCatDataCollection.from_index(filename)

TODO: add examples how to use it and links to notebooks.
