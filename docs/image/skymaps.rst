Sky Maps
========

Introduction and Concept
------------------------

The `~gammapy.image.SkyMap` class represents the main data container class for
map based gamma-ray data. It combines the raw 2D data arrays with sky coordinates
represented by WCS objects and Fits I/O functionality. Additionally it provides
convenience functions for and creating, exploring and accessing the data.
Data processing methods (except for very basic ones) are not coupled to this class.


Getting started
---------------

Most easily a `~gammapy.image.SkyMap` can be created from a fits file:

.. code::

    from gammapy.image import SkyMap
    from gammapy.datasets import load_poisson_stats_image

    f = load_poisson_stats_image(return_filenames=True)
    skymap = SkyMap.read(f)

Alternatively an empty sky map cen be created from the scratch, by specifying the
WCS information (see `~gammapy.image.SkyMap.empty` for a detailed description of
the parameters):

.. code::

    skymap_empty = SkyMap.empty('empty')

Where the optional string ``'empty'`` specifies the name of the sky map.

Some basic info on the map is shown when calling:

.. code::

    skymap.info()

To lookup the value of the data at a certain sky position one can do:

.. code::

    from astropy.coordinates import SkyCoord
    position = SkyCoord(0, 0, frame='galactic', unit='deg')
    skymap.lookup(position)

Or directly pass a tuple of ``(ra, dec)`` or ``(lon, lat)``, depending on the
type of WCS transformation, that is set.

The sky map can be easily displayed with an image viewer, by calling ``skymap.show()``:

.. plot::
        :include-source:

        from gammapy.image import SkyMap
        from gammapy.datasets import load_poisson_stats_image

        f = load_poisson_stats_image(return_filenames=True)
        counts = SkyMap.read(f)
        counts.name = 'Counts'
        counts.show()
