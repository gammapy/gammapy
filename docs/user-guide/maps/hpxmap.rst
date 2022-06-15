.. _hpxmap:

HEALPix-based maps
==================

This page provides examples and documentation specific to the HEALPix map
classes. All HEALPix classes inherit from `~gammapy.maps.Map` which provides generic
interface methods that can be be used to access or update the contents of a map
without reference to its pixelization scheme.

.. warning::

    Gammapy uses `NEST` as default pixel order scheme, while `~healpy`
    functions have `RING` as the default (see https://healpy.readthedocs.io/en/1.11.0/index.html).
    If you are interfacing Gammapy HEALPix maps with `~healpy` functions, you need to specify the pixelization scheme
    either while creating the Gammapy object or when using the `~healpy` functions.

HEALPix geometry
----------------

The `~gammapy.maps.HpxGeom` class encapsulates the geometry of a HEALPix map:
the pixel size (NSIDE), coordinate system, and definition of non-spatial axes
(e.g. energy).  By default a HEALPix geometry will encompass the full sky.  The
following example shows how to create an all-sky 2D HEALPix image:

.. testcode::

    from gammapy.maps import HpxGeom, HpxNDMap, HpxMap
    # Create a HEALPix geometry of NSIDE=16
    geom = HpxGeom(16, frame="galactic")
    m = HpxNDMap(geom)

    # Equivalent factory method call
    m = HpxMap.create(nside=16, frame="galactic")

Partial-sky maps can be created by passing a ``region`` argument to the map
geometry constructor or by setting the ``width`` argument to the
`~gammapy.maps.HpxMap.create` factory method:

.. testcode::

    from gammapy.maps import HpxGeom, HpxMap, HpxNDMap
    from astropy.coordinates import SkyCoord

    # Create a partial-sky HEALPix geometry of NSIDE=16
    geom = HpxGeom(16, region='DISK(0.0,5.0,10.0)', frame="galactic")
    m = HpxNDMap(geom)

    # Equivalent factory method call
    position = SkyCoord(0.0, 5.0, frame='galactic', unit='deg')
    m = HpxMap.create(nside=16, skydir=position, width=20.0)
