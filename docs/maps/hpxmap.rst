.. _hpxmap:

HEALPix-based Maps
==================

This page provides examples and documentation specific to the HEALPix
map classes (`~gammapy.maps.HpxMapND` and
`~gammapy.maps.HpxMapSparse`).  All HEALPix classes inherit from
`~gammapy.maps.MapBase` which provides generic interface methods that can be be
used to access or update the contents of a map without reference to
its pixelization scheme.

HEALPix Geometry
----------------

The `~gammapy.maps.HpxGeom` class encapsulates the geometry of a
HEALPix map: the pixel size (NSIDE), coordinate system, and definition
of non-spatial axes (e.g. energy).  By default a HEALPix geometry will
encompass the full sky.  The following example shows how to create
an all-sky 2D HEALPix image:

.. code:: python

   from gammapy.maps import HpxGeom, HpxMapND, HpxMap
   # Create a HEALPix geometry of NSIDE=16
   geom = HpxGeom(16, coordsys='GAL')
   m = HpxMapND(geom)

   # Equivalent factory method call
   m = HpxMap.create(nside=16, coordsys='GAL')

Partial-sky maps can be created by passing a ``region`` argument to
the map geometry constructor or by setting the ``width`` argument to
the `~gammapy.maps.HpxMap.create` factory method:

.. code:: python

   from gammapy.maps import HpxGeom, HpxMap, HpxMapND
   from astropy.coordinates import SkyCoord

   # Create a partial-sky HEALPix geometry of NSIDE=16
   geom = HpxGeom(16, region='DISK(0.0,5.0,10.0)', coordsys='GAL')
   m = HpxMapND(geom)

   # Equivalent factory method call
   position = SkyCoord(0.0, 5.0, frame='galactic', unit='deg')
   m = HpxMap.create(nside=16, skydir=position, width=20.0)





Sparse Maps
-----------

The `~gammapy.maps.HpxMapSparse` class is a memory-efficient
implementation of a HEALPix map that uses a sparse data structure to
store map values.  Sparse maps can be useful when working with maps
that have many empty pixels (e.g. a low-statistics counts map).
