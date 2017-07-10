.. include:: ../references.txt

.. _maps:

*****************************************************
Data Structures for Images and Cubes (`gammapy.maps`)
*****************************************************

.. currentmodule:: gammapy.maps

.. warning::

    The code in `gammapy.maps` is currently in an
    experimental/development state. Method and class names may change
    in the future.
                   
Introduction
============

`gammapy.maps` contains classes for representing pixelized data
structures with at least two spatial dimensions representing
coordinates on a sphere (e.g. an image in celestial coordinates).  These classes
support an arbitrary number of non-spatial dimensions and can
represent images (2D), cubes (3D), or hypercubes (4+D).  Two
pixelization schemes are supported:

* WCS : Projection onto a 2D cartesian grid following the conventions
  of the World Coordinate System (WCS).  Pixels are square in
  projected coordinates and as such are not equal area in spherical
  coordinates.
* HEALPix : Hierarchical Equal Area Iso Latitude pixelation of the
  sphere.  Pixels are equal area but have irregular shapes.


Getting Started
===============

All map objects have an abstract inteface provided through the methods
of the `~MapBase`.  These methods can be used for accessing and
manipulating the contents of a map without reference to the underlying
data representation (e.g. whether a map uses WCS or HEALPix
pixelization).  For applications which do depend on the specific
representation one can also work directly with the classes derived
from `~MapBase`.  In the following we review some of the basic methods
for working with map objects.

Constructing with Factory Methods
---------------------------------

The `~MapBase` class provides a `~MapBase.create` factory method to
facilitate creating an empty map object from scratch.  The
``map_type`` argument can be used to control the pixelization scheme
(WCS or HPX) and whether the map internally uses a sparse
representation of the data.

.. code::

   from gammapy.maps import MapBase
   from astropy.coordinates import SkyCoord
   position = SkyCoord(0.0, 5.0, frame='galactic', unit='deg')

   # Create a WCS Map
   m_wcs = MapBase.create(binsz=0.1, map_type='wcs', skydir=position, width=10.0)

   # Create a HPX Map
   m_hpx = MapBase.create(binsz=0.1, map_type='hpx', skydir=position, width=10.0)
   
Higher dimensional map objects (cubes and hypercubes) can be
constructed by passing a list of `~MapAxis` objects for non-spatial
dimensions with the ``axes`` parameter:

.. code::

   from gammapy.maps import MapBase, MapAxis
   from astropy.coordinates import SkyCoord
   position = SkyCoord(0.0, 5.0, frame='galactic', unit='deg')
   energy_axis = MapAxis.from_bounds(100., 1E5, 12, interp='log')
   
   # Create a WCS Map
   m_wcs = MapBase.create(binsz=0.1, map_type='wcs', skydir=position, width=10.0,
                          axes=[energy_axis])

   # Create a HPX Map
   m_hpx = MapBase.create(binsz=0.1, map_type='hpx', skydir=position, width=10.0,
                          axes=[energy_axis])

   
Get, Set, and Fill Methods
--------------------------

All map objects have a set of accessor methods provided through the
abstract `~MapBase` class that can be used to retrieve or update the
contents of the map.  Accessor methods accept as their first
argument a tuple of vectors (lists or numpy arrays) with the coordinates of
pixels within the map expressed in one of three coordinate systems:

* ``idx`` : Pixel indices.  These are explicit pixel indices into the map.  
* ``pix`` : Coordinates in pixel space.  Pixel coordinates are defined
  on the interval [0,N-1] where N is the number of pixels along a
  given map dimension with pixel centers at integer values.  For
  methods that reference a discrete pixel, pixel coordinates wil be
  rounded to the nearest pixel index and passed to the corresponding
  ``idx`` method.
* ``coord`` : Sky coordinates.  The tuple should contain longitude and
  latitude in degrees followed by one coordinate array for each
  non-spatial dimension.
  
The coordinate system accepted by a given accessor method can be
inferred from the suffix of the method name
(e.g. `~MapBase.get_by_idx`).

The ``get`` methods return the contents of the map for a sequence of
pixels.  The following demonstrates how one can access the same pixels
of a WCS map using each of the three coordinate systems:

.. code::

   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)
   
   vals = m.get_by_idx( ([49,50],[49,50]) )
   vals = m.get_by_pix( ([49.0,50.0],[49.0,50.0]) )
   vals = m.get_by_coords( ([-0.05,-0.05],[0.05,0.05]) )

The ``set`` methods can be used to set pixel values.  The following
demonstrates how one can set same pixel values:

.. code::

   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)
   
   m.set_by_idx( ([49,50],[49,50]), [0.5, 1.5] )
   m.set_by_pix( ([49.0,50.0],[49.0,50.0]), [0.5, 1.5] )
   m.set_by_coords( ([-0.05,-0.05],[0.05,0.05]), [0.5, 1.5] )

Finally the ``fill`` methods can be used to increment the value of a
set of pixels according to a weights vector.

.. code::

   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)
   
   m.fill_by_idx( ([49,50],[49,50]), weights=[0.5, 1.5] )
   m.fill_by_pix( ([49.0,50.0],[49.0,50.0]), weights=[0.5, 1.5] )
   m.fill_by_coords( ([-0.05,-0.05],[0.05,0.05]), weights=[0.5, 1.5] )

   
Slicing Methods
---------------

Iterating on a Map
------------------

File I/O
--------

Maps can be written to and read from a FITS file with the ``write``
and ``read`` methods.

.. code::

   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)
   m.write('file.fits', extname='IMAGE')
   m = MapBase.read('test.fits', extname='IMAGE')

Images can be serialized to a sparse data format by passing
``sparse=True``.  This will write the file to a sparse data table
appropriate to the pixelization scheme.

.. code::

   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)
   m.write('file.fits', extname='IMAGE', sparse=True)
   m = MapBase.read('test.fits', extname='IMAGE')

Using `gammapy.maps`
====================

More detailed documentation on the WCS and HPX classes in
`gammapy.maps` can be found in the following sub-pages:

.. toctree::
   :maxdepth: 1

   hpxmap
   wcsmap


Reference/API
=============

.. automodapi:: gammapy.maps
    :no-inheritance-diagram:



