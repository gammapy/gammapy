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

The `~MapBase` class provides a number of factory methods to facilitate
creating an empty map object from scratch.

.. code::

   from gammapy.maps import MapBase
   from astropy.coordinates import SkyCoord
   position = SkyCoord(0, 0, frame='galactic', unit='deg')

   MapBase.create(binsz=0.1, map_type='wcs')

   MapBase.create_image()

   MapBase.create_cube()

Get and Set Methods
-------------------
   
Slicing Methods
---------------

Iterating on a Map
------------------

File I/O
--------

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



