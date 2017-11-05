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

`gammapy.maps` is organized around two data structures:
*geometry* classes inheriting from `~MapGeom` and *map* classes
inheriting from `~MapBase`.  A geometry defines the map
boundaries, pixelization scheme, and provides methods for converting
to/from map and pixel coordinates.  A map owns a `~MapGeom`
instance as well as a data structure containing map values.  Where
possible it is recommended to use the abstract `~MapBase` interface
for accessing or updating the contents of a map as this allows
algorithms to be used interchangeably with different map
representations.  The following reviews methods of the abstract
map interface.  Documentation specific to WCS- and HEALPix-based maps is
provided in :doc:`hpxmap` and :doc:`wcsmap`.


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

.. code:: python

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

.. code:: python

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

Multi-resolution maps (maps with a different pixel size or geometry in
each image plane) can be constructed by passing a vector argument for
any of the geometry parameters.  This vector must have the same shape as the
non-spatial dimensions of the map.  The following example demonstrates
creating an energy cube with a pixel size proportional to the
Fermi-LAT PSF:

.. code:: python

   import numpy as np
   from gammapy.maps import MapBase, MapAxis
   from astropy.coordinates import SkyCoord
   position = SkyCoord(0.0, 5.0, frame='galactic', unit='deg')
   energy_axis = MapAxis.from_bounds(100., 1E5, 12, interp='log')

   binsz = np.sqrt((3.0*(energy_axis.center/100.)**-0.8)**2 + 0.1**2)

   # Create a WCS Map
   m_wcs = MapBase.create(binsz=binsz, map_type='wcs', skydir=position, width=10.0,
                          axes=[energy_axis])

   # Create a HPX Map
   m_hpx = MapBase.create(binsz=binsz, map_type='hpx', skydir=position, width=10.0,
                          axes=[energy_axis])

Get, Set, and Fill Methods
--------------------------

All map objects have a set of accessor methods provided through the
abstract `~MapBase` class.  These methods can be used to access or
update the contents of the map irrespective of its underlying
representation.  Three types of accessor methods are provided:

* ``get`` : Return the value of the map at the pixel containing the
  given coordinate (`~MapBase.get_by_idx`, `~MapBase.get_by_pix`,
  `~MapBase.get_by_coords`).  With the ``interp`` argument,
  `~MapBase.get_by_pix` and `~MapBase.get_by_coords` also support
  interpolation of map values between pixels (see `Interpolation`_).
* ``set`` : Set the value of the map at the pixel containing the
  given coordinate (`~MapBase.set_by_idx`, `~MapBase.set_by_pix`,
  `~MapBase.set_by_coords`).
* ``fill`` : Increment the value of the map at the pixel containing
  the given coordinate with a unit weight or the value in the optional
  ``weights`` argument (`~MapBase.fill_by_idx`,
  `~MapBase.fill_by_pix`, `~MapBase.fill_by_coords`).

All accessor methods accept as their first argument a
coordinate tuple containing scalars, lists, or numpy arrays with one
tuple element for each dimension of the map.  By convention the first
two elements in the coordinate tuple correspond to longitude and
latitude followed by one element for each non-spatial dimension.  Map
coordinates can be expressed in one of three coordinate systems:

* ``idx`` : Pixel indices.  These are explicit (integer) pixel indices into the map.
* ``pix`` : Coordinates in pixel space.  Pixel coordinates are continuous defined
  on the interval [0,N-1] where N is the number of pixels along a
  given map dimension with pixel centers at integer values.  For
  methods that reference a discrete pixel, pixel coordinates wil be
  rounded to the nearest pixel index and passed to the corresponding
  ``idx`` method.
* ``coord`` : Sky (spherical) coordinates.  The tuple should contain longitude and
  latitude in degrees followed by one coordinate array for each
  non-spatial dimension.

The coordinate system accepted by a given accessor method can be
inferred from the suffix of the method name
(e.g. `~MapBase.get_by_idx`).  The following demonstrates how one can
access the same pixels of a WCS map using each of the three coordinate
systems:

.. code:: python

   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)

   vals = m.get_by_idx( ([49,50],[49,50]) )
   vals = m.get_by_pix( ([49.0,50.0],[49.0,50.0]) )
   vals = m.get_by_coords( ([-0.05,-0.05],[0.05,0.05]) )

Coordinate arguments obey normal numpy broadcasting rules.  The
coordinate tuple may contain any combination of scalars, lists or
numpy arrays as long as they have compatible shapes.  For instance a
combination of scalar and vector arguments can be used to perform an
operation along a slice of the map at a fixed value along that
dimension.  Multi-dimensional arguments can be use to broadcast a
given operation across a grid of coordinate values.

.. code:: python

   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)
   coords = np.linspace(-4.0,4.0,9)

   # Equivalent calls for accessing value at pixel (49,49)
   vals = m.get_by_idx( (49,49) )
   vals = m.get_by_idx( ([49],[49]) )
   vals = m.get_by_idx( (np.array([49]),np.array([49])) )

   # Retrieve map values along latitude at fixed longitude=0.0
   vals = m.get_by_coords( (0.0, coords) )
   # Retrieve map values on a 2D grid of latitude/longitude points
   vals = m.get_by_coords( (coords[None,:], coords[:,None]) )
   # Set map values along slice at longitude=0.0 to twice their existing value
   m.set_by_coords((0.0, coords), 2.0*m.get_by_coords((0.0, coords)))

The ``set`` and ``fill`` methods can both be used to set pixel values.
The following demonstrates how one can set pixel values:

.. code:: python

   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)

   m.set_by_coords( ([-0.05,-0.05],[0.05,0.05]), [0.5, 1.5] )
   m.fill_by_coords( ([-0.05,-0.05],[0.05,0.05]), weights=[0.5, 1.5] )

Interpolation
-------------

Maps support interpolation via the `~MapBase.get_by_coords` and
`~MapBase.get_by_pix` methods.  Currently the following interpolation
methods are supported:

* ``nearest`` : Return value of nearest pixel (no interpolation).
* ``linear`` : Interpolation with first order polynomial.  This is the
  only interpolation method that is supported for all map types.
* ``quadratic`` : Interpolation with second order polynomial.
* ``cubic`` : Interpolation with third order polynomial.

Note that ``quadratic`` and ``cubic`` interpolation are currently only
supported for WCS-based maps with regular geometry (e.g. 2D or ND with
the same geometry in every image plane).  ``linear`` and higher order
interpolation by pixel coordinates is only supported for WCS-based
maps.

.. code:: python

   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)

   m.get_by_coords( ([-0.05,-0.05],[0.05,0.05]), interp='linear' )
   m.get_by_coords( ([-0.05,-0.05],[0.05,0.05]), interp='cubic' )


Projection
----------

The `~MapBase.reproject` method can be used to project a map onto a
different geometry.  This can be used to convert between different WCS
projections, extract a cut-out of a map, or to convert between WCS and
HPX map types.  If the projection geometry lacks non-spatial
dimensions then the non-spatial dimensions of the original map will be copied
over to the projected map.

.. code:: python

   from gammapy.maps import WcsMapND, HpxGeom
   m = WcsMapND.read('gll_iem_v06.fits')
   geom = HpxGeom.create(nside=8, coordsys='GAL')
   # Convert LAT standard IEM to HPX (nside=8)
   m_proj = m.project(geom)
   m_proj.write('gll_iem_v06_hpx_nside8.fits')


Slicing Methods
---------------

Iterating on a Map
------------------

Iterating over a map can be performed with the
`~MapBase.iter_by_coords` and `~MapBase.iter_by_pix` methods.  These
return an iterator that traverses the map returning (value,
coordinate) pairs with map and pixel coordinates, respectively.  The
optional ``buffersize`` argument can be used to split the iteration
into chunks of a given size.  The following example illustrates how
one can use this method to fill a map with a 2D Gaussian:

.. code:: python

   import numpy as np
   from astropy.coordinates import SkyCoord
   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.05, map_type='wcs', width=10.0)
   for val, coords in m.iter_by_coords(buffersize=10000):
       skydir = SkyCoord(coords[0],coords[1], unit='deg')
       sep = skydir.separation(m.geom.center_skydir).deg
       new_val = np.exp(-sep**2/2.0)
       m.set_by_coords(coords, new_val)

For maps with non-spatial dimensions the `~MapBase.iter_by_image`
method can be used to loop over image slices:

.. code:: python

   from astropy.coordinates import SkyCoord
   from astropy.convolution import Gaussian2DKernel, convolve
   from gammapy.maps import MapBase
   m = MapBase.create(binsz=0.05, map_type='wcs', width=10.0)
   for img, idx in m.iter_by_image():
       img = convolve(img, Gaussian2DKernel(stddev=2.0) )


FITS I/O
--------

Maps can be written to and read from a FITS file with the
`~MapBase.write` and ``read`` methods:

.. code:: python

   from gammapy.maps import MapBase, WcsMapND
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)
   m.write('file.fits', extname='IMAGE')
   m = WcsMapND.read('file.fits', hdu='IMAGE')

Maps can be serialized to a sparse data format by calling
`~MapBase.write` with ``sparse=True``.  This will write all non-zero
pixels in the map to a data table appropriate to the pixelization
scheme.

.. code:: python

   from gammapy.maps import MapBase, WcsMapND
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)
   m.write('file.fits', extname='IMAGE', sparse=True)
   m = WcsMapND.read('file.fits', hdu='IMAGE')

Sparse maps have the same ``read`` and ``write`` methods with the
exception that they will be written to a sparse format by default:

.. code:: python

   from gammapy.maps import MapBase, HpxMapSparse
   m = MapBase.create(binsz=0.1, map_type='hpx-sparse', width=10.0)
   m.write('file.fits', extname='IMAGE')
   m = HpxMapSparse.read('file.fits', hdu='IMAGE')

By default files will be written to the *gamma-astro-data-format*
specification for sky maps (see `here
<http://gamma-astro-data-formats.readthedocs.io/en/latest/skymaps/index.html>`_).
The GADF format offers a number of enhancements over existing map
formats such as support for writing multi-resolution maps, sparse
maps, and cubes with different geometries to the same file.  For
backward compatibility with software using other formats, the ``conv``
keyword option is provided to write a file using a format other than
the GADF format:

.. code:: python

   from gammapy.maps import MapBase, MapAxis
   energy_axis = MapAxis.from_bounds(100., 1E5, 12, interp='log')
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0,
                      axes=[energy_axis])
   # Write a counts cube in a format compatible with the Fermi Science Tools
   m.write('ccube.fits', conv='fgst-ccube')

Visualization
-------------

All map objects provide a `~MapBase.plot` method for generating a
visualization of a map.  This method returns figure, axes, and image
objects that can be used to further tweak/customize the image.

.. code:: python

   import matplotlib.pyplot as plt
   from gammapy.maps import MapBase
   from gammapy.maps.utils import fill_poisson
   m = MapBase.create(binsz=0.1, map_type='wcs', width=10.0)
   fill_poisson(m, mu=1.0, random_state=0)
   fig, ax, im = m.plot(cmap='magma')
   plt.colorbar(im)


Examples
========

Creating a Counts Cube from an FT1 File
---------------------------------------

This example shows how to fill a counts cube from an FT1 file:

.. code:: python

   from astropy.io import fits
   from gammapy.maps import WcsGeom, WcsMapND, MapAxis

   h = fits.open('ft1.fits')
   energy_axis = MapAxis.from_bounds(100., 1E5, 12, interp='log')
   m = WcsMapND.create(binsz=0.1, width=10.0, skydir=(45.0,30.0),
                       coordsys='CEL', axes=[energy_axis])
   m.fill_by_coords((h['EVENTS'].data.field('RA'),
                     h['EVENTS'].data.field('DEC'),
                     h['EVENTS'].data.field('ENERGY')))
   m.write('ccube.fits', conv='fgst-ccube')

Generating a Cutout of a Model Cube
-----------------------------------

This example shows how to extract a cut-out of LAT galactic
diffuse model cube using the `~MapBase.reproject` method:

.. code:: python

   from gammapy.maps import WcsGeom, WcsMapND
   m = WcsMapND.read('gll_iem_v06.fits')
   geom = WcsGeom(binsz=0.125, skydir=(45.0,30.0), coordsys='GAL', proj='AIT')
   m_proj = m.reproject(geom)
   m_proj.write('cutout.fits', conv='fgst-template')


Using `gammapy.maps`
====================

:ref:`tutorials` that show examples using ``gammapy.maps``:

* :gp-extra-notebook:`data_fermi_lat`

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
    :include-all-objects:
