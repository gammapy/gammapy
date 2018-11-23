.. include:: ../references.txt

.. _maps:

***************
maps - Sky maps
***************

.. currentmodule:: gammapy.maps

Introduction
============

`gammapy.maps` contains classes for representing pixelized data structures with
at least two spatial dimensions representing coordinates on a sphere (e.g. an
image in celestial coordinates).  These classes support an arbitrary number of
non-spatial dimensions and can represent images (2D), cubes (3D), or hypercubes
(4+D).  Two pixelization schemes are supported:

* WCS : Projection onto a 2D cartesian grid following the conventions
  of the World Coordinate System (WCS).  Pixels are square in projected
  coordinates and as such are not equal area in spherical coordinates.
* HEALPix : Hierarchical Equal Area Iso Latitude pixelation of the
  sphere. Pixels are equal area but have irregular shapes.

`gammapy.maps` is organized around two data structures: *geometry* classes
inheriting from `~MapGeom` and *map* classes inheriting from `~Map`. A geometry
defines the map boundaries, pixelization scheme, and provides methods for
converting to/from map and pixel coordinates. A map owns a `~MapGeom` instance
as well as a data array containing map values. Where possible it is recommended
to use the abstract `~Map` interface for accessing or updating the contents of a
map as this allows algorithms to be used interchangeably with different map
representations. The following reviews methods of the abstract map interface.
Documentation specific to WCS- and HEALPix-based maps is provided in
:doc:`hpxmap` and :doc:`wcsmap`.


Getting Started
===============

All map objects have an abstract inteface provided through the methods of the
`~Map`. These methods can be used for accessing and manipulating the contents of
a map without reference to the underlying data representation (e.g. whether a
map uses WCS or HEALPix pixelization). For applications which do depend on the
specific representation one can also work directly with the classes derived from
`~Map`. In the following we review some of the basic methods for working with
map objects.

Constructing with Factory Methods
---------------------------------

The `~Map` class provides a `~Map.create` factory method to facilitate creating
an empty map object from scratch. The ``map_type`` argument can be used to
control the pixelization scheme (WCS or HPX) and whether the map internally uses
a sparse representation of the data.

.. code:: python

    from gammapy.maps import Map
    from astropy.coordinates import SkyCoord

    position = SkyCoord(0.0, 5.0, frame='galactic', unit='deg')

    # Create a WCS Map
    m_wcs = Map.create(binsz=0.1, map_type='wcs', skydir=position, width=10.0)

    # Create a HPX Map
    m_hpx = Map.create(binsz=0.1, map_type='hpx', skydir=position, width=10.0)

Higher dimensional map objects (cubes and hypercubes) can be constructed by
passing a list of `~MapAxis` objects for non-spatial dimensions with the
``axes`` parameter:

.. code:: python

    from gammapy.maps import Map, MapAxis
    from astropy.coordinates import SkyCoord

    position = SkyCoord(0.0, 5.0, frame='galactic', unit='deg')
    energy_axis = MapAxis.from_bounds(100., 1E5, 12, interp='log', name='energy', unit='GeV')

    # Create a WCS Map
    m_wcs = Map.create(binsz=0.1, map_type='wcs', skydir=position, width=10.0,
                          axes=[energy_axis])

    # Create a HPX Map
    m_hpx = Map.create(binsz=0.1, map_type='hpx', skydir=position, width=10.0,
                          axes=[energy_axis])

Multi-resolution maps (maps with a different pixel size or geometry in each
image plane) can be constructed by passing a vector argument for any of the
geometry parameters. This vector must have the same shape as the non-spatial
dimensions of the map. The following example demonstrates creating an energy
cube with a pixel size proportional to the Fermi-LAT PSF:

.. code:: python

    import numpy as np
    from gammapy.maps import Map, MapAxis
    from astropy.coordinates import SkyCoord

    position = SkyCoord(0.0, 5.0, frame='galactic', unit='deg')
    energy_axis = MapAxis.from_bounds(100., 1E5, 12, interp='log', name='energy', unit='GeV')

    binsz = np.sqrt((3.0*(energy_axis.center/100.)**-0.8)**2 + 0.1**2)

    # Create a WCS Map
    m_wcs = Map.create(binsz=binsz, map_type='wcs', skydir=position, width=10.0,
                          axes=[energy_axis])

    # Create a HPX Map
    m_hpx = Map.create(binsz=binsz, map_type='hpx', skydir=position, width=10.0,
                          axes=[energy_axis])

.. _mapslicing:

Indexing and Slicing
--------------------

All map objects feature a `~Map.slice_by_idx()` method, which can be used to
slice and index non-spatial axes of the map to create arbitrary sub-maps. The
method accepts a `dict` specifying the axes name and correspoding integer index
or `slice` objects. When indexing an axis with an integer the corresponding axes
is dropped from the returned sub-map. To keep the axes (with length 1) in the
returned sub-map use a `slice` object of length one. This behaviour is
equivalent to regular numpy array indexing. The following example demonstrates
the use of `~Map.slice_by_idx()` on a map with a time and energy axes:

.. code:: python

    import numpy as np
    from gammapy.maps import Map, MapAxis
    from astropy.coordinates import SkyCoord

    position = SkyCoord(0.0, 5.0, frame='galactic', unit='deg')
    energy_axis = MapAxis.from_bounds(100., 1E5, 12, interp='log', unit='GeV', name='energy')
    time_axis = MapAxis.from_bounds(0., 12, 12, interp='lin', unit='h', name='time')

    # Create a WCS Map
    m_wcs = Map.create(binsz=0.02, map_type='wcs', skydir=position, width=10.0,
                          axes=[energy_axis, time_axis])

    # index first image plane of the energy axes and third from the time axis
    m_wcs.slice_by_idx({'energy': 0, 'time': 2})

    # index first image plane of the energy axes and keep time axis unchanged
    m_wcs.slice_by_idx({'energy': 0})

    # slice first three images of the energy axis at a fixed time
    m_wcs.slice_by_idx({'energy': slice(0, 3), 'time': 0})

    # slice first three images of the energy axis as well as time axis
    m_wcs.slice_by_idx({'energy': slice(0, 3), 'time': slice(0, 3)})

Accessor Methods
----------------

All map objects have a set of accessor methods provided through the abstract
`~Map` class. These methods can be used to access or update the contents of the
map irrespective of its underlying representation. Four types of accessor
methods are provided:

* ``get`` : Return the value of the map at the pixel containing the
  given coordinate (`~Map.get_by_idx`, `~Map.get_by_pix`, `~Map.get_by_coord`).
* ``interp`` : Interpolate or extrapolate the value of the map at an arbitrary
  coordinate (see also `Interpolation`_).
* ``set`` : Set the value of the map at the pixel containing the
  given coordinate (`~Map.set_by_idx`, `~Map.set_by_pix`, `~Map.set_by_coord`).
* ``fill`` : Increment the value of the map at the pixel containing
  the given coordinate with a unit weight or the value in the optional
  ``weights`` argument (`~Map.fill_by_idx`, `~Map.fill_by_pix`,
  `~Map.fill_by_coord`).

Accessor methods accept as their first argument a coordinate tuple containing
scalars, lists, or numpy arrays with one tuple element for each dimension of the
map. ``coord`` methods optionally support a `dict` or `~MapCoord` argument.

When using tuple input the first two elements in the tuple should be longitude
and latitude followed by one element for each non-spatial dimension. Map
coordinates can be expressed in one of three coordinate systems:

* ``idx`` : Pixel indices.  These are explicit (integer) pixel indices into the
  map.
* ``pix`` : Coordinates in pixel space.  Pixel coordinates are continuous defined
  on the interval [0,N-1] where N is the number of pixels along a given map
  dimension with pixel centers at integer values.  For methods that reference a
  discrete pixel, pixel coordinates wil be rounded to the nearest pixel index
  and passed to the corresponding ``idx`` method.
* ``coord`` : The true map coordinates including angles on the sky (longitude
  and latitude).  This coordinate system supports three coordinate
  representations: `tuple`, `dict`, and `~MapCoord`.  The tuple representation
  should contain longitude and latitude in degrees followed by one coordinate
  array for each non-spatial dimension.

The coordinate system accepted by a given accessor method can be inferred from
the suffix of the method name (e.g. `~Map.get_by_idx`).  The following
demonstrates how one can access the same pixels of a WCS map using each of the
three coordinate systems:

.. code:: python

    from gammapy.maps import Map

    m = Map.create(binsz=0.1, map_type='wcs', width=10.0)

    vals = m.get_by_idx( ([49,50],[49,50]) )
    vals = m.get_by_pix( ([49.0,50.0],[49.0,50.0]) )
    vals = m.get_by_coord( ([-0.05,-0.05],[0.05,0.05]) )

Coordinate arguments obey normal numpy broadcasting rules.  The coordinate tuple
may contain any combination of scalars, lists or numpy arrays as long as they
have compatible shapes.  For instance a combination of scalar and vector
arguments can be used to perform an operation along a slice of the map at a
fixed value along that dimension. Multi-dimensional arguments can be use to
broadcast a given operation across a grid of coordinate values.

.. code:: python

    from gammapy.maps import Map

    m = Map.create(binsz=0.1, map_type='wcs', width=10.0)
    coords = np.linspace(-4.0, 4.0, 9)

    # Equivalent calls for accessing value at pixel (49,49)
    vals = m.get_by_idx( (49,49) )
    vals = m.get_by_idx( ([49],[49]) )
    vals = m.get_by_idx( (np.array([49]), np.array([49])) )

    # Retrieve map values along latitude at fixed longitude=0.0
    vals = m.get_by_coord( (0.0, coords) )
    # Retrieve map values on a 2D grid of latitude/longitude points
    vals = m.get_by_coord( (coords[None,:], coords[:,None]) )
    # Set map values along slice at longitude=0.0 to twice their existing value
    m.set_by_coord((0.0, coords), 2.0*m.get_by_coord((0.0, coords)))

The ``set`` and ``fill`` methods can both be used to set pixel values. The
following demonstrates how one can set pixel values:

.. code:: python

    from gammapy.maps import Map

    m = Map.create(binsz=0.1, map_type='wcs', width=10.0)

    m.set_by_coord(([-0.05, -0.05], [0.05, 0.05]), [0.5, 1.5])
    m.fill_by_coord( ([-0.05, -0.05], [0.05, 0.05]), weights=[0.5, 1.5])

Interface with `~MapCoord` and `~astropy.coordinates.SkyCoord`
--------------------------------------------------------------

The ``coord`` accessor methods accept `dict`, `~MapCoord`, and
`~astropy.coordinates.SkyCoord` arguments in addition to the standard `tuple` of
`~numpy.ndarray` argument.  When using a `tuple` argument a
`~astropy.coordinates.SkyCoord` can be used instead of longitude and latitude
arrays.  The coordinate frame of the `~astropy.coordinates.SkyCoord` will be
transformed to match the coordinate system of the map.

.. code:: python

    import numpy as np
    from astropy.coordinates import SkyCoord
    from gammapy.maps import Map, MapCoord, MapAxis

    lon = [0, 1]
    lat = [1, 2]
    energy = [100, 1000]
    energy_axis = MapAxis.from_bounds(100, 1E5, 12, interp='log', name='energy')

    skycoord = SkyCoord(lon, lat, unit='deg', frame='galactic')
    m = Map.create(binsz=0.1, map_type='wcs', width=10.0,
                  coordsys='GAL', axes=[energy_axis])

    m.set_by_coord((skycoord, energy), [0.5, 1.5])
    m.get_by_coord((skycoord, energy))

A `~MapCoord` or `dict` argument can be used to interact with a map object
without reference to the axis ordering of the map geometry:

.. code:: python

    coord = MapCoord.create(dict(lon=lon, lat=lat, energy=energy))
    m.set_by_coord(coord, [0.5, 1.5])
    m.get_by_coord(coord)
    m.set_by_coord(dict(lon=lon, lat=lat, energy=energy), [0.5, 1.5])
    m.get_by_coord(dict(lon=lon, lat=lat, energy=energy))

However when using the named axis interface the axis name string (e.g. as given
by `MapAxis.name`) must match the name given in the method argument.  The two
spatial axes must always be named ``lon`` and ``lat``.

.. _mapcoord:

MapCoord
--------

`MapCoord` is an N-dimensional coordinate object that stores both spatial and
non-spatial coordinates and is accepted by all ``coord`` methods. A `~MapCoord`
can be created with or without explicitly named axes with `MapCoord.create`.
Axes of a `MapCoord` can be accessed by index, name, or attribute.  A `MapCoord`
without explicit axis names can be created by calling `MapCoord.create` with a
`tuple` argument:

.. code:: python

    import numpy as np
    from astropy.coordinates import SkyCoord
    from gammapy.maps import MapCoord

    lon = [0.0, 1.0]
    lat = [1.0, 2.0]
    energy = [100, 1000]
    skycoord = SkyCoord(lon, lat, unit='deg', frame='galactic')

    # Create a MapCoord from a tuple (no explicit axis names)
    c = MapCoord.create((lon, lat, energy))
    print(c[0], c['lon'], c.lon)
    print(c[1], c['lat'], c.lat)
    print(c[2], c['axis0'])

    # Create a MapCoord from a tuple + SkyCoord (no explicit axis names)
    c = MapCoord.create((skycoord, energy))
    print(c[0], c['lon'], c.lon)
    print(c[1], c['lat'], c.lat)
    print(c[2], c['axis0'])

The first two elements of the tuple argument must contain longitude and
latitude.  Non-spatial axes are assigned a default name ``axis{I}`` where
``{I}`` is the index of the non-spatial dimension. `MapCoord` objects created
without named axes must have the same axis ordering as the map geometry.

A `MapCoord` with named axes can be created by calling `MapCoord.create` with a
`dict` or `~collections.OrderedDict`:

.. code:: python

    # Create a MapCoord from a dict
    c = MapCoord.create(dict(lon=lon, lat=lat, energy=energy))
    print(c[0], c['lon'], c.lon)
    print(c[1], c['lat'], c.lat)
    print(c[2], c['energy'])

    # Create a MapCoord from an OrderedDict
    from collections import OrderedDict
    c = MapCoord.create(OrderedDict([('energy',energy), ('lon',lon), ('lat', lat)]))
    print(c[0], c['energy'])
    print(c[1], c['lon'], c.lon)
    print(c[2], c['lat'], c.lat)

    # Create a MapCoord from a dict + SkyCoord
    c = MapCoord.create(dict(skycoord=skycoord, energy=energy))
    print(c[0], c['lon'], c.lon)
    print(c[1], c['lat'], c.lat)
    print(c[2], c['energy'])

Spatial axes must be named ``lon`` and ``lat``. `MapCoord` objects created with
named axes do not need to have the same axis ordering as the map geometry.
However the name of the axis must match the name of the corresponding map
geometry axis.

Interpolation
-------------

Maps support interpolation via the `~Map.interp_by_coord` and
`~Map.interp_by_pix` methods.  Currently the following interpolation methods are
supported:

* ``nearest`` : Return value of nearest pixel (no interpolation).
* ``linear`` : Interpolation with first order polynomial.  This is the
  only interpolation method that is supported for all map types.
* ``quadratic`` : Interpolation with second order polynomial.
* ``cubic`` : Interpolation with third order polynomial.

Note that ``quadratic`` and ``cubic`` interpolation are currently only supported
for WCS-based maps with regular geometry (e.g. 2D or ND with the same geometry
in every image plane). ``linear`` and higher order interpolation by pixel
coordinates is only supported for WCS-based maps.

.. code:: python

    from gammapy.maps import Map

    m = Map.create(binsz=0.1, map_type='wcs', width=10.0)

    m.interp_by_coord(([-0.05, -0.05], [0.05, 0.05]), interp='linear')
    m.interp_by_coord(([-0.05, -0.05], [0.05, 0.05]), interp='cubic')

Projection
----------

The `~Map.reproject` method can be used to project a map onto a different
geometry.  This can be used to convert between different WCS projections,
extract a cut-out of a map, or to convert between WCS and HPX map types.  If the
projection geometry lacks non-spatial dimensions then the non-spatial dimensions
of the original map will be copied over to the projected map.

.. code:: python

    from gammapy.maps import WcsNDMap, HpxGeom

    m = WcsNDMap.read('$GAMMAPY_DATA/fermi_3fhl/gll_iem_v06_cutout.fits')
    geom = HpxGeom.create(nside=8, coordsys='GAL')
    # Convert LAT standard IEM to HPX (nside=8)
    m_proj = m.reproject(geom)
    m_proj.write('gll_iem_v06_hpx_nside8.fits')

.. _mapiter:

Iterating on a Map
------------------

Iterating over a map can be performed with the `~Map.iter_by_coord` and
`~Map.iter_by_pix` methods.  These return an iterator that traverses the map
returning (value, coordinate) pairs with map and pixel coordinates,
respectively.  The optional ``buffersize`` argument can be used to split the
iteration into chunks of a given size.  The following example illustrates how
one can use this method to fill a map with a 2D Gaussian:

.. code:: python

    import numpy as np
    from astropy.coordinates import SkyCoord
    from gammapy.maps import Map

    m = Map.create(binsz=0.05, map_type='wcs', width=10.0)
    for val, coord in m.iter_by_coord(buffersize=10000):
        skydir = SkyCoord(coord[0],coord[1], unit='deg')
        sep = skydir.separation(m.geom.center_skydir).deg
        new_val = np.exp(-sep**2/2.0)
        m.set_by_coord(coord, new_val)

For maps with non-spatial dimensions the `~Map.iter_by_image` method can be used
to loop over image slices. The image plane index `idx` is returned in data order,
so that the data array can be indexed directly. Here is an example for an in-place
convolution of an image using `astropy.convolution.convolve` to interpolate NaN
values:

.. code:: python

    import numpy as np
    from astropy.convolution import convolve

    axis1 = MapAxis([1, 10, 100], interp='log', name='energy')
    axis2 = MapAxis([1, 2, 3], interp='lin', name='time')
    m = Map.create(width=(5, 3), axes=[axis1, axis2], binsz=0.1)
    m.data[:, :, 15:18, 20:25] = np.nan

    for img, idx in m.iter_by_image():
        kernel = np.ones((5, 5))
        m.data[idx] = convolve(img, kernel)

    assert not np.isnan(m.data).any()

FITS I/O
--------

Maps can be written to and read from a FITS file with the `~Map.write` and
`~Map.read` methods:

.. code:: python

    from gammapy.maps import Map

    m = Map.create(binsz=0.1, map_type='wcs', width=10.0)
    m.write('file.fits', hdu='IMAGE')
    m = Map.read('file.fits', hdu='IMAGE')

If ``map_type`` argument is not given when calling `~Map.read` a non-sparse map
object will be instantiated with the pixelization of the input HDU.

Maps can be serialized to a sparse data format by calling `~Map.write` with
``sparse=True``. This will write all non-zero pixels in the map to a data table
appropriate to the pixelization scheme.

.. code:: python

    from gammapy.maps import Map

    m = Map.create(binsz=0.1, map_type='wcs', width=10.0)
    m.write('file.fits', hdu='IMAGE', sparse=True)
    m = Map.read('file.fits', hdu='IMAGE', map_type='wcs')

Sparse maps have the same ``read`` and ``write`` methods with the exception that
they will be written to a sparse format by default:

.. code:: python

    from gammapy.maps import Map

    m = Map.create(binsz=0.1, map_type='hpx-sparse', width=10.0)
    m.write('file.fits', hdu='IMAGE')
    m = Map.read('file.fits', hdu='IMAGE', map_type='hpx-sparse')

By default files will be written to the *gamma-astro-data-format* specification
for sky maps (see `here
<http://gamma-astro-data-formats.readthedocs.io/en/latest/skymaps/index.html>`_).
The GADF format offers a number of enhancements over existing map formats such
as support for writing multi-resolution maps, sparse maps, and cubes with
different geometries to the same file.  For backward compatibility with software
using other formats, the ``conv`` keyword option is provided to write a file
using a format other than the GADF format:

.. code:: python

    from gammapy.maps import Map, MapAxis

    energy_axis = MapAxis.from_bounds(100., 1E5, 12, interp='log')
    m = Map.create(binsz=0.1, map_type='wcs', width=10.0,
                      axes=[energy_axis])
    # Write a counts cube in a format compatible with the Fermi Science Tools
    m.write('ccube.fits', conv='fgst-ccube')

Visualization
-------------

All map objects provide a ``plot`` method for generating a visualization of a
map.  This method returns figure, axes, and image objects that can be used to
further tweak/customize the image.

.. code:: python

    import matplotlib.pyplot as plt
    from gammapy.maps import Map

    m = Map.read("$GAMMAPY_DATA/fermi_2fhl/fermi_2fhl_gc.fits.gz")
    m.plot(cmap='magma', add_cbar=True)
    plt.show()


Examples
========

Creating a Counts Cube from an FT1 File
---------------------------------------

This example shows how to fill a counts cube from an FT1 file:

.. code:: python

    from gammapy.data import EventList
    from gammapy.maps import WcsGeom, WcsNDMap, MapAxis


    energy_axis = MapAxis.from_bounds(10., 2E3, 12, interp='log', name='energy', unit='GeV')
    m = WcsNDMap.create(binsz=0.1, width=10.0, skydir=(45.0,30.0),
                        coordsys='CEL', axes=[energy_axis])

    events = EventList.read('$GAMMAPY_DATA/fermi_2fhl/2fhl_events.fits.gz')

    m.fill_by_coord({'skycoord': events.radec, 'energy': events.energy})
    m.write('ccube.fits', conv='fgst-ccube')


Generating a Cutout of a Model Cube
-----------------------------------

This example shows how to extract a cut-out of LAT galactic diffuse model cube
using the `~Map.cutout` method:

.. code:: python

    from gammapy.maps import WcsGeom, WcsNDMap
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    m = WcsNDMap.read('$GAMMAPY_DATA/fermi_3fhl/gll_iem_v06_cutout.fits')
    position = SkyCoord(0, 0, frame="galactic", unit="deg")
    m_cutout = m.cutout(position=position, width=(5 * u.deg, 2 * u.deg))
    m_cutout.write('cutout.fits', conv='fgst-template')

Using `gammapy.maps`
====================

:ref:`tutorials` that show examples using ``gammapy.maps``:

* :gp-extra-notebook:`intro_maps`
* :gp-extra-notebook:`analysis_3d`
* :gp-extra-notebook:`simulate_3d`
* :gp-extra-notebook:`fermi_lat`

More detailed documentation on the WCS and HPX classes in `gammapy.maps` can be
found in the following sub-pages:

.. toctree::
    :maxdepth: 1

    hpxmap
    wcsmap

Reference/API
=============

.. automodapi:: gammapy.maps
    :include-all-objects:
