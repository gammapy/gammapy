"""
Maps
====

A thorough tutorial to work with WCS maps.

.. figure:: ../../_static/gammapy_maps.png
   :alt: Gammapy Maps Illustration

   Gammapy Maps Illustration

Introduction
------------

The `gammapy.maps` submodule contains classes for representing
pixilised data on the sky with an arbitrary number of non-spatial
dimensions such as energy, time, event class or any possible
user-defined dimension (illustrated in the image above). The main
`~gammapy.maps.Map` data structure features a uniform API for
`WCS <https://fits.gsfc.nasa.gov/fits_wcs.html>`__ as well as
`HEALPix <https://en.wikipedia.org/wiki/HEALPix>`__ based images. The
API also generalizes simple image based operations such as smoothing,
interpolation and reprojection to the arbitrary extra dimensions and
makes working with (2 + N)-dimensional hypercubes as easy as working
with a simple 2D image. Further information is also provided on the
`~gammapy.maps` docs page.

In the following introduction we will learn all the basics of working
with WCS based maps. HEALPix based maps will be covered in a future
tutorial. Make sure you have worked through the :doc:`Gammapy
overview </tutorials/starting/overview>`, because a solid knowledge
about working with `~astropy.coordinates.SkyCoord` and `~astropy.units.Quantity` objects as well as
`Numpy <http://www.numpy.org/>`__ is required for this tutorial.

This notebook is rather lengthy, but getting to know the `~gammapy.maps.Map` data
structure in detail is essential for working with Gammapy and will allow
you to fulfill complex analysis tasks with very few and simple code in
future!

"""

######################################################################
# Setup
# -----
#

import os

# %matplotlib inline
import numpy as np
from astropy import units as u
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import matplotlib.pyplot as plt
from IPython.display import display
from gammapy.data import EventList
from gammapy.maps import (
    LabelMapAxis,
    Map,
    MapAxes,
    MapAxis,
    TimeMapAxis,
    WcsGeom,
    WcsNDMap,
)

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# Creating WCS Maps
# -----------------
#
# Using Factory Methods
# ~~~~~~~~~~~~~~~~~~~~~
#
# Maps are most easily created using the `~gammapy.maps.Map.create`
# factory method:
#

m_allsky = Map.create()


######################################################################
# Calling `~gammapy.maps.Map.create` without any further arguments creates by
# default an allsky WCS map using a CAR projection, ICRS coordinates and a
# pixel size of 1 deg. This can be easily checked by printing the
# `~gammapy.maps.Map.geom` attribute of the map:
#

print(m_allsky.geom)


######################################################################
# The `~gammapy.maps.Map.geom` attribute is a `~gammapy.maps.Geom` object, that defines the basic
# geometry of the map, such as size of the pixels, width and height of the
# image, coordinate system etc., but we will learn more about this object
# later.
#
# Besides the ``.geom`` attribute the map has also a ``.data`` attribute,
# which is just a plain `numpy.ndarray` and stores the data associated
# with this map:
#

print(m_allsky.data)


######################################################################
# By default maps are filled with zeros.
#
# The ``map_type`` argument can be used to control the pixelization scheme
# (WCS or HPX).
#

position = SkyCoord(0.0, 5.0, frame="galactic", unit="deg")

# Create a WCS Map
m_wcs = Map.create(binsz=0.1, map_type="wcs", skydir=position, width=10.0)

# Create a HPX Map
m_hpx = Map.create(binsz=0.1, map_type="hpx", skydir=position, width=10.0)


######################################################################
# Here is an example that creates a WCS map centered on the Galactic
# center and now uses Galactic coordinates:
#

skydir = SkyCoord(0, 0, frame="galactic", unit="deg")
m_gc = Map.create(
    binsz=0.02, width=(10, 5), skydir=skydir, frame="galactic", proj="TAN"
)
print(m_gc.geom)


######################################################################
# In addition we have defined a TAN projection, a pixel size of ``0.02``
# deg and a width of the map of ``10 deg x 5 deg``. The ``width`` argument
# also takes scalar value instead of a tuple, which is interpreted as both
# the width and height of the map, so that a quadratic map is created.
#


######################################################################
# Creating from a Map Geometry
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As we have seen in the first examples, the `~gammapy.maps.Map` object couples the
# data (stored as a `~numpy.ndarray`) with a `~gammapy.maps.Geom` object. The
# `~gammapy.maps.Geom` object can be seen as a generalization of an
# `astropy.wcs.WCS` object, providing the information on how the data
# maps to physical coordinate systems. In some cases e.g. when creating
# many maps with the same WCS geometry it can be advantageous to first
# create the map geometry independent of the map object it-self:
#

wcs_geom = WcsGeom.create(binsz=0.02, width=(10, 5), skydir=(0, 0), frame="galactic")


######################################################################
# And then create the map objects from the ``wcs_geom`` geometry
# specification:
#

maps = {}

for name in ["counts", "background"]:
    maps[name] = Map.from_geom(wcs_geom)


######################################################################
# The `~gammapy.maps.Geom` object also has a few helpful methods. E.g. we can check
# whether a given position on the sky is contained in the map geometry:
#

# define the position of the Galactic center and anti-center
positions = SkyCoord([0, 180], [0, 0], frame="galactic", unit="deg")
wcs_geom.contains(positions)


######################################################################
# Or get the image center of the map:
#

print(wcs_geom.center_skydir)


######################################################################
# Or we can also retrieve the solid angle per pixel of the map:
#

print(wcs_geom.solid_angle())


######################################################################
# Adding Non-Spatial Axes
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# In many analysis scenarios we would like to add extra dimension to the
# maps to study e.g. energy or time dependency of the data. Those
# non-spatial dimensions are handled with the `~gammapy.maps.MapAxis` object. Let us
# first define an energy axis, with 4 bins:
#

energy_axis = MapAxis.from_bounds(
    1, 100, nbin=4, unit="TeV", name="energy", interp="log"
)
print(energy_axis)


######################################################################
# Where ``interp='log'`` specifies that a logarithmic spacing is used
# between the bins, equivalent to ``np.logspace(0, 2, 4)``. This
# `~gammapy.maps.MapAxis` object we can now pass to `~gammapy.maps.Map.create()` using the
# ``axes=`` argument:
#

m_cube = Map.create(binsz=0.02, width=(10, 5), frame="galactic", axes=[energy_axis])
print(m_cube.geom)


######################################################################
# Now we see that besides ``lon`` and ``lat`` the map has an additional
# axes named ``energy`` with 4 bins. The total dimension of the map is now
# ``ndim=3``.
#
# We can also add further axes by passing a list of `~gammapy.maps.MapAxis` objects.
# To demonstrate this we create a time axis with linearly spaced bins and
# pass both axes to `~gammapy.maps.Map.create()`:
#

time_axis = MapAxis.from_bounds(0, 24, nbin=24, unit="hour", name="time", interp="lin")

m_4d = Map.create(
    binsz=0.02, width=(10, 5), frame="galactic", axes=[energy_axis, time_axis]
)
print(m_4d.geom)


######################################################################
# The `~gammapy.maps.MapAxis` object internally stores the coordinates or “position
# values” associated with every map axis bin or “node”. We distinguish
# between two node types: ``"edges"`` and ``"center"``. The node type
# ``"edges"`` (which is also the default) specifies that the data
# associated with this axis is integrated between the edges of the bin
# (e.g. counts data). The node type ``"center"`` specifies that the data is
# given at the center of the bin (e.g. exposure or differential fluxes).
#
# The edges of the bins can be checked with `~gammapy.maps.MapAxis.edges` attribute:
#

print(energy_axis.edges)


######################################################################
# The numbers are given in the units we specified above, which can be
# checked again with:
#

print(energy_axis.unit)


######################################################################
# The centers of the axis bins can be checked with the `~gammapy.maps.MapAxis.center`
# attribute:
#

print(energy_axis.center)

######################################################################
# Adding Non-contiguous axes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Non-spatial map axes can also be handled through two other objects known as the `~gammapy.maps.TimeMapAxis`
# and the `~gammapy.maps.LabelMapAxis`.
#


######################################################################
# TimeMapAxis
# ^^^^^^^^^^^
#
# The `~gammapy.maps.TimeMapAxis` object provides an axis for non-adjacent
# time intervals.
#

time_map_axis = TimeMapAxis(
    edges_min=[1, 5, 10, 15] * u.day,
    edges_max=[2, 7, 13, 18] * u.day,
    reference_time=Time("2020-03-19"),
)

print(time_map_axis)

######################################################################
# This ``time_map_axis`` can then be utilised in a similar way to the previous implementation to create
# a `~gammapy.maps.Map`.
#

map_4d = Map.create(
    binsz=0.02, width=(10, 5), frame="galactic", axes=[energy_axis, time_map_axis]
)
print(map_4d.geom)

######################################################################
# It is possible to utilise the `~gammapy.maps.TimeMapAxis.slice` attrribute
# to create new a `~gammapy.maps.TimeMapAxis`. Here we are slicing
# between the first and third axis to extract the subsection of the axis
# between indice 0 and 2.

print(time_map_axis.slice([0, 2]))

######################################################################
# It is also possible to `~gammapy.maps.TimeMapAxis.squash` the axis,
# which squashes the existing axis into one bin. This creates a new axis
# between the extreme edges of the initial axis.

print(time_map_axis.squash())


######################################################################
# The `~gammapy.maps.TimeMapAxis.is_contiguous` method returns a boolean
# which indicates whether the `~gammapy.maps.TimeMapAxis` is contiguous or not.

print(time_map_axis.is_contiguous)

######################################################################
# As we have a non-contiguous axis we can print the array of bin edges for both
# the minimum axis edges (`~gammapy.maps.TimeMapAxis.edges_min`) and the maximum axis
# edges (`~gammapy.maps.TimeMapAxis.edges_max`).

print(time_map_axis.edges_min)

print(time_map_axis.edges_max)

######################################################################
# Next, we use the `~gammapy.maps.TimeMapAxis.to_contiguous` functionality to
# create a contiguous axis and expose `~gammapy.maps.TimeMapAxis.edges`. This
# method returns a `~astropy.units.Quantity` with respect to the reference time.

time_map_axis_contiguous = time_map_axis.to_contiguous()

print(time_map_axis_contiguous.is_contiguous)

print(time_map_axis_contiguous.edges)


######################################################################
# The `~gammapy.maps.TimeMapAxis.time_edges` will return the `~astropy.time.Time` object directly

print(time_map_axis_contiguous.time_edges)


######################################################################
# `~gammapy.maps.TimeMapAxis` also has both functionalities of
# `~gammapy.maps.TimeMapAxis.coord_to_pix` and `~gammapy.maps.TimeMapAxis.coord_to_idx`.
# The `~gammapy.maps.TimeMapAxis.coord_to_idx` attribute will give the index of the
# ``time`` specified, similarly for `~gammapy.maps.TimeMapAxis.coord_to_pix` which returns
# the pixel. A linear interpolation is assumed.
#
# Start by choosing a time which we know is within the `~gammapy.maps.TimeMapAxis` and see the results.


time = Time(time_map_axis.time_max.mjd[0], format="mjd")

print(time_map_axis.coord_to_pix(time))

print(time_map_axis.coord_to_idx(time))


######################################################################
# This functionality can also be used with an array of `~astropy.time.Time` values.

times = Time(time_map_axis.time_max.mjd, format="mjd")

print(time_map_axis.coord_to_pix(times))

print(time_map_axis.coord_to_idx(times))

######################################################################
# Note here we take a `~astropy.time.Time` which is outside the edges.
# A linear interpolation is assumed for both methods, therefore for a time
# outside the ``time_map_axis`` there is no extrapolation and -1 is returned.
#
# Note: due to this, the `~gammapy.maps.TimeMapAxis.coord_to_pix` method will
# return ``nan`` and the `~gammapy.maps.TimeMapAxis.coord_to_idx` method returns -1.

print(time_map_axis.coord_to_pix(Time(time.mjd + 1, format="mjd")))

print(time_map_axis.coord_to_idx(Time(time.mjd + 1, format="mjd")))


######################################################################
# LabelMapAxis
# ^^^^^^^^^^^^
#
# The `~gammapy.maps.LabelMapAxis` object allows for handling of labels for map axes.
# It provides an axis for non-numeric entries.
#

label_axis = LabelMapAxis(
    labels=["dataset-1", "dataset-2", "dataset-3"], name="dataset"
)

print(label_axis)

######################################################################
# The labels can be checked using the `~gammapy.maps.LabelMapAxis.center` attribute:
print(label_axis.center)


######################################################################
# To obtain the position of the label, one can utilise the `~gammapy.maps.LabelMapAxis.coord_to_pix` attribute

print(label_axis.coord_to_pix(["dataset-3"]))

######################################################################
# To adapt and create new axes the following attributes can be utilised:
# `~gammapy.maps.LabelMapAxis.concatenate`, `~gammapy.maps.LabelMapAxis.slice` and
# `~gammapy.maps.LabelMapAxis.squash`.
#
# Combining two different `~gammapy.maps.LabelMapAxis` is done in the following way:

label_axis2 = LabelMapAxis(labels=["dataset-a", "dataset-b"], name="dataset")

print(label_axis.concatenate(label_axis2))

######################################################################
# A new `~gammapy.maps.LabelMapAxis` can be created by slicing an already existing one.
# Here we are slicing between the second and third bins to extract the subsection.
print(label_axis.slice([1, 2]))

######################################################################
# A new axis object can be created by squashing the axis into a single bin.

print(label_axis.squash())


######################################################################
# Mixing the three previous axis types (`~gammapy.maps.MapAxis`,
# `~gammapy.maps.TimeMapAxis` and `~gammapy.maps.LabelMapAxis`)
# would be done like so:
#

axes = MapAxes(axes=[energy_axis, time_map_axis, label_axis])
hdu = axes.to_table_hdu(format="gadf")
table = Table.read(hdu)
display(table)


######################################################################
# Reading and Writing
# -------------------
#
# Gammapy `~gammapy.maps.Map` objects are serialized using the Flexible Image
# Transport Format (FITS). Depending on the pixelization scheme (HEALPix
# or WCS) and presence of non-spatial dimensions the actual convention to
# write the FITS file is different. By default Gammapy uses a generic
# convention named ``"gadf"``, which will support WCS and HEALPix formats as
# well as an arbitrary number of non-spatial axes. The convention is
# documented in detail on the `Gamma Astro Data
# Formats <https://gamma-astro-data-formats.readthedocs.io/en/latest/skymaps/index.html>`__
# page.
#
# Other conventions required by specific software (e.g. the Fermi Science
# Tools) are supported as well. At the moment those are the following
#
# -  ``"fgst-ccube"``: Fermi counts cube format.
# -  ``"fgst-ltcube"``: Fermi livetime cube format.
# -  ``"fgst-bexpcube"``: Fermi exposure cube format
# -  ``"fgst-template"``: Fermi Galactic diffuse and source template format.
# -  ``"fgst-srcmap"`` and ``"fgst-srcmap-sparse"``: Fermi source map and
#    sparse source map format.
#
# The conventions listed above only support an additional energy axis.
#
# Reading Maps
# ~~~~~~~~~~~~
#
# Reading FITS files is mainly exposed via the `~gammapy.maps.Map.read()` method. Let
# us take a look at a first example:
#

filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz"
m_3fhl_gc = Map.read(filename)
print(m_3fhl_gc)


######################################################################
# If ``map_type`` argument is not given when calling read a map object
# will be instantiated with the pixelization of the input HDU.
#
# By default ``Map.read()`` will try to find the first valid data hdu in
# the filename and read the data from there. If multiple HDUs are present
# in the FITS file, the desired one can be chosen with the additional
# `hdu=` argument:
#

m_3fhl_gc = Map.read(filename, hdu="PRIMARY")
print(m_3fhl_gc)


######################################################################
# In rare cases e.g. when the FITS file is not valid or meta data is
# missing from the header it can be necessary to modify the header of a
# certain HDU before creating the `~gammapy.maps.Map` object. In this case we can use
# `astropy.io.fits` directly to read the FITS file:
#

filename = os.environ["GAMMAPY_DATA"] + "/fermi-3fhl-gc/fermi-3fhl-gc-exposure.fits.gz"
hdulist = fits.open(filename)
print(hdulist.info())


######################################################################
# And then modify the header keyword and use `~gammapy.maps.from_hdulist()` to
# create the `~gammapy.maps.Map` object after:
#

hdulist["PRIMARY"].header["BUNIT"] = "cm2 s"
print(Map.from_hdulist(hdulist=hdulist))


######################################################################
# Writing Maps
# ~~~~~~~~~~~~
#
# Writing FITS files on disk via the `~gammapy.maps.Map.write()` method.
# Here is a first example:
#

m_cube.write("example_cube.fits", overwrite=True)


######################################################################
# By default Gammapy does not overwrite files. In this example we set
# ``overwrite=True`` in case the cell gets executed multiple times. Now we
# can read back the cube from disk using `~gammapy.maps.Map.read()`:
#

m_cube = Map.read("example_cube.fits")
print(m_cube)


######################################################################
# We can also choose a different FITS convention to write the example cube
# in a format compatible to the Fermi Galactic diffuse background model:
#

m_cube.write("example_cube_fgst.fits", format="fgst-template", overwrite=True)


######################################################################
# To understand a little bit better the generic ``gadf`` convention we use
# `~gammapy.maps.Map.to_hdulist()` to generate a list of FITS HDUs first:
#

hdulist = m_4d.to_hdulist(format="gadf")
print(hdulist.info())


######################################################################
# As we can see the ``HDUList`` object contains to HDUs. The first one
# named ``PRIMARY`` contains the data array with shape corresponding to
# our data and the WCS information stored in the header:
#

print(hdulist["PRIMARY"].header)


######################################################################
# The second HDU is a ``BinTableHDU`` named ``PRIMARY_BANDS`` contains the
# information on the non-spatial axes such as name, order, unit, min, max
# and center values of the axis bins. We use an `astropy.table.Table` to
# show the information:
#

print(Table.read(hdulist["PRIMARY_BANDS"]))


######################################################################
# Maps can be serialized to a sparse data format by calling write with
# ``sparse=True``. This will write all non-zero pixels in the map to a
# data table appropriate to the pixelization scheme.
#

m = Map.create(binsz=0.1, map_type="wcs", width=10.0)
m.write("file.fits", hdu="IMAGE", sparse=True, overwrite=True)
m = Map.read("file.fits", hdu="IMAGE", map_type="wcs")


######################################################################
# Accessing Data
# --------------
#
# How to get data values
# ~~~~~~~~~~~~~~~~~~~~~~
#
# All map objects have a set of accessor methods, which can be used to
# access or update the contents of the map irrespective of its underlying
# representation. Those accessor methods accept as their first argument a
# coordinate `tuple` containing scalars, `list`, or `numpy.ndarray`
# with one tuple element for each dimension. Some methods additionally
# accept a `dict` or `~gammapy.maps.MapCoord` argument, of which both allow to
# assign coordinates by axis name.
#
# Let us first begin with the `~gammapy.maps.Map.get_by_idx()` method, that accepts a
# tuple of indices. The order of the indices corresponds to the axis order
# of the map:
#

print(m_gc.get_by_idx((50, 30)))


######################################################################
# **Important:** Gammapy uses a reversed index order in the map API with
# the longitude axes first. To achieve the same by directly indexing into
# the numpy array we have to call:
#

print(m_gc.data[([30], [50])])


######################################################################
# To check the order of the axes you can always print the ``.geom``
# attribute:
#

print(m_gc.geom)


######################################################################
# To access values directly by sky coordinates we can use the
# `~gammapy.maps.Map.get_by_coord()` method. This time we pass in a `dict`, specifying
# the axes names corresponding to the given coordinates:
#

print(m_gc.get_by_coord({"lon": [0, 180], "lat": [0, 0]}))


######################################################################
# The units of the coordinates are assumed to be in degrees in the
# coordinate system used by the map. If the coordinates do not correspond
# to the exact pixel center, the value of the nearest pixel center will be
# returned. For positions outside the map geometry `numpy.nan` is returned.
#
# The coordinate or idx arrays follow normal `Numpy broadcasting
# rules <https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html>`__.
# So the following works as expected:
#

lons = np.linspace(-4, 4, 10)
print(m_gc.get_by_coord({"lon": lons, "lat": 0}))


######################################################################
# Or as an even more advanced example, we can provide ``lats`` as column
# vector and broadcasting to a 2D result array will be applied:
#

lons = np.linspace(-4, 4, 8)
lats = np.linspace(-4, 4, 8).reshape(-1, 1)
print(m_gc.get_by_coord({"lon": lons, "lat": lats}))


######################################################################
# Indexing and Slicing Sub-Maps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When you have worked with Numpy arrays in the past you are probably
# familiar with the concept of indexing and slicing into data arrays. To
# support slicing of non-spatial axes of `~gammapy.maps.Map` objects, the `~gammapy.maps.Map`
# object has a `~gammapy.maps.Map.slice_by_idx()` method, which allows to extract
# sub-maps from a larger map.
#
# The following example demonstrates how to get the map at the energy bin
# number 3:
#

m_sub = m_cube.slice_by_idx({"energy": 3})
print(m_sub)


######################################################################
# Note that the returned object is again a `~gammapy.maps.Map` with updated axes
# information. In this case, because we extracted only a single image, the
# energy axes is dropped from the map.
#
# To extract a sub-cube with a sliced energy axes we can use a normal
# ``slice`` object:
#

m_sub = m_cube.slice_by_idx({"energy": slice(1, 3)})
print(m_sub)


######################################################################
# Note that the returned object is also a `~gammapy.maps.Map` object, but this time
# with updated energy axis specification.
#
# Slicing of multiple dimensions is supported by adding further entries to
# the dict passed to `~gammapy.maps.Map.slice_by_idx()`
#

m_sub = m_4d.slice_by_idx({"energy": slice(1, 3), "time": slice(4, 10)})
print(m_sub)


######################################################################
# For convenience there is also a `~gammapy.maps.Map.get_image_by_coord()` method which
# allows to access image planes at given non-spatial physical coordinates.
# This method also supports `~astropy.units.Quantity` objects:
#

image = m_4d.get_image_by_coord({"energy": 4 * u.TeV, "time": 5 * u.h})
print(image.geom)


######################################################################
# Iterating by image
# ~~~~~~~~~~~~~~~~~~
#
# For maps with non-spatial dimensions the `~gammapy.maps.Map.iter_by_image_data`
# method can be used to loop over image slices. The image plane index
# ``idx`` is returned in data order, so that the data array can be indexed
# directly. Here is an example for an in-place convolution of an image
# using `~astropy.convolution.convolve` to interpolate NaN values:
#

axis1 = MapAxis([1, 10, 100], interp="log", name="energy")
axis2 = MapAxis([1, 2, 3], interp="lin", name="time")
m = Map.create(width=(5, 3), axes=[axis1, axis2], binsz=0.1)
m.data[:, :, 15:18, 20:25] = np.nan

for img, idx in m.iter_by_image_data():
    kernel = np.ones((5, 5))
    m.data[idx] = convolve(img, kernel)

assert not np.isnan(m.data).any()


######################################################################
# Modifying Data
# --------------
#
# How to set data values
# ~~~~~~~~~~~~~~~~~~~~~~
#
# To modify and set map data values the `~gammapy.maps.Map` object features as well a
# `~gammapy.maps.Map.set_by_idx()` method:
#

m_cube.set_by_idx(idx=(10, 20, 3), vals=42)


######################################################################
# here we check that data have been updated:
#

print(m_cube.get_by_idx((10, 20, 3)))


######################################################################
# Of course there is also a `~gammapy.maps.Map.set_by_coord()` method, which allows to
# set map data values in physical coordinates.
#

m_cube.set_by_coord({"lon": 0, "lat": 0, "energy": 2 * u.TeV}, vals=42)


######################################################################
# Again the ``lon`` and ``lat`` values are assumed to be given in degrees
# in the coordinate system used by the map. For the energy axis, the unit
# is the one specified on the axis (use ``m_cube.geom.axes[0].unit`` to
# check if needed…).
#
# All ``.xxx_by_coord()`` methods accept `~astropy.coordinates.SkyCoord` objects as well. In
# this case we have to use the ``"skycoord"`` keyword instead of ``"lon"`` and
# ``"lat"``:
#

skycoords = SkyCoord([1.2, 3.4], [-0.5, 1.1], frame="galactic", unit="deg")
m_cube.set_by_coord({"skycoord": skycoords, "energy": 2 * u.TeV}, vals=42)


######################################################################
# Filling maps from event lists
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This example shows how to fill a counts cube from an event list:
#

energy_axis = MapAxis.from_bounds(
    10.0, 2e3, 12, interp="log", name="energy", unit="GeV"
)
counts_3d = WcsNDMap.create(
    binsz=0.1, width=10.0, skydir=(0, 0), frame="galactic", axes=[energy_axis]
)

events = EventList.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz")

counts_3d.fill_by_coord({"skycoord": events.radec, "energy": events.energy})
counts_3d.write("ccube.fits", format="fgst-ccube", overwrite=True)


######################################################################
# Alternatively you can use the `~gammapy.maps.Map.fill_events` method:
#

counts_3d = WcsNDMap.create(
    binsz=0.1, width=10.0, skydir=(0, 0), frame="galactic", axes=[energy_axis]
)

counts_3d.fill_events(events)


######################################################################
# If you have a given map already, and want to make a counts image with
# the same geometry (not using the pixel data from the original map), you
# can also use the `~gammapy.maps.Map.fill_events` method.
#

events = EventList.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz")
reference_map = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz")
counts = Map.from_geom(reference_map.geom)
counts.fill_events(events)


######################################################################
# It works for IACT and Fermi-LAT events, for WCS or HEALPix map
# geometries, and also for extra axes. Especially energy axes are
# automatically handled correctly.
#


######################################################################
# Filling maps from interpolation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Maps support interpolation via the `~~gammapy.maps.Map.interp_by_coord` and
# `~~gammapy.maps.Map.interp_by_pix` methods. Currently, the following interpolation
# methods are supported:
#
# -  ``"nearest"`` : Return value of nearest pixel (no interpolation).
# -  ``"linear"`` : Interpolation with first order polynomial. This is the
#    only interpolation method that is supported for all map types.
# -  ``quadratic`` : Interpolation with second order polynomial.
# -  ``cubic`` : Interpolation with third order polynomial.
#
# Note that ``"quadratic"`` and ``"cubic"`` interpolation are currently only
# supported for WCS-based maps with regular geometry (e.g. 2D or ND with
# the same geometry in every image plane). ``"linear"`` and higher order
# interpolation by pixel coordinates is only supported for WCS-based maps.
#
# In the following example we create a new map and fill it by
# interpolating another map:
#

# read map
filename = "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz"
m_iem_gc = Map.read(filename)

# create new geometry
skydir = SkyCoord(266.4, -28.9, frame="icrs", unit="deg")
wcs_geom_cel = WcsGeom.create(skydir=skydir, binsz=0.1, frame="icrs", width=(8, 4))

# create new empty map from geometry
m_iem_10GeV = Map.from_geom(wcs_geom_cel)
coords = m_iem_10GeV.geom.get_coord()

# fill new map using interpolation
m_iem_10GeV.data = m_iem_gc.interp_by_coord(
    {"skycoord": coords.skycoord, "energy_true": 10 * u.GeV},
    method="linear",
    fill_value=np.nan,
)


######################################################################
# Interpolating onto a different geometry
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# For 3d geometries this operation can be performed directly using the
# `~gammapy.maps.Map.interp_to_geom()` method. This is very useful, ex: while using map
# arithmetic.
#

# create new geometry
energy_axis = MapAxis.from_bounds(
    10.0, 2e3, 6, interp="log", name="energy_true", unit="GeV"
)
skydir = SkyCoord(266.4, -28.9, frame="icrs", unit="deg")
wcs_geom_3d = WcsGeom.create(
    skydir=skydir, binsz=0.1, frame="icrs", width=(8, 4), axes=[energy_axis]
)

# create the interpolated map
m_iem_interp = m_iem_gc.interp_to_geom(
    wcs_geom_3d, preserve_counts=False, method="linear", fill_value=np.nan
)
print(m_iem_interp)


######################################################################
# Note that ``preserve_counts=`` option should be true if the map is an
# integral quantity (e.g. counts) and false if the map is a differential
# quantity (e.g. intensity).
#


######################################################################
# Maps operations
# ---------------
#
# Basic operators
# ~~~~~~~~~~~~~~~
#
# One can perform simple arithmetic on maps using the ``+``, ``-``, ``*``,
# ``/`` operators, this works only for maps with the same geometry:
#

iem_plus_iem = m_iem_10GeV + m_iem_10GeV

iem_minus_iem = m_iem_10GeV - m_iem_10GeV


######################################################################
# These operations can be applied between a Map and a scalar in that
# specific order:
#

iem_times_two = m_iem_10GeV * 2
# iem_times_two = 2 * m_iem_10GeV # this won't work


######################################################################
# The logic operators can also be applied on maps (the result is a map of
# boolean type):
#

is_null = iem_minus_iem == 0
print(is_null)


######################################################################
# Here we check that the result is ``True`` for all the well-defined
# pixels (not ``NaN``):
#

print(np.all(is_null.data[~np.isnan(iem_minus_iem)]))


######################################################################
# Cutouts
# ~~~~~~~
#
# The `~gammapy.maps.WcsNDMap` objects features a `~gammapy.maps.WcsNDMap.cutout()` method, which allows
# you to cut out a smaller part of a larger map. This can be useful,
# e.g. when working with all-sky diffuse maps. Here is an example:
#

position = SkyCoord(0, 0, frame="galactic", unit="deg")
m_iem_cutout = m_iem_gc.cutout(position=position, width=(4 * u.deg, 2 * u.deg))


######################################################################
# The returned object is again a `~gammapy.maps.Map` object with updated WCS
# information and data size. As one can see the cutout is automatically
# applied to all the non-spatial axes as well. The cutout width is given
# in the order of ``(lon, lat)`` and can be specified with units that will
# be handled correctly.
#


######################################################################
# Visualizing and Plotting
# ------------------------
#
# All map objects provide a `~gammapy.maps.Map.plot()` method for generating a visualization
# of a map. This method returns figure, axes, and image objects that can
# be used to further tweak/customize the image. The `~gammapy.maps.Map.plot()` method should
# be used with 2D maps, while 3D maps can be displayed with the
# `~gammapy.maps.Map.plot_interactive()` or `~gammapy.maps.Map.plot_grid()` methods.
#
# Image Plotting
# ~~~~~~~~~~~~~~
#
# For debugging and inspecting the map data it is useful to plot or
# visualize the images planes contained in the map.
#

filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz"
m_3fhl_gc = Map.read(filename)


######################################################################
# After reading the map we can now plot it on the screen by calling the
# ``.plot()`` method:
#

m_3fhl_gc.plot()
plt.show()


######################################################################
# We can easily improve the plot by calling `~gammapy.maps.Map.smooth()` first and
# providing additional arguments to `~gammapy.maps.Map.plot()`. Most of them are passed
# further to
# `plt.imshow() <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html>`__:
#

smoothed = m_3fhl_gc.smooth(width=0.2 * u.deg, kernel="gauss")
smoothed.plot(stretch="sqrt", add_cbar=True, vmax=4, cmap="inferno")
plt.show()


######################################################################
# We can use the
# `plt.rc_context() <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.rc_context.html>`__
# context manager to further tweak the plot by adapting the figure and
# font size:
#

rc_params = {"figure.figsize": (12, 5.4), "font.size": 12}
with plt.rc_context(rc=rc_params):
    smoothed = m_3fhl_gc.smooth(width=0.2 * u.deg, kernel="gauss")
    smoothed.plot(stretch="sqrt", add_cbar=True, vmax=4)
plt.show()


######################################################################
# Cube plotting
# ~~~~~~~~~~~~~
#
# For maps with non-spatial dimensions the `~gammapy.maps.Map` object features an
# interactive plotting method, that works in jupyter notebooks only (Note:
# it requires the package ``ipywidgets`` to be installed). We first read a
# small example cutout from the Fermi Galactic diffuse model and display
# the data cube by calling `~gammapy.maps.Map.plot_interactive()`:
#

rc_params = {
    "figure.figsize": (12, 5.4),
    "font.size": 12,
    "axes.formatter.limits": (2, -2),
}
m_iem_gc.plot_interactive(add_cbar=True, stretch="sqrt", rc_params=rc_params)
plt.show()


######################################################################
# Now you can use the interactive slider to select an energy range and the
# corresponding image is displayed on the screen. You can also use the
# radio buttons to select your preferred image stretching. We have passed
# additional keywords using the ``rc_params`` argument to improve the
# figure and font size. Those keywords are directly passed to the
# `plt.rc_context() <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.rc_context.html>`__
# context manager.
#
# Additionally, all the slices of a 3D `~gammapy.maps.Map` can be displayed using the
# `~gammapy.maps.Map.plot_grid()` method. By default the colorbars bounds of the subplots
# are not the same, we can make them consistent using the ``vmin`` and
# ``vmax`` options:
#

counts_3d.plot_grid(ncols=4, figsize=(16, 12), vmin=0, vmax=100, stretch="log")
plt.show()
