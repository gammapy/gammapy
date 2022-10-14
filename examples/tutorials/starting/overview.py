"""
Data structures
===============

Introduction to basic data structures handling.

Introduction
------------

This is a getting started tutorial for Gammapy.

In this tutorial we will use the `Second Fermi-LAT Catalog of
High-Energy Sources (3FHL)
catalog <http://fermi.gsfc.nasa.gov/ssc/data/access/lat/3FHL/>`__,
corresponding event list and images to learn how to work with some of
the central Gammapy data structures.

We will cover the following topics:

-  **Sky maps**

   -  We will learn how to handle image based data with gammapy using a
      Fermi-LAT 3FHL example image. We will work with the following
      classes:

      -  `~gammapy.maps.WcsNDMap`
      -  `~astropy.coordinates.SkyCoord`
      -  `~numpy.ndarray`

-  **Event lists**

   -  We will learn how to handle event lists with Gammapy. Important
      for this are the following classes:

      -  `~gammapy.data.EventList`
      -  `~astropy.table.Table`

-  **Source catalogs**

   -  We will show how to load source catalogs with Gammapy and explore
      the data using the following classes:

      -  `~gammapy.catalog.SourceCatalog`, specifically
         `~gammapy.catalog.SourceCatalog3FHL`
      -  `~astropy.table.Table`

-  **Spectral models and flux points**

   -  We will pick an example source and show how to plot its spectral
      model and flux points. For this we will use the following classes:

      -  `~gammapy.modeling.models.SpectralModel`, specifically the
         `~gammapy.modeling.models.PowerLaw2SpectralModel`
      -  `~gammapy.estimators.FluxPoints`
      -  `~astropy.table.Table`

"""


######################################################################
# Setup
# -----
#
# **Important**: to run this tutorial the environment variable
# ``GAMMAPY_DATA`` must be defined and point to the directory on your
# machine where the datasets needed are placed. To check whether your
# setup is correct you can execute the following cell:
#


import astropy.units as u
from astropy.coordinates import SkyCoord

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

# %matplotlib inline


check_tutorials_setup()


######################################################################
# Maps
# ----
#
# The `~gammapy.maps` package contains classes to work with sky images
# and cubes.
#
# In this section, we will use a simple 2D sky image and will learn how
# to:
#
# -  Read sky images from FITS files
# -  Smooth images
# -  Plot images
# -  Cutout parts from images
#

from gammapy.maps import Map

gc_3fhl = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz")


######################################################################
# The image is a `~gammapy.maps.WcsNDMap` object:
#

print(gc_3fhl)


######################################################################
# The shape of the image is 400 x 200 pixel and it is defined using a
# cartesian projection in galactic coordinates.
#
# The ``geom`` attribute is a `~gammapy.maps.WcsGeom` object:
#

print(gc_3fhl.geom)


######################################################################
# Let’s take a closer look a the ``.data`` attribute:
#

print(gc_3fhl.data)


######################################################################
# That looks familiar! It just an *ordinary* 2 dimensional numpy array,
# which means you can apply any known numpy method to it:
#

print(f"Total number of counts in the image: {gc_3fhl.data.sum():.0f}")


######################################################################
# To show the image on the screen we can use the ``plot`` method. It
# basically calls
# `plt.imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`__,
# passing the ``gc_3fhl.data`` attribute but in addition handles axis with
# world coordinates using
# `astropy.visualization.wcsaxes <https://docs.astropy.org/en/stable/visualization/wcsaxes/>`__
# and defines some defaults for nicer plots (e.g. the colormap ‘afmhot’):
#

gc_3fhl.plot(stretch="sqrt")


######################################################################
# To make the structures in the image more visible we will smooth the data
# using a Gaussian kernel.
#

gc_3fhl_smoothed = gc_3fhl.smooth(kernel="gauss", width=0.2 * u.deg)

gc_3fhl_smoothed.plot(stretch="sqrt")


######################################################################
# The smoothed plot already looks much nicer, but still the image is
# rather large. As we are mostly interested in the inner part of the
# image, we will cut out a quadratic region of the size 9 deg x 9 deg
# around Vela. Therefore we use `~gammapy.maps.Map.cutout` to make a
# cutout map:
#

# define center and size of the cutout region
center = SkyCoord(0, 0, unit="deg", frame="galactic")
gc_3fhl_cutout = gc_3fhl_smoothed.cutout(center, 9 * u.deg)
gc_3fhl_cutout.plot(stretch="sqrt")


######################################################################
# For a more detailed introduction to `~gammapy.maps`, take a look a the
# `maps.ipynb <../api/maps.ipynb>`__ notebook.
#
# Exercises
# ~~~~~~~~~
#
# -  Add a marker and circle at the position of ``Sag A*`` (you can find
#    examples in
#    `astropy.visualization.wcsaxes <https://docs.astropy.org/en/stable/visualization/wcsaxes/>`__).
#


######################################################################
# Event lists
# -----------
#
# Almost any high level gamma-ray data analysis starts with the raw
# measured counts data, which is stored in event lists. In Gammapy event
# lists are represented by the ``~gammapy.data.EventList`` class.
#
# In this section we will learn how to:
#
# -  Read event lists from FITS files
# -  Access and work with the ``EventList`` attributes such as ``.table``
#    and ``.energy``
# -  Filter events lists using convenience methods
#
# Let’s start with the import from the ``~gammapy.data`` submodule:
#

from gammapy.data import EventList

######################################################################
# Very similar to the sky map class an event list can be created, by
# passing a filename to the ``~gammapy.data.EventList.read()`` method:
#

events_3fhl = EventList.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz")


######################################################################
# This time the actual data is stored as an
# `~astropy.table.Table `
# object. It can be accessed with ``.table`` attribute:
#

print(events_3fhl.table)


######################################################################
# You can do *len* over event_3fhl.table to find the total number of
# events.
#

print(len(events_3fhl.table))


######################################################################
# And we can access any other attribute of the ``Table`` object as well:
#

print(events_3fhl.table.colnames)


######################################################################
# For convenience we can access the most important event parameters as
# properties on the ``EventList`` objects. The attributes will return
# corresponding Astropy objects to represent the data, such as
# `~astropy.units.Quantity`,
# `~astropy.coordinates.SkyCoord`
# or
# `~astropy.time.Time`
# objects:
#

print(events_3fhl.energy.to("GeV"))

print(events_3fhl.galactic)
# events_3fhl.radec

print(events_3fhl.time)


######################################################################
# In addition ``EventList`` provides convenience methods to filter the
# event lists. One possible use case is to find the highest energy event
# within a radius of 0.5 deg around the vela position:
#

# select all events within a radius of 0.5 deg around center
from gammapy.utils.regions import SphericalCircleSkyRegion

region = SphericalCircleSkyRegion(center, radius=0.5 * u.deg)
events_gc_3fhl = events_3fhl.select_region(region)

# sort events by energy
events_gc_3fhl.table.sort("ENERGY")

# and show highest energy photon
print("highest energy photon: ", events_gc_3fhl.energy[-1].to("GeV"))


######################################################################
# Exercises
# ~~~~~~~~~
#
# -  Make a counts energy spectrum for the galactic center region, within
#    a radius of 10 deg.
#


######################################################################
# Source catalogs
# ---------------
#
# Gammapy provides a convenient interface to access and work with catalog
# based data.
#
# In this section we will learn how to:
#
# -  Load builtins catalogs from `~gammapy.catalog`
# -  Sort and index the underlying Astropy tables
# -  Access data from individual sources
#
# Let’s start with importing the 3FHL catalog object from the
# ``~gammapy.catalog`` submodule:
#

from gammapy.catalog import SourceCatalog3FHL

######################################################################
# First we initialize the Fermi-LAT 3FHL catalog and directly take a look
# at the ``.table`` attribute:
#

fermi_3fhl = SourceCatalog3FHL()
print(fermi_3fhl.table)


######################################################################
# This looks very familiar again. The data is just stored as an
# `~astropy.table.Table`
# object. We have all the methods and attributes of the ``Table`` object
# available. E.g. we can sort the underlying table by ``Signif_Avg`` to
# find the top 5 most significant sources:
#

# sort table by significance
fermi_3fhl.table.sort("Signif_Avg")

# invert the order to find the highest values and take the top 5
top_five_TS_3fhl = fermi_3fhl.table[::-1][:5]

# print the top five significant sources with association and source class
print(top_five_TS_3fhl[["Source_Name", "ASSOC1", "ASSOC2", "CLASS", "Signif_Avg"]])


######################################################################
# If you are interested in the data of an individual source you can access
# the information from catalog using the name of the source or any alias
# source name that is defined in the catalog:
#

mkn_421_3fhl = fermi_3fhl["3FHL J1104.4+3812"]

# or use any alias source name that is defined in the catalog
mkn_421_3fhl = fermi_3fhl["Mkn 421"]
print(mkn_421_3fhl.data["Signif_Avg"])


######################################################################
# Exercises
# ~~~~~~~~~
#
# -  Try to load the Fermi-LAT 2FHL catalog and check the total number of
#    sources it contains.
# -  Select all the sources from the 2FHL catalog which are contained in
#    the Galactic Center region. The methods
#    `~gammapy.maps.WcsGeom.contains()` and
#    `~gammapy.catalog.SourceCatalog.positions` might be helpful for
#    this. Add markers for all these sources and try to add labels with
#    the source names.
# -  Try to find the source class of the object at position ra=68.6803,
#    dec=9.3331
#


######################################################################
# Spectral models and flux points
# -------------------------------
#
# In the previous section we learned how access basic data from individual
# sources in the catalog. Now we will go one step further and explore the
# full spectral information of sources. We will learn how to:
#
# -  Plot spectral models
# -  Compute integral and energy fluxes
# -  Read and plot flux points
#
# As a first example we will start with the Crab Nebula:
#

crab_3fhl = fermi_3fhl["Crab Nebula"]
crab_3fhl_spec = crab_3fhl.spectral_model()
print(crab_3fhl_spec)


######################################################################
# The ``crab_3fhl_spec`` is an instance of the
# `~gammapy.modeling.models.PowerLaw2SpectralModel` model, with the
# parameter values and errors taken from the 3FHL catalog.
#
# Let’s plot the spectral model in the energy range between 10 GeV and
# 2000 GeV:
#

ax_crab_3fhl = crab_3fhl_spec.plot(energy_bounds=[10, 2000] * u.GeV, energy_power=0)


######################################################################
# We assign the return axes object to variable called ``ax_crab_3fhl``,
# because we will re-use it later to plot the flux points on top.
#
# To compute the differential flux at 100 GeV we can simply call the model
# like normal Python function and convert to the desired units:
#

print(crab_3fhl_spec(100 * u.GeV).to("cm-2 s-1 GeV-1"))


######################################################################
# Next we can compute the integral flux of the Crab between 10 GeV and
# 2000 GeV:
#

print(
    crab_3fhl_spec.integral(energy_min=10 * u.GeV, energy_max=2000 * u.GeV).to(
        "cm-2 s-1"
    )
)


######################################################################
# We can easily convince ourself, that it corresponds to the value given
# in the Fermi-LAT 3FHL catalog:
#

print(crab_3fhl.data["Flux"])


######################################################################
# In addition we can compute the energy flux between 10 GeV and 2000 GeV:
#

print(
    crab_3fhl_spec.energy_flux(energy_min=10 * u.GeV, energy_max=2000 * u.GeV).to(
        "erg cm-2 s-1"
    )
)


######################################################################
# Next we will access the flux points data of the Crab:
#

print(crab_3fhl.flux_points)


######################################################################
# If you want to learn more about the different flux point formats you can
# read the specification
# `here <https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/flux_points/index.html>`__.
#
# No we can check again the underlying astropy data structure by accessing
# the ``.table`` attribute:
#

print(crab_3fhl.flux_points.to_table(sed_type="dnde", formatted=True))


######################################################################
# Finally let’s combine spectral model and flux points in a single plot
# and scale with ``energy_power=2`` to obtain the spectral energy
# distribution:
#

ax = crab_3fhl_spec.plot(energy_bounds=[10, 2000] * u.GeV, energy_power=2)
crab_3fhl.flux_points.plot(ax=ax, sed_type="dnde", energy_power=2)


######################################################################
# Exercises
# ~~~~~~~~~
#
# -  Plot the spectral model and flux points for PKS 2155-304 for the 3FGL
#    and 2FHL catalogs. Try to plot the error of the model (aka
#    “Butterfly”) as well.


######################################################################
# What next?
# ----------
#
# This was a quick introduction to some of the high level classes in
# Astropy and Gammapy.
#
# -  To learn more about those classes, go to the API docs (links are in
#    the introduction at the top).
# -  To learn more about other parts of Gammapy (e.g. Fermi-LAT and TeV
#    data analysis), check out the other tutorial notebooks.
# -  To see what’s available in Gammapy, browse the Gammapy docs or use
#    the full-text search.
# -  If you have any questions, ask on the mailing list.
#
