.. include:: ../references.txt

.. _data:

***************************************
data - DL3 data access and observations
***************************************

.. currentmodule:: gammapy.data

Introduction
============
IACT data is typically structured in "observations", which define a given
time interval during with the instrument response is considered stable.


`gammapy.data` currently contains the `~gammapy.data.EventList` class,
as well as classes for IACT data and observation handling.


The main classes in Gammapy to access the DL3 data library are the
`~gammapy.data.DataStore` and `~gammapy.data.Observation`.
They are used to store and retrieve dynamically the datasets
relevant to any observation (event list in the form of an `~gammapy.data.EventList`,
IRFs see :ref:`irf` and other relevant informations).

Once some observation selection has been selected, the user can build a list of observations:
a `~gammapy.data.Observations` object, which will be used for the data reduction process.


Getting started
===============

You can use the `~gammapy.data.EventList` class to load IACT gamma-ray event lists:

.. testcode::

    from gammapy.data import EventList
    filename = '$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz'
    events = EventList.read(filename)

To load Fermi-LAT event lists, use the `~gammapy.data.EventListLAT` class:

.. testcode::

    from gammapy.data import EventList
    filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"
    events = EventList.read(filename)

The other main class in `gammapy.data` is the `~gammapy.data.DataStore`, which makes it easy
to load IACT data. E.g. an alternative way to load the events for observation ID 23523 is this:

.. testcode::

    from gammapy.data import DataStore
    data_store = DataStore.from_dir('$GAMMAPY_DATA/hess-dl3-dr1')
    events = data_store.obs(23523).events

Using `gammapy.data`
====================

Gammapy tutorial notebooks that show examples using ``gammapy.data``:

.. nbgallery::

   ../tutorials/data/cta.ipynb
   ../tutorials/data/hess.ipynb
   ../tutorials/data/fermi_lat.ipynb

Reference/API
=============

.. automodapi:: gammapy.data
    :no-inheritance-diagram:
    :include-all-objects:
