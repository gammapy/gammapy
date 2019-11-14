.. include:: ../references.txt

.. _data:

****************************
data - Data and observations
****************************

.. currentmodule:: gammapy.data

Introduction
============

`gammapy.data` currently contains the `~gammapy.data.EventList` class,
as well as classes for IACT data and observation handling.

Getting Started
===============

You can use the `~gammapy.data.EventList` class to load IACT gamma-ray event lists:

.. code-block:: python

    >>> from gammapy.data import EventList
    >>> filename = '$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz'
    >>> events = EventList.read(filename)

To load Fermi-LAT event lists, use the `~gammapy.data.EventListLAT` class:

.. code-block:: python

    >>> from gammapy.data import EventListLAT
    >>> filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"
    >>> events = EventListLAT.read(filename)

The other main class in `gammapy.data` is the `~gammapy.data.DataStore`, which makes it easy
to load IACT data. E.g. an alternative way to load the events for observation ID 23523 is this:

.. code-block:: python

    >>> from gammapy.data import DataStore
    >>> data_store = DataStore.from_dir('$GAMMAPY_DATA/hess-dl3-dr1')
    >>> events = data_store.obs(23523).events

Using `gammapy.data`
====================

Gammapy tutorial notebooks that show examples using ``gammapy.data``:

* `cta.html <../notebooks/cta.html>`__
* `hess.html <../notebooks/hess.html>`__
* `fermi_lat.html <../notebooks/fermi_lat.html>`__

Reference/API
=============

.. automodapi:: gammapy.data
    :no-inheritance-diagram:
    :include-all-objects:
