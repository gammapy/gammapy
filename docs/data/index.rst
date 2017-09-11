.. include:: ../references.txt

.. _data:

**********************************************
Data and observation handling (`gammapy.data`)
**********************************************

.. currentmodule:: gammapy.data

Introduction
============

`gammapy.data` currently contains the `~gammapy.data.EventList` class,
as well as classes for IACT data and observation handling.

Getting Started
===============

You can use the `~gammapy.data.EventList` class to load gamma-ray event lists:

.. code-block:: python

    >>> from gammapy.data import EventListDataset
    >>> filename = '$GAMMAPY_EXTRA/datasets/vela_region/events_vela.fits'
    >>> events = EventListDataset.read(filename)

TODO: ``events.info()`` gives s ``KeyError: 'ONTIME'``.
Should we introduce a sub-class ``EventListIACT``?

.. code-block:: python

    >>> from gammapy.data import EventListDataset
    >>> filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_events_023523.fits.gz'
    >>> events = EventListDataset.read(filename)
    >>> events.info()

Using `gammapy.data`
====================

If you'd like to learn more about using `gammapy.data`, read the following sub-pages:

.. toctree::
   :maxdepth: 1

   obs_select
   obs_group
   dm
   server
   index-old-obs


Reference/API
=============

.. automodapi:: gammapy.data
    :no-inheritance-diagram:
    :include-all-objects:
