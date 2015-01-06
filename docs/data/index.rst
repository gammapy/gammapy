.. include:: ../references.txt

.. _data:

*****************************
Data classes (`gammapy.data`)
*****************************

.. currentmodule:: gammapy.data

Introduction
============

`gammapy.data` contains classes to represent gamma-ray data:

* An `~gammapy.data.EventList` contains unbinned data,
  i.e. the longitude, latitude, energy and time for each event.
* A `~gammapy.data.SpectralCube` contains binned dat as a 3-dimensional array
  with longitude, latitude and log energy axes.
* The event time information has been removed ... it is represented by
  a `~gammapy.data.GoodTimeInterval` (a.k.a. ``GTI``) class that is needed
  for exposure and flux computations.

Getting Started
===============

TODO

Reference/API
=============

.. automodapi:: gammapy.data
    :no-inheritance-diagram:
