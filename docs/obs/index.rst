.. _obs:

****************************************
Observation bookkeeping  (`gammapy.obs`)
****************************************

.. currentmodule:: gammapy.obs

Introduction
============

`gammapy.obs` contains methods to do the bookkeeping for processing multiple observations.

In TeV astronomy an observation (a.k.a. a run) means pointing the telescopes at some
position on the sky (fixed in celestial coordinates, not in horizon coordinates)
for a given amount of time (e.g. half an hour) and switching the central trigger on.

The total dataset for a given target will usually consist of a few to a few 100 runs
and some book-keeping is required when running the analysis.

Getting Started
===============

.. code-block:: python

   >>> from gammapy.obs import observatory_locations
   >>> observatory_locations.HESS
   <EarthLocation (7237.152530011689, 2143.7727767623487, -3229.3927009565496) km>
   >>> print(observatory_locations.HESS.geodetic)
   (<Longitude 16.500222222222224 deg>, <Latitude -23.271777777772456 deg>, <Quantity 1835.0 km>)

Using `gammapy.obs`
=====================

.. toctree::
   :maxdepth: 1

   findruns

Reference/API
=============

.. automodapi:: gammapy.obs
    :no-inheritance-diagram:
