.. include:: ../references.txt

.. _spectrum:

**************************
Regions (`gammapy.region`)
**************************

.. currentmodule:: gammapy.region

Introduction
============

`gammapy.region` contains classes and functions for region handling.


Getting Started
===============

Creating regions
----------------

.. code-block:: python
    
    >>> from gammapy.region import SkyCircleRegion
    >>> from astropy.coordinates import Angle, SkyCoord
    >>> pos = SkyCoord(80.2, 23.5, unit='deg', frame='icrs')
    >>> radius = Angle(0.4, 'deg')
    >>> region = SkyCircleRegion(pos=pos, radius=radius)

Containment
-----------

Masks
-----

Rotate
------

Read / write
------------

Reflected Regions
-----------------

Reference/API
=============

.. automodapi:: gammapy.region
    :no-inheritance-diagram:
