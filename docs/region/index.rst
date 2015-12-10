.. include:: ../references.txt

.. _region:

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

.. _region_reflected:

Reflected Regions
-----------------

Details on the reflected regions method can be found in [Berge2007]_

The following example illustrates how to create reflected regions
for a given ImageHDU

.. plot:: region/make_reflected_regions.py
   :include-source:

Reference/API
=============

.. automodapi:: gammapy.region
    :no-inheritance-diagram:
