**********************************************************
Background estimation and modeling  (`gammapy.background`)
**********************************************************

.. currentmodule:: gammapy.background

Introduction
============

`gammapy.background` contains methods to estimate and model background.

The central data structure is the `~gammapy.background.maps.Maps` container.
TODO: describe

Most of the methods implemented are described in [Berge2007]_.
Section 7.3 "Background subtraction"
and Section 7.4 "Acceptance determination and predicted background"
in [Naurois2012]_ describe mostly the same methods as [Berge2007]_,
except for the "2D acceptance model" described in Section 7.4.3.

Getting Started
===============

TODO

Reference/API
=============

.. automodapi:: gammapy.background
    :no-inheritance-diagram:

.. automodapi:: gammapy.background.models
    :no-inheritance-diagram:
