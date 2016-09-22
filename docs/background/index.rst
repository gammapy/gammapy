.. _background:

**********************************************************
Background estimation and modeling  (`gammapy.background`)
**********************************************************

.. currentmodule:: gammapy.background

Introduction
============

`gammapy.background` contains methods to estimate and model background for specral,
image based and cube analyses.

Most of the methods implemented are described in [Berge2007]_.
Section 7.3 "Background subtraction"
and Section 7.4 "Acceptance determination and predicted background"
in [Naurois2012]_ describe mostly the same methods as [Berge2007]_,
except for the "2D acceptance model" described in Section 7.4.3.

The background models implemented in Gammapy are documented in :ref:`bg_models`.

Getting Started
===============


* TODO: example how to read and use a 2D and A 3D background model

Using `gammapy.background`
--------------------------

If you'd like to learn more about using `gammapy.background`, read the following sub-pages:

.. toctree::
   :maxdepth: 1

   models
   make_models
   energy_offset_array
   reflected

Reference/API
=============

.. automodapi:: gammapy.background
    :no-inheritance-diagram:
