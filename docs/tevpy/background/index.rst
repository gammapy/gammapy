********************************************************
Background estimation and modeling  (`tevpy.background`)
********************************************************

.. currentmodule:: tevpy.background

Introduction
============

`tevpy.background` contains methods to estimate and model background.

At the moment it also contains a lot of image-processing related functionality
that maybe should be split into a separate `image` package.

The main data structure is the `~tevpy.background.maps.Maps` container ... TODO

Most of the methods implemented are described in [Berge2007]_.
Section 7.3 "Background subtraction"
and Section 7.4 "Acceptance determination and predicted background"
in [Naurois2012]_ describe mostly the same methods as [Berge2007]_,
except for the "2D acceptance model" described in Section 7.4.3.

Getting Started
===============

TODO

.. automodapi:: tevpy.background
    :no-inheritance-diagram:

