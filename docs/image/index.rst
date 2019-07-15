.. include:: ../references.txt

.. _image:

**************************
image - Map image analysis
**************************

.. currentmodule:: gammapy.image

Introduction
============

`gammapy.image` contains functions and classes for image based analysis.
`gammapy.image.models` contains image models that can be evaluated and fitted.

Getting Started
===============

The functions and classes in `gammapy.image` take `gammapy.maps` objects as
input and output. Currently they only work on WCS-based 2D images. For some, we
will improve them to also work on HPX-based images and on maps with extra axes,
such as e.g. an energy axis.

.. plot::
    :include-source:

    from gammapy.maps import Map
    filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz"
    image = Map.read(filename)
    image.smooth("0.1 deg").plot()

TODO: Show some gammapy.image functionality, e.g. evaluating a model image or
making a profile.

Using `gammapy.image`
=====================

:ref:`tutorials` that contain use of maps:

* :gp-notebook:`first_steps`
* :gp-notebook:`intro_maps`
* :gp-notebook:`analysis_3d`

Documentation pages with more detailed information:

.. toctree::
    :maxdepth: 1

    plotting

Reference/API
=============

.. automodapi:: gammapy.image
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.image.models
    :no-inheritance-diagram:
    :include-all-objects:
