.. include:: ../references.txt

.. _image:

*****************************************************
Image processing and analysis tools (`gammapy.image`)
*****************************************************

.. currentmodule:: gammapy.image

Introduction
============

`gammapy.image` contains data classes and methods for image based analysis
of gamma-ray data. Currently it includes multi purpose image processing
methods as well. The goal longterm goal is to contribute most of these methods
to `scipy.ndimage`, `scikit-image`_, `astropy`_ or `photutils`_, and to only keep
gamma-ray analysis specific functionality here.

Getting Started
===============

Most of the functions in this module have objects of type `numpy.array`
or an `astropy.io.fits.ImageHDU` or `astropy.io.fits.PrimaryHDU`
as input and output:

.. code-block:: python

   >>> from gammapy.datasets import poisson_stats_image
   >>> from gammapy.image import lookup
   >>> image = poisson_stats_image() # image is a 2D numpy array
   >>> lookup(image, 42, 44, world=False)
   3.0

Using `gammapy.image`
=====================

If you'd like to learn more about using `gammapy.image`, read the following sub-pages:

.. toctree::
   :maxdepth: 1

   skymaps
   plotting
   bounding_box


Reference/API
=============

.. automodapi:: gammapy.image
    :no-inheritance-diagram:
