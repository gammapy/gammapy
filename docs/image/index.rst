.. include:: ../references.txt

*****************************************************
Image processing and analysis tools (`gammapy.image`)
*****************************************************

.. currentmodule:: gammapy.image

Introduction
============

`gammapy.image` contains some image processing and analysis methods that are not readily available elsewhere.

The goal is to contribute most of these methods to `scipy.ndimage`, `scikit-image`_, `astropy`_ or `photutils`_,
and to only keep gamma-ray analysis specific functionality here.

Getting Started
===============

Most of the functions in this module have objects of type `numpy.array`
or an `astropy.io.fits.ImageHDU` or `astropy.io.fits.PrimaryHDU`
as input and output::

   >>> from gammapy.datasets import poisson_stats_image
   >>> from gammapy.image import lookup
   >>> image = poisson_stats_image() # image is a 2D numpy array
   >>> lookup(image, 42, 44, world=False)
   3.0


Using `gammapy.image`
=====================

.. toctree::
   :maxdepth: 1

   plotting


Reference/API
=============

.. automodapi:: gammapy.image
    :no-inheritance-diagram:
