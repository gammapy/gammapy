.. include:: ../references.txt

.. _image:

*****************************************************
Image processing and analysis tools (`gammapy.image`)
*****************************************************

.. currentmodule:: gammapy.image

Introduction
============

`gammapy.image` contains data classes and methods for image based analysis
of gamma-ray data.


Getting Started
===============

The central data structure in `gammapy.image` is the `SkyImage`
class, which combines the raw data with WCS information, FITS I/O functionality
and many other methods, that allow easy handling, processing and plotting of
image based data. Here is a first example:

.. plot::
    :include-source:

	from gammapy.image import SkyImage
	filename = '$GAMMAPY_EXTRA/datasets/fermi_2fhl/fermi_2fhl_vela.fits.gz'
	image = SkyImage.read(filename, hdu=2)
	image.show()

This loads a prepared Fermi 2FHL FITS image of the Vela region, creates a
`SkyImage` and shows it on the the screen by calling `SkyImage.show()`.

To explore further the SkyImage class try tab completion on the ``image`` object
in an interactive python environment or see the :doc:`sky_image` page.


Using `gammapy.image`
=====================

If you'd like to learn more about using `gammapy.image`, read the following sub-pages:

.. toctree::
   :maxdepth: 1

   sky_image
   plotting
   models


Reference/API
=============

.. automodapi:: gammapy.image
    :no-inheritance-diagram:

.. automodapi:: gammapy.image.models
    :no-inheritance-diagram:

