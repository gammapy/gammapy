.. include:: ../references.txt

.. note::

    A new set of map and cube classes is being developed in `gammapy.maps`
    and long-term will replace the existing `gammapy.image.SkyImage` and
    `gammapy.cube.SkyCube` classes. Please consider trying out `gammapy.maps`
    and changing your scripts to use those new classes. See :ref:`maps`.

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

Many of the :ref:`tutorials` show examples using ``gammapy.image``:

* :gp-extra-notebook:`first_steps`
* :gp-extra-notebook:`image_pipe`
* :gp-extra-notebook:`image_analysis`

Documentation pages with more detailed information:

.. toctree::
   :maxdepth: 1

   sky_image
   plotting
   models


Reference/API
=============

.. automodapi:: gammapy.image
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.image.models
    :no-inheritance-diagram:
    :include-all-objects:

