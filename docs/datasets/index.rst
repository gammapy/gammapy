.. _datasets:

************************************
Access datasets (`gammapy.datasets`)
************************************

.. currentmodule:: gammapy.datasets

Introduction
============

`gammapy.datasets` contains function to easily access datasets that are
relevant for gamma-ray astronomy.

The functions have a naming pattern (following the `sklearn.datasets` lead):

* ``load_*`` functions load datasets that are distributed with Gammapy (bundled in the repo) 
* ``fetch_*`` functions fetch datasets from the web (either from ``gammapy-extra`` or other sites)
* ``make_*`` functions create datasets programatically (sometimes involving a random number generator)

.. note:: The `gammapy.datasets` sub-package shouldn't be confused with the `gammapy.data`
          sub-package, which contains classes representing gamma-ray data.
          And there is a separate section describing the :ref:`dataformats`
          that are commonly used in gamma-ray astronomy.

Getting Started
===============

Example how to load a dataset that is distributed with the code
in the ``gammapy`` repo (i.e. will be available even if you're offline)

.. code-block:: python

   >>> from gammapy.datasets import load_crab_flux_points
   >>> flux_points = load_crab_flux_points()

Example how to fetch a dataset from the web (i.e. will download to
the Astropy cache and need internet access on first call):

.. code-block:: python

   >>> from gammapy.datasets import fetch_fermi_catalog
   >>> catalog = fetch_fermi_catalog('2FGL', 'LAT_Point_Source_Catalog')  

TODO: explain how the Astropy cache works and make it configurable for Gammapy.

Example how to make a dataset (from scratch, no file is loaded): 

.. code-block:: python

   >>> from gammapy.datasets import make_test_psf
   >>> psf = make_test_psf(energy_bins=20)

Reference/API
=============

.. automodapi:: gammapy.datasets
    :no-inheritance-diagram:
