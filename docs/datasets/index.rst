.. include:: ../references.txt

.. _datasets:

*************************
datasets - Dataset access
*************************

.. currentmodule:: gammapy.datasets

Introduction
============

`gammapy.datasets` contains function to easily access datasets that are relevant
for gamma-ray astronomy.

The functions have a naming pattern (following scikit-learn lead):

* ``load_*`` functions load datasets that are distributed with Gammapy (bundled in the repo)
* ``fetch_*`` functions fetch datasets from the web (either from ``gammapy-extra`` repository or other sites)
* ``make_*`` functions create datasets programatically (sometimes involving a random number generator)

.. note::
    The `gammapy.datasets` sub-package shouldn't be confused with the
    `gammapy.data` sub-package, which contains classes representing gamma-ray
    data. And there is a separate section describing the :ref:`dataformats` that
    are commonly used in gamma-ray astronomy.

.. _gammapyextra:

gammapy-extra
=============

To keep the Gammapy code repository at https://github.com/gammapy/gammapy small
and clean, we are putting sample data files in an extra repository at
https://github.com/gammapy/gammapy-extra/ . These sample data files may be
fetched with ``gammapy-download datasets`` and then point your `$GAMMAPY_DATA` to the local
path you have chosen.

.. code-block:: bash

    # Download GAMMAPY_DATA
    cd code
    gammapy download datasets --out GAMMAPY_DATA
    export GAMMAPY_DATA=$PWD/GAMMAPY_DATA

.. _gamma-cat:

Gamma-cat
===========

Gamma-cat is an open catalog for TeV gamma-ray sources. It is maintained as an
open git repository and hosted on github. To get the data you can use the ``git
clone`` command:

.. code-block:: bash

    git clone https://github.com/gammapy/gamma-cat.git
    git clone git@github.com:gammapy/gamma-cat.git

If you don't have git, you can also fetch the latest version as a zip file as well:

.. code-block:: bash

    wget https://github.com/gammapy/gamma-cat/archive/master.zip unzip
    master.zip # will result in a `gamma-cat-master` folder

Gamma-cat is also shipped in the dataset collection provided by ```gammapy download datasets``

Getting Started
===============

Example how to load a dataset that is distributed with the code in the
``gammapy`` repo (i.e. will be available even if you're offline)

.. code-block:: python

    >>> from gammapy.datasets import load_crab_flux_points
    >>> flux_points = load_crab_flux_points()

Example how to fetch a dataset from the web (i.e. will download to the Astropy
cache and need internet access on first call):

.. code-block:: python

    >>> from gammapy.datasets import fetch_fermi_diffuse_background_model
    >>> catalog = fetch_fermi_diffuse_background_model()

TODO: explain how the Astropy cache works and make it configurable for Gammapy.

Example how to make a dataset (from scratch, no file is loaded):

.. code-block:: python

    >>> from gammapy.datasets import make_test_psf
    >>> psf = make_test_psf(energy_bins=20)

Using `gammapy.datasets`
========================

tbd

Reference/API
=============

.. automodapi:: gammapy.datasets
    :no-inheritance-diagram:
    :include-all-objects:
