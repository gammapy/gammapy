.. include:: ../references.txt

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

.. _gammapyextra:

gammapy-extra
=============

To keep the Gammapy code repository at https://github.com/gammapy/gammapy small and clean,
we are putting sample data files and IPython notebooks in an extra repository
at https://github.com/gammapy/gammapy-extra/ .

To get the repository, ``git clone`` it to a location of your choosing using a git protocol of your choosing
(try HTTPS or see the `Github clone URL help article`_ if you're not sure which you want).

.. code-block:: bash

    git clone https://github.com/gammapy/gammapy-extra.git
    git clone git@github.com:gammapy/gammapy-extra.git

If you don't have git, you can also fetch the latest version as a zip file:

.. code-block:: bash

    wget https://github.com/gammapy/gammapy-extra/archive/master.zip
    unzip master.zip # will result in a `gammapy-extra-master` folder

The Gammapy tests, docs generator, examples and tutorials will access files from the ``gammapy-extra``
repo using the `gammapy.datasets.gammapy_extra` object.

For this to work, you have to set the ``GAMMAPY_EXTRA`` shell environment variable to point to that folder.
We suggest you put this in you ``.bashrc`` or ``.profile``

.. code-block:: bash

    export GAMMAPY_EXTRA=/path/on/your/machine/to/gammapy-extra

After you've done this, open up a new terminal (or ``source .profile``) and check if ``gammapy-extra`` is found:

.. code-block:: bash

    # TODO: make this print some info about gammapy-extra (including a version!!!)
    gammapy info

Example usage:

.. code-block:: python

    >>> from gammapy.datasets import gammapy_extra
    >>> gammapy_extra.filename('logo/gammapy_banner.png')
    '/Users/deil/code/gammapy-extra/logo/gammapy_banner.png'


.. _gammapy-cat:

Gamma-cat
===========

Gamma-cat is an open catalog for TeV gamma-ray sources. It is maintained as an
open git repository and hosted on github. To get the data you can use the
`git clone` command:

.. code-block:: bash

  git clone https://github.com/gammapy/gamma-cat.git
  git clone git@github.com:gammapy/gamma-cat.git

If you don't have git, you can also fetch the latest version as a zip file as well:

.. code-block:: bash

    wget https://github.com/gammapy/gamma-cat/archive/master.zip
    unzip master.zip # will result in a `gamma-cat-master` folder


The `~gammapy.catalog.SourceCatalogGammaCat` and `~gammapy.catalog.SourceCatalogObjectGammaCat`
classes need to know where the gamma-cat repository is located on your machine.
For this reason the ``GAMMA_CAT`` shell environment variable has to be set using:

.. code-block:: bash

    export GAMMAPY_CAT=/path/on/your/machine/to/gamma-cat


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

   >>> from gammapy.datasets import fetch_fermi_diffuse_background_model
   >>> catalog = fetch_fermi_diffuse_background_model()

TODO: explain how the Astropy cache works and make it configurable for Gammapy.

Example how to make a dataset (from scratch, no file is loaded):

.. code-block:: python

   >>> from gammapy.datasets import make_test_psf
   >>> psf = make_test_psf(energy_bins=20)

Using `gammapy.datasets`
========================

.. toctree::
   :maxdepth: 1

   make_datasets

Reference/API
=============

.. automodapi:: gammapy.datasets
    :no-inheritance-diagram:
    :include-all-objects:
