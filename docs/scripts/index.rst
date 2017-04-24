.. include:: ../references.txt

.. _scripts:

***************************************
Command line tools  (`gammapy.scripts`)
***************************************

.. currentmodule:: gammapy.scripts

Introduction
============

Gammapy contains a bunch of command line tools,
and we've started to add `flask`_ web apps for cases where a GUI makes sense.

Note that these are all still very much work in progress, we haven't
settled on a very good scheme to organise the functionality, do
argument and config handling, logging, integration with the high-level
Sphinx docs, ...

So please use these tools and help us make them better:
complain if something doesn't work, file feature requests,
... on Github or the mailing list!

.. _scripts_overview:

Overview
========

Here's a list of available tools, grouped by category.

Utilities
---------

- ``gammapy-info`` calls `gammapy.scripts.info`
- ``gammapy-test`` calls `gammapy.scripts.check` (TODO: merge with ``gammapy-info``?)

Data
----

- ``gammapy-data-browse`` (web GUI tool) calls `gammapy.scripts.data_browser`
- ``gammapy-data-manage`` calls `gammapy.scripts.data_manage`
- ``gammapy-data-select`` calls `gammapy.scripts.data_select`
- ``gammapy-data-show`` calls `gammapy.scripts.data_show`

Spectrum
--------

- ``gammapy-spectrum`` calls `gammapy.spectrum.spectrum_analysis`
- ``gammapy-spectrum-pipe`` runs `gammapy.scripts.SpectrumPipe`

Image
-----

- ``gammapy-image-bin`` calls `gammapy.scripts.image_bin`
- ``gammapy-image-fit`` calls `gammapy.scripts.image_fit`
- ``gammapy-image-model`` calls `gammapy.scripts.image_model`
- ``gammapy-image-model-sherpa`` calls `gammapy.scripts.image_model_sherpa`
- ``gammapy-image-pipe`` uses `gammapy.scripts.ImageAnalysis`
- ``gammapy-image-residual`` calls `gammapy.scripts.image_residual`
- ``gammapy-image-significance`` calls `gammapy.scripts.image_significance`
- ``gammapy-image-ts`` calls `gammapy.scripts.image_ts`

Cube
----

- ``gammapy-cube-background`` calls `gammapy.scripts.cube_background`
- ``gammapy-cube-bin`` calls `gammapy.scripts.cube_bin`

Detect
------

- ``gammapy-detect`` calls `gammapy.scripts.detect`
- ``gammapy-detect-iterative`` calls `gammapy.scripts.detect_iterative`

Catalog
-------

- ``gammapy-catalog-browse`` (web GUI tool) calls `gammapy.scripts.catalog_browser` (TODO: list in the API docs)
- ``gammapy-catalog-query`` calls `gammapy.scripts.catalog_query`


Technical background
====================

The Gammapy command line tools and web apps use the
`setuptools entry points <https://pythonhosted.org/setuptools/setuptools.html#automatic-script-creation>`__
method to automatically create command line tools when Gammapy is installed.

This means that to be able to use the tools you have to install Gammapy:

.. code-block:: bash

    $ pip install --user .


This will install the ``gammapy-*`` wrappers in a ``bin`` folder that you need to add to your ``$PATH``,
which will then call into the appropriate function in the Gammapy package.

For Gammapy development we recommend you run this command so that you can edit
Gammapy and the tools and don't have to re-install after every change.

.. code-block:: bash

    $ pip install --user --editable .


Most of the command line tools are implemented in the `gammapy.scripts` sub-package as thin wrappers
around functionality that's implemented in the Gammapy package as re-usable functions and classes.
In most cases all the command line tool ``main`` function does is argument passing and setting up logging.

If you'd like to write your own command line tool that uses Gammapy functionality,
you don't need to know about this or implement it in Gammapy source tree or install it into site-packages.
Just write a ``myscript.py`` file and import the Gammapy functions, classes you need.

Reference/API
=============

.. automodapi:: gammapy.scripts
    :no-inheritance-diagram:
