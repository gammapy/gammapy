.. _astro:

Astrophysics
============

This module contains utility functions for some astrophysical scenarios:

* `~gammapy.astro.source` for astrophysical source models
* `~gammapy.astro.population` for astrophysical population models
* `~gammapy.astro.darkmatter` for dark matter spatial and spectral models

The `gammapy.astro` sub-package is in a prototyping phase and its scope and future
are currently being discussed. It is likely that some functionality will
be removed or split out into a separate package at some point.

Getting started
---------------

The `gammapy.astro` namespace is empty. Use these import statements:

.. testcode::

    from gammapy.astro import source
    from gammapy.astro import population
    from gammapy.astro import darkmatter

Please refer to the Getting Started section of each sub-package for a further introduction.

Sub-packages
------------

.. toctree::
    :maxdepth: 1

    source/index
    population/index
    darkmatter/index
