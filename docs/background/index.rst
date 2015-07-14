.. _background:

**********************************************************
Background estimation and modeling  (`gammapy.background`)
**********************************************************

.. currentmodule:: gammapy.background

Introduction
============

`gammapy.background` contains methods to estimate and model background.

The central data structure is the `~gammapy.background.Maps` container.
TODO: describe

Most of the methods implemented are described in [Berge2007]_.
Section 7.3 "Background subtraction"
and Section 7.4 "Acceptance determination and predicted background"
in [Naurois2012]_ describe mostly the same methods as [Berge2007]_,
except for the "2D acceptance model" described in Section 7.4.3.

Getting Started
===============

TODO

Hello World.

.. _bg_models:

Background Models
=================

The naming of the models in this section follows the convention from
:ref:`dataformats_overview`.

TODO: because of the link capabilities, this looks horrible:
      what is the best way to have a link to this info?

.. [BACKGROUND_3D] is a bacground rate 3D cube (X, Y, energy) in
units of per energy, per time, per solid angle. `X` and `Y` are
given in detector coordinates `(DETX, DETY)`, a.k.a.
`nominal system`. This is a tangential system to the instrument
during observations.

The `~CubeBackgroundModel` is used as container class for this model.
It has methods to read, write and operate the 3D cubes.

For the moment, only I/O and visualization methods are implemented.
A test file is located in the `~gammapy-extra` repository
(`bg_cube_model_test.fits <https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/background/bg_cube_model_test.fits>`_).
The file comes originally from the `~GammaLib` repository but has
been slightly modified.

TODO: the scripts produce some white canvases that I need to remove!

An example script of how to read/write the files and perform some
simple plots is given in the `examples` directory:

.. plot:: ../examples/plot_background_model.py
   :include-source:

More complex plots can be easily produced with a few lines of code:
TODO: mosaic/stack plots examples!!!

.. plot:: background/plot_bgcube_images_mosaic.py
   :include-source:

.. plot:: background/plot_bgcube_spectra_stack.py
   :include-source:

Reference/API
=============

.. automodapi:: gammapy.background
    :no-inheritance-diagram:
