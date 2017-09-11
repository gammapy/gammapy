.. _irf:

****************************************************************
Instrument response function (IRF) functionality (`gammapy.irf`)
****************************************************************

.. currentmodule:: gammapy.irf

Introduction
============

`gammapy.irf` handles instrument response functions (IRFs):

* Effective area (AEFF)
* Energy dispersion (EDISP)
* Point spread function (PSF)

Most of the formats defined at :ref:`gadf:iact-irf` are supported.  Otherwise,
at the moment, there is very little support for Fermi-LAT or other instruments.

Most users will not use `gammapy.irf` directly, but will instead use IRFs as
part of their spectrum, image or cube analysis to compute exposure and
effective EDISP and PSF for a given dataset.

Most (at some point maybe all) classes in `gammapy.irf` have an
`gammapy.utils.nddata.NDDataArray` as data attribute to support interpolation.

Getting Started
===============

see :gp-extra-notebook:`data_iact` for an example how to access IACT IRFs.


Using `gammapy.irf`
===================

If you'd like to learn more about using `gammapy.irf`, read the following sub-pages:

.. toctree::
   :maxdepth: 1

   theory
   aeff
   edisp
   psf

Reference/API
=============

.. automodapi:: gammapy.irf
    :no-inheritance-diagram:
    :include-all-objects:
