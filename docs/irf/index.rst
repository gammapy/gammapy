.. _irf:

***********************************
irf - Instrument response functions
***********************************

.. currentmodule:: gammapy.irf

Introduction
============

`gammapy.irf` handles instrument response functions (IRFs):

* Effective area (AEFF)
* Energy dispersion (EDISP)
* Point spread function (PSF)
* Template background (BKG)

Most of the formats defined at :ref:`gadf:iact-irf` are supported.  Otherwise,
at the moment, there is very little support for Fermi-LAT or other instruments.

Most users will not use `gammapy.irf` directly, but will instead use IRFs as
part of their spectrum, image or cube analysis to compute exposure and effective
EDISP and PSF for a given dataset.

Most (at some point maybe all) classes in `gammapy.irf` have an
`gammapy.utils.nddata.NDDataArray` as data attribute to support interpolation.

Getting Started
===============

See `cta.html <../notebooks/cta.html>`__ for an example how to access IACT IRFs.

Effective area
==============

See `~gammapy.irf.EffectiveAreaTable` and `~gammapy.irf.EffectiveAreaTable2D`.

Background
==========

See `~gammapy.irf.Background2D` and `~gammapy.irf.Background2D`.

PSF
===

The `~gammapy.irf.TablePSF` and `~gammapy.irf.EnergyDependentTablePSF` classes
represent radially-symmetric PSFs where the PSF is given at a number of offsets.

The `~gammapy.cube.PSFKernel` represents a PSF kernel.

.. plot:: irf/plot_fermi_psf.py

Energy Dispersion
=================

The `~gammapy.irf.EnergyDispersion` class represents an energy migration matrix
(finite probabilities per pixel) with ``y=log(energy_reco)``.

The `~gammapy.irf.EnergyDispersion2D` class represents a probability density
with ``y=energy_reco/energy_true`` that can also have a FOV offset dependence.

.. plot:: irf/plot_edisp.py


Using `gammapy.irf`
===================

If you'd like to learn more about using `gammapy.irf`, read the following
sub-pages:

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
