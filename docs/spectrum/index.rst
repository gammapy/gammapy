.. include:: ../references.txt

.. _spectrum:

*****************************************************
Spectrum estimation and modeling (`gammapy.spectrum`)
*****************************************************

.. currentmodule:: gammapy.spectrum

Introduction
============

`gammapy.spectrum` holds functions and classes to fit spectral models and compute flux points.

Physical radiative models (synchrotron, inverse Compton and pion-decay emission)
for arbitrary cosmic ray particle spectra are available in the `naima`_ package.

Explain spectrum estimation basics.

Define vocabulary.

A good reference for the forward-folding on-off likelihood fitting methods is Section 7.5 "Spectra and Light Curves" in [Naurois2012]_,
in publications usually the reference [Piron2001]_ is used.
A standard reference for the unfolding method is [Albert2007]_.

Getting Started
===============

TODO

Reference/API
=============

.. automodapi:: gammapy.spectrum
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.fitting_utils
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.models
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.powerlaw
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.sherpa_chi2asym
    :no-inheritance-diagram:
    