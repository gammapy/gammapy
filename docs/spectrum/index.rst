.. include:: ../references.txt

.. _spectrum:

*******************************
spectrum - 1D spectrum analysis
*******************************

.. currentmodule:: gammapy.spectrum

Introduction
============

`gammapy.spectrum` holds functions and classes related to 1D region based
spectral analysis. This includes also simulation tools.

The basic of 1D spectral analysis are explained in `this
<https://github.com/gammapy/PyGamma15/tree/gh-pages/talks/analysis-classical>`__
talk. A good reference for the forward-folding on-off likelihood fitting methods
is Section 7.5 "Spectra and Light Curves" in [Naurois2012]_, in publications
usually the reference [Piron2001]_ is used.  A standard reference for the
unfolding method is [Albert2007]_.


Using `gammapy.spectrum`
========================

For more advanced use cases please go to the tutorial notebooks:

* `spectrum_simulation.html <../notebooks/spectrum_simulation.html>`__ - simulate and fit 1D spectra using
  pre-defined or a user-defined model.
* `spectrum_analysis.html <../notebooks/spectrum_analysis.html>`__ - spectral analysis starting from event
  lists and field-of-view IRFs.


Reference/API
=============

.. automodapi:: gammapy.spectrum
    :no-inheritance-diagram:
    :include-all-objects:
