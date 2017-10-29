.. include:: ../references.txt

.. _spectrum:

*****************************************************
Spectrum estimation and modeling (`gammapy.spectrum`)
*****************************************************

.. currentmodule:: gammapy.spectrum

Introduction
============

`gammapy.spectrum` holds functions and classes related to 1D region based
spectral analysis. This includes also simulation tools. 

The basic of 1D spectral analysis are explained in `this
<https://github.com/gammapy/PyGamma15/tree/gh-pages/talks/analysis-classical>`__
talk. A good reference for the forward-folding on-off likelihood fitting
methods is Section 7.5 "Spectra and Light Curves" in [Naurois2012]_, in
publications usually the reference [Piron2001]_ is used.  A standard reference
for the unfolding method is [Albert2007]_.

Getting Started
===============

The following code snippet demonstrates how to load an observation stored in
OGIP format and fit a spectral model.

.. code-block:: python

    import astropy.units as u
    from gammapy.datasets import gammapy_extra
    from gammapy.spectrum import SpectrumObservation, SpectrumFit, models

    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23592.fits'
    obs = SpectrumObservation.read(filename)

    model = models.PowerLaw(
        index=2 * u.Unit(''),
        amplitude=1e-12*u.Unit('cm-2 s-1 TeV-1'),
        reference=1*u.TeV,
    )
    fit = SpectrumFit(obs_list=obs, model=model)
    fit.fit()
    fit.est_errors()
    print(fit.result[0])

It will print the following output to the console:

.. code-block:: text

    Fit result info
    ---------------
    Model: PowerLaw
    ParameterList
    Parameter(name='index', value=2.1473880540790522, unit=Unit(dimensionless), min=0, max=None, frozen=False)
    Parameter(name='amplitude', value=2.7914083679020973e-11, unit=Unit("1 / (cm2 s TeV)"), min=0, max=None, frozen=False)
    Parameter(name='reference', value=1.0, unit=Unit("TeV"), min=None, max=None, frozen=True)

    Covariance: [[  6.89132245e-03   1.12566759e-13   0.00000000e+00]
     [  1.12566759e-13   7.26865610e-24   0.00000000e+00]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00]]

    Statistic: 46.051 (wstat)
    Fit Range: [  5.99484250e+08   1.00000000e+11] keV



Using `gammapy.spectrum`
========================

For more advanced use cases please go to the tutorial notebooks:

* :gp-extra-notebook:`spectrum_simulation` - simulate and fit 1D spectra using pre-defined or a user-defined model.
* :gp-extra-notebook:`spectrum_analysis` - spectral analysis starting from event lists and field-of-view IRFs.

The following pages describe ``gammapy.spectrum`` in more detail:

.. toctree::
   :maxdepth: 1

   fitting
   energy_group

Reference/API
=============

.. automodapi:: gammapy.spectrum
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.spectrum.models
    :no-inheritance-diagram:
    :include-all-objects:
