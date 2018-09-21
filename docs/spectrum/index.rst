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

Getting Started
===============

The following code snippet demonstrates how to load an observation stored in
OGIP format and fit a spectral model.

.. code-block:: python

    from gammapy.spectrum import SpectrumObservation, SpectrumFit
    from gammapy.spectrum.models import PowerLaw

    filename = '$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits'
    obs = SpectrumObservation.read(filename)

    model = PowerLaw(
        index=2,
        amplitude='1e-12 cm-2 s-1 TeV-1',
        reference='1 TeV',
    )
    fit = SpectrumFit(obs_list=[obs], model=model)
    fit.run()
    print(fit.result[0])

It will print the following output to the console:

.. code-block:: text

    Fit result info
    ---------------
    Model: PowerLaw

    Parameters:

           name     value     error         unit         min    max
        --------- --------- --------- --------------- --------- ---
            index 2.791e+00 1.456e-01                       nan nan
        amplitude 5.030e-11 6.251e-12 1 / (cm2 s TeV)       nan nan
        reference 1.000e+00 0.000e+00             TeV 0.000e+00 nan

    Covariance:

           name           index               amplitude        reference
        --------- --------------------- ---------------------- ---------
            index  0.021213640646334082  5.788340722422449e-13       0.0
        amplitude 5.788340722422449e-13 3.9079614123597625e-23       0.0
        reference                   0.0                    0.0       0.0

    Statistic: 41.756 (wstat)
    Fit Range: [8.79922544e+08 1.00000000e+11] keV

Using `gammapy.spectrum`
========================

For more advanced use cases please go to the tutorial notebooks:

* :gp-extra-notebook:`spectrum_simulation` - simulate and fit 1D spectra using
  pre-defined or a user-defined model.
* :gp-extra-notebook:`spectrum_analysis` - spectral analysis starting from event
  lists and field-of-view IRFs.

The following pages describe ``gammapy.spectrum`` in more detail:

.. toctree::
    :maxdepth: 1

    fitting

Reference/API
=============

.. automodapi:: gammapy.spectrum
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.spectrum.models
    :no-inheritance-diagram:
    :include-all-objects:
