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

    from gammapy.spectrum import SpectrumDatasetOnOff
    from gammapy.modeling import Fit
    from gammapy.modeling.models import PowerLawSpectralModel

    filename = '$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits'
    dataset = SpectrumDatasetOnOff.from_ogip_files(filename)

    model = PowerLawSpectralModel(
        index=2,
        amplitude='1e-12 cm-2 s-1 TeV-1',
        reference='1 TeV',
    )

    dataset.model = model

    fit = Fit([dataset])
    result = fit.run()
    model.parameters.covariance = result.parameters.covariance
    print(model)

It will print the following output to the console:

.. code-block:: text

    PowerLawSpectralModel

    Parameters:

           name     value     error        unit      min max frozen
        --------- --------- --------- -------------- --- --- ------
            index 2.817e+00 1.496e-01                nan nan  False
        amplitude 5.142e-11 6.423e-12 cm-2 s-1 TeV-1 nan nan  False
        reference 1.000e+00 0.000e+00            TeV nan nan   True

    Covariance:

           name     index   amplitude reference
        --------- --------- --------- ---------
            index 2.239e-02 6.160e-13 0.000e+00
        amplitude 6.160e-13 4.126e-23 0.000e+00
        reference 0.000e+00 0.000e+00 0.000e+00

Using `gammapy.spectrum`
========================

For more advanced use cases please go to the tutorial notebooks:

* `spectrum_simulation.html <../notebooks/spectrum_simulation.html>`__ - simulate and fit 1D spectra using
  pre-defined or a user-defined model.
* `spectrum_analysis.html <../notebooks/spectrum_analysis.html>`__ - spectral analysis starting from event
  lists and field-of-view IRFs.

The following pages describe ``gammapy.spectrum`` in more detail:

.. toctree::
    :maxdepth: 1

    fitting
    reflected

Reference/API
=============

.. automodapi:: gammapy.spectrum
    :no-inheritance-diagram:
    :include-all-objects:
