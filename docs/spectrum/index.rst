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

The following code snipped demonstrates how to load an observation stored in
OGIP format and fit a spectral model.

    >>> from gammapy.datasets import gammapy_extra
    >>> from gammapy.spectrum import SpectrumObservation, SpectrumFit, models
    >>> filename = gammapy_extra.filename('datasets/hess-crab4_pha/pha_obs23592.fits')
    >>> obs = SpectrumObservation.read(filename)
    >>> import astropy.units as u
    >>> model = models.PowerLaw(index=2 * u.Unit(''),
    ... amplitude=1e-12*u.Unit('cm-2 s-1 TeV-1'),
    ... reference=1*u.TeV)
    >>> fit = SpectrumFit(obs_list=obs, model=model)
    >>> fit.fit()
    >>> fit.est_errors()
    >>> print(fit.result[0])
    Fit result info
    ---------------
    Model: PowerLaw
    ParameterList
    Parameter(name=u'index', value=2.1473880542786756+/-0.08301398949415907,
    unit='', min=0, max=None, frozen=False)
    Parameter(name=u'amplitude',
    value=2.791408367990182e-11+/-2.6960445284190296e-12, unit='', min=0,
    max=None, frozen=False)
    Parameter(name=u'reference', value=1.0, unit='', min=None, max=None,
    frozen=True)
    Covariance: None
    
    Statistic: 46.051 (wstat)
    Covariance:
    [u'index', u'amplitude']
    [[  6.89132245e-03   1.12566759e-13]
    [  1.12566759e-13   7.26865610e-24]]
    Fit Range: [  5.99484250e+08   1.00000000e+11] keV

For more advanced use cases please go to the tutorial notebooks. In
:gp-extra-notebook:`spectrum_simulation` you will learn how to simulate and fit
1D spectra. It also contains an example how to set up a user defined spectral
model. In :gp-extra-notebook:`spectrum_analysis` the spectral analysis starting
from event lists and field-of-view IRFs is demonstrated.


Content
=======
.. toctree::
   :maxdepth: 1

   fitting
   flux_points
   energy_group
   plotting_fermi_spectra

Reference/API
=============

.. automodapi:: gammapy.spectrum
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.models
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.powerlaw
    :no-inheritance-diagram:
