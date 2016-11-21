.. include:: ../references.txt

.. _spectral_fitting:

****************
Spectral Fitting
****************

.. currentmodule:: gammapy.spectrum

In the following you will see how to fit spectral data in OGIP format. The
format is described at :ref:`gadf:ogip`. An example dataset is available in the
`gammapy-extra repo <https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-crab4_pha>`_. For a description of the available fit statstics see :ref:`fit-statistics`.

Getting Started
===============

The following example shows how to fit a power law simultaneously to two
simulated crab runs using the `~gammapy.spectrum.SpectrumFit` class.

.. code-block:: python

    import astropy.units as u
    from gammapy.spectrum import SpectrumObservation, SpectrumObservationList, SpectrumFit
    from gammapy.spectrum.models import PowerLaw
    import matplotlib.pyplot as plt

    pha1 = "$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23592.fits"
    pha2 = "$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits"
    obs1 = SpectrumObservation.read(pha1)
    obs2 = SpectrumObservation.read(pha2)
    obs_list = SpectrumObservationList([obs1, obs2])

    model = PowerLaw(index = 2 * u.Unit(''),
                     amplitude = 10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
                     reference = 1 * u.TeV)

    fit = SpectrumFit(obs_list=obs_list, model=model)
    fit.statistic = 'WStat'
    fit.fit()
    
You can check the fit results by looking at
`~gammapy.spectrum.SpectrumFitResult` that is attached to the
`~gammapy.spectrum.SpectrumFit` for each observation.


.. code-block:: python

    >>> print(fit.global_result)

    Fit result info 
    --------------- 
    Best Fit Model: PowerLaw
    index : 2.12+/-0.05
    reference : 1e+09
    amplitude : (2.08+/-0.00)e-20 
    --> Units: keV, cm, s

    Statistic: 103.596 (wstat)
    Covariance:
    [u'index', u'amplitude']
    [[  2.95033865e-03   3.08066478e-43]
     [  3.08066478e-43   1.70801015e-82]]
    Fit Range: [  0.49582929  82.70931131] TeV


Interactive Sherpa Fit
======================

If you want to do something specific that is not handled by the
`~gammapy.spectrum.SpectrumFit` class you can always fit the PHA data directly
using `Sherpa`_. The following example illustrates how to do this with the
example dataset used above. It makes use of the Sherpa `datastack module
<http://cxc.harvard.edu/sherpa/ahelp/datastack.html>`_.

.. code-block:: python

    from gammapy.datasets import gammapy_extra
    from sherpa.astro import datastack
    from sherpa.models import PowLaw1D

    pha1 = gammapy_extra.filename('datasets/hess-crab4_pha/pha_obs23592.fits')
    pha2 = gammapy_extra.filename('datasets/hess-crab4_pha/pha_obs23523.fits')
    phalist = ','.join([pha1, pha2])

    ds = datastack.DataStack()
    ds.load_pha(phalist)

    model = PowLaw1D('powlaw1d.default')
    model.ampl = 1 
    model.ref = 1e9
    model.gamma = 2

    ds.set_source(model*1e-20)

    for i in range(1, len(ds.datasets) + 1):
        datastack.ignore_bad(i)
        datastack.ignore_bad(i, 1)

    datastack.set_stat('wstat')
    ds.fit()
    datastack.covar()

This should give the following output

.. code-block:: python

    Datasets              = 1, 2
    Method                = levmar
    Statistic             = wstat
    Initial fit statistic = 218.385
    Final fit statistic   = 103.596 at function evaluation 19
    Data points           = 82
    Degrees of freedom    = 80
    Probability [Q-value] = 0.0392206
    Reduced statistic     = 1.29494
    Change in statistic   = 114.79
    powlaw1d.default.gamma   2.11641     
    powlaw1d.default.ampl   2.08095     
    Datasets              = 1, 2
    Confidence Method     = covariance
    Iterative Fit Method  = None
    Fitting Method        = levmar
    Statistic             = wstat
    covariance 1-sigma (68.2689%) bounds:
       Param            Best-Fit  Lower Bound  Upper Bound
       -----            --------  -----------  -----------
       powlaw1d.default.gamma      2.11641   -0.0543186    0.0543186
       powlaw1d.default.ampl      2.08095    -0.130691     0.130691


