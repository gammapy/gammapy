.. include:: ../references.txt

.. _spectral_fitting:

****************
Spectral Fitting
****************

.. currentmodule:: gammapy.spectrum

In the following you will see how to fit spectral data in OGIP format. The
format is described at :ref:`gadf:ogip`. An example dataset is available in the
``$GAMMAPY_DATA`` repo. For a description of the available fit statstics see
:ref:`fit-statistics`.

Getting Started
===============

The following example shows how to fit a power law simultaneously to two
simulated crab runs using the `~gammapy.modeling.Fit` class.

.. code-block:: python

    from gammapy.spectrum import SpectrumDatasetOnOff
    from gammapy.modeling import Fit
    from gammapy.modeling.models import PowerLawSpectralModel
    import matplotlib.pyplot as plt

    path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
    obs_1 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23523.fits")
    obs_2 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23592.fits")

    model = PowerLawSpectralModel(
        index=2,
        amplitude='1e-12  cm-2 s-1 TeV-1',
        reference='1 TeV',
    )

    obs_1.model = model
    obs_2.model = model

    fit = Fit([obs_1, obs_2])
    result = fit.run()

model.parameters.covariance = result.parameters.covariance
You can check the fit results by looking at the result and model object:

.. code-block:: python

    >>> print(result)

        OptimizeResult

        backend    : minuit
        method     : minuit
        success    : True
        nfev       : 115
        total stat : 65.36
        message    : Optimization terminated successfully.


    >>> print(model)

        PowerLawSpectralModel

        Parameters:

               name     value     error        unit      min max frozen
            --------- --------- --------- -------------- --- --- ------
                index 2.781e+00 1.120e-01                nan nan  False
            amplitude 5.201e-11 4.965e-12 cm-2 s-1 TeV-1 nan nan  False
            reference 1.000e+00 0.000e+00            TeV nan nan   True

        Covariance:

               name     index   amplitude reference
            --------- --------- --------- ---------
                index 1.255e-02 3.578e-13 0.000e+00
            amplitude 3.578e-13 2.465e-23 0.000e+00
            reference 0.000e+00 0.000e+00 0.000e+00


Interactive Sherpa Fit
======================

If you want to do something specific you can always fit the PHA data directly
using `Sherpa`_. The following example illustrates how to do this with the
example dataset used above. It makes use of the Sherpa `datastack module
<http://cxc.harvard.edu/sherpa/ahelp/datastack.html>`_.

.. code-block:: python

    from pathlib import Path
    import os
    from sherpa.astro import datastack
    from sherpa.models import PowLaw1D

    pha1 = str(Path(os.environ["GAMMAPY_DATA"]) / "joint-crab/spectra/hess/pha_obs23592.fits")
    pha2 = str(Path(os.environ["GAMMAPY_DATA"]) / "joint-crab/spectra/hess/pha_obs23523.fits")
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

.. code-block:: text

    Datasets              = 1, 2
    Method                = levmar
    Statistic             = wstat
    Initial fit statistic = 253.552
    Final fit statistic   = 65.361 at function evaluation 25
    Data points           = 82
    Degrees of freedom    = 80
    Probability [Q-value] = 0.88159
    Reduced statistic     = 0.817012
    Change in statistic   = 188.191
       powlaw1d.default.gamma   2.78053      +/- 0.121423
       powlaw1d.default.ampl   5.20034      +/- 0.510299
    Datasets              = 1, 2
    Confidence Method     = covariance
    Iterative Fit Method  = None
    Fitting Method        = levmar
    Statistic             = wstat
    covariance 1-sigma (68.2689%) bounds:
       Param            Best-Fit  Lower Bound  Upper Bound
       -----            --------  -----------  -----------
       powlaw1d.default.gamma      2.78053    -0.112025     0.112025
       powlaw1d.default.ampl      5.20034    -0.496564     0.496564


