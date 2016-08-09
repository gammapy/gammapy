.. include:: ../references.txt

.. _spectral_fitting:

****************
Spectral Fitting
****************

.. currentmodule:: gammapy.spectrum

In the following you will see how to fit spectral data in OGIP format. The
format is described at :ref:`gadf:ogip`. An example dataset is available in the
`gammapy-extra repo <https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-crab4_pha>`_.

Getting Started
===============

The following example shows how to fit a power law simultaneously to two
simulated crab runs using the `~gammapy.spectrum.SpectrumFit` class.

.. code-block:: python

    import astropy.units as u
    from gammapy.spectrum import (
        SpectrumObservation,
        SpectrumObservationList,
        SpectrumFit,
        models,
    )
    import matplotlib.pyplot as plt

    pha1 = "$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23592.fits"
    pha2 = "$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits"
    obs1 = SpectrumObservation.read(pha1)
    obs2 = SpectrumObservation.read(pha2)
    obs_list = SpectrumObservationList([obs1, obs2])

    model = models.PowerLaw(index = 2 * u.Unit(''),
                            amplitude = 10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
                            reference = 1 * u.TeV)

    fit = SpectrumFit(obs_list, model)
    fit.run()
    
Now you can check the results by looking at the
`~gammapy.spectrum.SpectrumResult` that is attached to the
`~gammapy.spectrum.SpectrumFit` for each observation.

TODO : Add image from gammapy-extra illustrating the results


Interactive Sherpa Fit
======================

If you want to do something specific that is not handled by the
`~gammapy.spectrum.SpectrumFit` class you can always fit the PHA data directly
using `Sherpa`_. The following examples illustrates how to do this with the
example dataset used above.

.. code-block:: python

    from gammapy.datasets import gammapy_extra
    from sherpa.astro import datastack
    import sherpa.models as models

    pha1 = gammapy_extra.filename('datasets/hess-crab4_pha/pha_obs23592.fits')
    pha2 = gammapy_extra.filename('datasets/hess-crab4_pha/pha_obs23523.fits')
    phalist = ','.join([pha1, pha2])

    ds = datastack.DataStack()
    ds.load_pha(phalist)

    model = models.PowLaw1D('powlaw1d.default')
    model.ampl = 1e-20 
    model.ref = 1e9
    model.gamma = 2

    ds.set_source(model*1e20)

    for i in range(1, len(ds.datasets) + 1):
        datastack.ignore_bad(i)
        datastack.ignore_bad(i, 1)

    datastack.set_stat('wstat')
    ds.fit()
    datastack.covar()

The main downside of this approach is that the data must be present as FITS
files on disk. So if you create a `~gammapy.spectrum.SpectrumObservation` in a
python script you need to save it to disk and read it back with Sherpa.

