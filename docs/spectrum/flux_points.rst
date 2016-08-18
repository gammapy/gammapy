.. include:: ../references.txt

.. _flux-point-computation:

**********************
Flux point computation
**********************

.. currentmodule:: gammapy.spectrum

In the following you will see how to compute
`~gammapy.spectrum.DifferentialFluxPoints` given a global model and a
`~gammapy.spectrum.SpectrumObservation`. We will use the example dataset in
`gammapy-extra
<https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-crab4_pha>`_.
The flux points binning is chosen as 5 equally log-spaced bins between the
observation thresholds. In order to obtain the global model we first perform
the global fit again, for more info see :ref:`spectral_fitting`.

.. code-block:: python

    import astropy.units as u
    from gammapy.spectrum import SpectrumObservation, SpectrumFit, DifferentialFluxPoints
    from gammapy.spectrum.models import PowerLaw
    from gammapy.utils.energy import EnergyBounds

    pha = "$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits"
    obs = SpectrumObservation.read(pha)

    model = PowerLaw(index = 2 * u.Unit(''),
                     amplitude = 10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
                     reference = 1 * u.TeV)

    global_fit = SpectrumFit(obs_list=[obs], model=model)
    global_fit.fit()

    global_model = global_fit.result[0].fit.model
    binning = EnergyBounds.equal_log_spacing(obs.lo_threshold, obs.hi_threshold, 5) 
    
    points = DifferentialFluxPoints.compute(model=global_model,
                                            binning=binning, 
                                            obs_list = [obs])

    

Note, that in this case (where we just performed the global fit) we can get the
flux points more conveniently as

.. code-block:: python

    import astropy.units as u
    from gammapy.spectrum import SpectrumObservation, SpectrumFit
    from gammapy.spectrum.models import PowerLaw
    from gammapy.utils.energy import EnergyBounds
    import matplotlib.pyplot as plt

    pha = "$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits"
    obs = SpectrumObservation.read(pha)

    model = PowerLaw(index = 2 * u.Unit(''),
                     amplitude = 10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
                     reference = 1 * u.TeV)

    fit = SpectrumFit(obs_list=[obs], model=model)
    fit.fit()

    binning = EnergyBounds.equal_log_spacing(obs.lo_threshold, obs.hi_threshold, 5) 
    fit.compute_fluxpoints(binning=binning)
    fit.result[0].plot_spectrum()
    plt.show()

