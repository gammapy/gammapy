.. include:: ../references.txt

.. _spectrum-simulation:

******************
Simulating Spectra 
******************

.. currentmodule:: gammapy.spectrum

In the following, a few example are given to demonstrate the
`~gammapy.spectrum.SpectrumSimulation` class.

Getting Started
===============

This section show you how to simulate a `~gammapy.spectrum.SpectrumObservation`
given a source model and instrument response functions. The latter will also be
simulated to make this tutorial independent of any real data. No background
measurement is simulated in this example. Simulating the IRFs works like this

.. code-block:: python

    import numpy as np
    import astropy.units as u
    from gammapy.irf import EnergyDispersion, EffectiveAreaTable
        
    e_true = np.logspace(-2, 2.5, 109) * u.TeV
    e_reco = np.logspace(-2,2, 79) * u.TeV

    edisp = EnergyDispersion.from_gauss(e_true=e_true, e_reco=e_reco, sigma=0.2)
    aeff = EffectiveAreaTable.from_parametrization(energy=e_true)

We'll have a look at the simulated IRFs in a second. But first we define a
`~gammapy.spectrum.models.SpectralModel`,


.. code-block:: python

    from gammapy.spectrum.models import PowerLaw

    index = 2.3 * u.Unit('')
    amplitude = 2.5 * 1e-12 * u.Unit('cm-2 s-1 TeV-1')
    reference = 1 * u.TeV

    model = PowerLaw(index=index, amplitude=amplitude, reference=reference)
    
and simulate one `~gammapy.spectrum.SpectrumObservation` for one 4 hour
observation. We set the low energy threshold to 20% of the peak effective area
and the high energy threshold to 60 TeV.

.. code-block:: python

    from gammapy.spectrum import SpectrumSimulation
    import matplotlib.pyplot as plt

    livetime = 4 * u.h
    lo_threshold = aeff.find_energy(0.2 * aeff.max_area)
    hi_threshold = 60 * u.TeV

    sim = SpectrumSimulation(aeff=aeff, edisp=edisp, model=model, livetime=livetime)
    
    obs = sim.simulate_obs(obs_id=42, seed=42,
                           lo_threshold=lo_threshold,
                           hi_threshold=hi_threshold)
    obs.peek()
    plt.show()

.. TODO: link to image https://github.com/gammapy/gammapy-extra/blob/master/figures/simulated_obs.png
.. .. image:: simulated_obs.png


Simulating Many Spectra
=======================

To get a more quantitative impression of the
`~gammapy.spectrum.SpectrumSimulation` we are now going to simulate ``n``
spectra, fit a model to them and compare the model parameter estimates with the
input values. Please read :ref:`spectral_fitting` if you want to know how to
fit a `~gammapy.spectrum.SpectrumObservation`. We'll use the model and the
simulated IRFs from the Getting Started section in the following.


.. code-block:: python

    # mute the sherpa logger
    import sherpa
    import logging

    sherpa_logger = logging.getLogger('sherpa')
    sherpa_logger.setLevel(50)

    n_sim = 100

    results = []
    for obs_id in range(n_sim):
        obs = sim.simluate_obs(obs_id=obs_id, seed=obs_id)

        results.append(dict(
            obs_id=obs_id,
            # TODO: store other results
        ))

        # At the moment you have to save the obs to disk so sherpa can read it
        # There's an open PR to fix this continue example once this is done


To be continued ...
