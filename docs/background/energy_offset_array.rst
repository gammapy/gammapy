.. _energyoffset_array:

EnergyOffset Array
=================

The `~gammapy.background.EnergyOffsetArray` class represents a 2D array *(energy,offset)* that is filled with an eventlist. For a set of observations, by giving an energy binning and an offset binning, you fill the events in this histogram.


Four Crab observations are located in the ``gammapy-extra`` repository as
examples:
`hess_events_simulated_023523.fits`_ , `hess_events_simulated_023526.fits`_ , `hess_events_simulated_023559.fits`_ and `hess_events_simulated_023592.fits`_

An example script of how to fill this Array from these four observations and plots the result is given in the ``examples`` directory:
:download:`fill_and_plot_energy_offset_array.py <../../examples/fill_and_plot_energy_offset_array.py>`

.. plot:: ../examples/fill_and_plot_energy_offset_array.py
   :include-source:



.. _hess_events_simulated_023523.fits: https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-crab4/hess_events_simulated_023523.fits
.. _hess_events_simulated_023526.fits: https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-crab4/hess_events_simulated_023526.fits
.. _hess_events_simulated_023559.fits: https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-crab4/hess_events_simulated_023559.fits
.. _hess_events_simulated_023592.fits: https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-crab4/hess_events_simulated_023592.fits
