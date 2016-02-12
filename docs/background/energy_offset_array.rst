.. _energyoffset_array:

EnergyOffset Array
==================

The `~gammapy.background.EnergyOffsetArray` class represents a 2D array *(energy,offset)* that is filled with an eventlist.
For a set of observations, by giving an energy binning and an offset binning, you fill the events in this histogram.

Four Crab observations are located at ``$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2``

An example script of how to fill this array from these four observations and plots the result is given in the ``examples`` directory:
:download:`example_energy_offset_array.py <../../examples/example_energy_offset_array.py>`

.. plot:: ../examples/example_energy_offset_array.py
   :include-source:

