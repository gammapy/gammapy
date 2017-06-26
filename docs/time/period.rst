.. _time:
**********************************************************************************************
Period detection  (`gammapy.time.lomb_scargle`) and plotting (`gammapy.time.plot_periodogram`)
**********************************************************************************************
.. currentmodule:: gammapy.time.lomb_scargle
.. currentmodule:: gammapy.time.plot_periodogram
Introduction
============
`~gammapy.time.lomb_scargle` establishes methods for period detection in unevenly sampled time series. It computes the Lomb-Scargle periodogram and the spectral window function on a light curve and returns the period of an intrinsic periodic beahviour in respect of different significance criteria and an adjustable false alarm probability. The result can be plotted with `gammapy.time.plot_periodogram`.
The Lomb Scargle algorithm is provided by `astropy.stats.LombScargle`.

Getting Started
===============
Input
-----
`~gammapy.time.lomb_scargle` takes a light curve in format time, flux and flux error.
Additionally, the linear frequency grid can be narrowed down with an oversampling factor.
For the significance criteria, the false alarm probability need to be defined.
For the bootstrap resamling, the number of resamlings has to be defined.
`~gammapy.time.plot_periodogram` takes the output of `~gammapy.time.lomb_scargle` as input.

Output
------
`~gammapy.time.lomb_scargle` returns the frequency grid, the periodogram peaks of the Lomb-Scargle periodogram and the spectral window function, the percentiles of all significance criteria for a specified false alarm probability, as well as the best period if found.

Reference
=========
