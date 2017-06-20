.. _time:
*******************************************
Period detection  (`gammapy.time.lombscargle`) 
*******************************************
.. currentmodule:: gammapy.time.lombscargle
Introduction
============
`gammapy.time.lombscargle` establishes the methods for time series analysis. `gammapy.time.lombscargle.lombscargle` computes the Lomb-Scargle periodogram and the spectral window function on a light curve and returns the period of an intrinsic periodic beahviour in respect of different significance criteria and an adjustable false alarm probability. The result can be plotted with `gammapy.time.lombscargle.plotting`.
The Lomb Scargle algorithm is provided by `astropy.stats.LombScargle`.
Getting Started
===============
Input
-----
`lombscargle.lombscargle` takes a light curve in format time, flux and flux error.
Additionally, the linear frequency grid can be narrowed down with an oversampling factor.
For the significance criteria, the false alarm probability need to be defined.
For the bootstrap resamling, the number of resamlings has to be defined.
`lombscargle.plotting` takes the output of `lombscargle.lombscargle` as input.
Output
------
`lombscargle.lombscargle` returns the frequency grid, the periodogram peaks of the Lomb-Scargle periodogram and the spectral window function, the percentiles of all significance criteria for a specified false alarm probability, as well as the best period if found.
Reference
=========
