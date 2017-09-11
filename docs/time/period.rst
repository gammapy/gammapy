*****************************
Period detection and plotting
*****************************

Introduction
============

`~gammapy.time.lomb_scargle` establishes methods for period detection in unevenly sampled time series.
It computes the Lomb-Scargle periodogram and the spectral window function on a light curve and returns
the period of an intrinsic periodic beahviour in respect of different significance criteria and an
adjustable false alarm probability. The result can be plotted with `~gammapy.time.plot_periodogram`.
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

`~gammapy.time.lomb_scargle` returns the frequency grid, the periodogram peaks of the
Lomb-Scargle periodogram and the spectral window function, the percentiles of all
significance criteria for a specified false alarm probability, as well as the best period if found.

Example
=======

An example of detecting a period is shown in the figure below.
The light curve is from the X-ray binary LS 5039 observed with H.E.S.S. at energies above 0.1 TeV in 2005 [1]_.
The Lomb-Scargle reveals the period of :math:`(3.907 \pm 0.001)` days in agreement with [1]_ and [2]_.

.. gp-extra-image:: lomb_scargle_long.png
    :width: 100 %
    :alt: alternate text
    :align: left

The periodogram shows fluctuations for small periods and a smoothed behaviour for longer periods that are
due to sampling effects and aliasing.
If this is the case, `max_period` can be defined to limit the period range for the analysis.
This way, the resoultion can be increased with equal computation time.

.. gp-extra-image:: lomb_scargle_short.png
    :width: 100 %
    :alt: alternate text
    :align: left

The periodogram has many spurious peaks, which are due to several factors:

1. Errors in observations lead to leakage of power from the true peaks.
2. The signal is not a perfect sinusoid, so additional peaks can indicate higher-frequency components in the signal.
3. The spectral window function shows two prominent peaks around one and 27 days.
   The first one arises from the nightly observation cycle, the second from the lunar phase.
   Thus, aliases are expected to appear at :math:`f_{{alias}} = f_{{true}} + n f_{{window}}`
   for integer values of :math:`n`. For the peak in the spectral window function at
   :math:`f_{{window}} = 1 day^{{-1}}`, this corresponds to the third highest peak in
   the periodogram at :math:`p_{{alias}} = 0.796`.

The returned significance must be used with caution. If the resolution is too rough, several periods
will be detected with a significance of 100 per cent. Thus, an eyesight inspection is obligatory.

.. [1] F. Aharonian, 3.9 day orbital modulation in the TeV gamma-ray flux and spectrum from the X-ray binary LS 5039,
   `Link <https://www.aanda.org/articles/aa/pdf/forth/aa5940-06.pdf>`_ 
.. [2] J. Casares, A possible black hole in the gamma-ray microquasar LS 5039,
   `Link <https://academic.oup.com/mnras/article/364/3/899/1187228/A-possible-black-hole-in-the-ray-microquasar-LS>`_
