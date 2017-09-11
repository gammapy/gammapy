*****************************
Period detection and plotting
*****************************

.. currentmodule:: gammapy.time.lomb_scargle
.. currentmodule:: gammapt.time.robust_periodogram
.. currentmodule:: gammapy.time.plot_periodogram

Introduction
============
`~gammapy.time.lomb_scargle` and `~gammapy.time.robust_periodogram` establish methods for period detection in unevenly sampled time series.
They compute the periodogram and the spectral window function of a light curve.
As model, a single sinusoidal is fitted with linear (Lomb-Scargle) or robust (robust periodogram) least square regression.
The period and significance of an intrinsic periodic beahviour is returned in respect of different significance criteria.
The results can be plotted with `gammapy.time.plot_periodogram`.
The Lomb Scargle algorithm is provided by `astropy.stats.LombScargle`.
The robust loss function are provided by `scipy.optimize.least_squares`.

Getting Started
===============
Input
-----
`~gammapy.time.lomb_scargle` and `~gammapy.time.robust_periodogram` take a light curve in format `time`, `flux` and `flux error`.
For the robust periodogram, the loss function `loss` and the loss scale parameter `scale` need to be given.
The trial period grid can optionally be specified by the resolution `dt` and a maximum period `max_period`.
If these parameters are not given, `dt` will be set by the Nyquist frequency and `max_period` by the length of the light curve.
A significance criteria can be chosen from `criteria`.
If not, all significance criteria will be used for the analysis.
For the bootstrap resamling, the number of resamlings `n_bootstrap` can be defined.
Its default value is set to 100.
`~gammapy.time.plot_periodogram` takes the output of `~gammapy.time.lomb_scargle` and `~gammapy.time.robust_periodogram` as input.

Output
------
`~gammapy.time.lomb_scargle` and `~gammapy.time.robust_periodogram` return the period grid `pgrid`, the periodogram peaks of the periodogram `psd` and the spectral window function `swf`.
The period of the highest periodogram peak `period` is assumed as the period of an intrinsic periodic behaviour.
Its inverse false alarm probability `significance` is obtained for all utilised significance criteria by the quantile of the respective periodogram peak distribution.

Example
=======
An example of detecting a period is shown in the figure below. The light curve is from the X-ray binary LS 5039 observed with H.E.S.S. at energies above 0.1 TeV in 2005 [1]_.
The Lomb-Scargle reveals the period of :math:`(3.907 \pm 0.001)` days in agreement with [1]_ and [2]_.

.. gp-extra-image:: lomb_scargle_long.png
   :width: 100 %
   :alt: alternate text
   :align: left

The periodogram shows fluctuations for small periods and a smoothed behaviour for longer periods that are due to sampling effects and aliasing.
=======

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

The returned significance of the periodogram peak must be used with caution.
A value of :math:`100 \%` indicates that the respective significance criteria is not an appropriate choice for this light curve.
For the bootstrap, the number of resamplings can be raised to obtain a higher resolution in the significance.
Alternatively, more restrictive significance criteria have to be used.

.. [1] F. Aharonian, 3.9 day orbital modulation in the TeV gamma-ray flux and spectrum from the X-ray binary LS 5039,
   `Link <https://www.aanda.org/articles/aa/pdf/forth/aa5940-06.pdf>`_
.. [2] J. Casares, A possible black hole in the gamma-ray microquasar LS 5039,
   `Link <https://academic.oup.com/mnras/article/364/3/899/1187228/A-possible-black-hole-in-the-ray-microquasar-LS>`_
