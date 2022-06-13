.. include:: ../references.txt

.. _fit-statistics:

Fit statistics
==============

Introduction
------------

This page describes the fit statistics used in gammapy. These fit statistics are
used by datasets to perform model fitting and parameter estimation.

Fit statistics in gammapy are all log-likelihood functions normalized like chi-squares,
i.e. if :math:`L` is the likelihood function used, they follow the expression
:math:`2 \times log L`.

All functions compute per-bin statistics. If you want the summed statistics for
all bins, call sum on the output array yourself.


.. _cash:

Cash : Poisson data with background model
-----------------------------------------

The number of counts, :math:`n`, is a Poisson random variable of mean value
:math:`\mu_{\mathrm{sig}} + \mu_{\mathrm{bkg}}`. The former is the expected
number of counts from the source (the signal), the latter is the number of
expected background counts, which is supposed to be known. We can write
the likelihood :math:`L` and applying the expression above, we obtain
the following formula for the Cash fit statistic:

.. math::
    C = 2 \times \left(\mu_{\mathrm{sig}} + \mu_{\mathrm{bkg}} - n \times log (\mu_{\mathrm{sig}}
    + \mu_{\mathrm{bkg}}) \right)

The Cash statistic is implemented in `~gammapy.stats.cash` and is used as a `stat`
function by the `~gammapy.datasets.MapDataset` and the `~gammapy.datasets.SpectrumDataset`.


Example
^^^^^^^
Here's an example for the `~gammapy.stats.cash` statistic::

    >>> from gammapy.stats import cash
    >>> data = [3, 5, 9]
    >>> model = [3.3, 6.8, 9.2]
    >>> cash(data, model)
    array([ -0.56353481,  -5.56922612, -21.54566271])
    >>> cash(data, model).sum()
    -27.67842364564512


.. _wstat:

WStat : Poisson data with background measurement
------------------------------------------------

In the absence of a reliable background model, it is possible to use a second
measurement containing only background to estimate it.

In the OFF region, which contains background only, the number of counts
:math:`n_{\mathrm{off}}` is a Poisson random variable of mean value
:math:`\alpha,\mu_{\mathrm{bkg}}`
In the ON region which contains signal and background contribution, the number
of counts, :math:`n_{\mathrm{on}}`, is a Poisson random variable of mean value
:math:`\mu_{\mathrm{sig}} +  \mu_{\mathrm{bkg}}`, where :math:`\alpha` is
the ratio of the ON and OFF region acceptances, and :math:`\mu_{\mathrm{bkg}}`
the mean background counts in the ON region.

It is possible define a likelihood function and marginalize it over the unknown
:math:`\mu_{\mathrm{bkg}}` to obtain :math:`\mu_{\mathrm{sig}}`.
This yields the so-called WStat or ON-OFF statistics which is traditionally used
for ON-OFF measurements in ground based gamma-ray astronomy.

The WStat fit statistics is given by the following formula:

.. math::
    W = 2 \big(\mu_{\mathrm{sig}} + (1 + 1/\alpha)\mu_{\mathrm{bkg}}
    - n_{\mathrm{on}} \log{(\mu_{\mathrm{sig}} + \mu_{\mathrm{bkg}})}
    - n_{\mathrm{off}} \log{(\mu_{\mathrm{bkg}}/\alpha)}\big)

To see how to derive it see the :ref:`wstat derivation <wstat_derivation>`.

The WStat statistic is implemented in `~gammapy.stats.wstat` and is used as a `stat`
function by the `~gammapy.datasets.MapDatasetOnOff` and the `~gammapy.datasets.SpectrumDatasetOnOff`.


Caveat
^^^^^^

- Since WStat takes into account background estimation uncertainties and makes no assumption such as a background model, it usually gives larger statistical uncertainties on the fitted parameters. If a background model exists, to properly compare with parameters estimated using the Cash statistics, one should include some systematic uncertainty on the background model.

- Note also that at very low counts, WStat is known to result in biased estimates. This can be an issue when studying the high energy behaviour of faint sources. When performing spectral fits with WStat, it is recommended to randomize observations and check whether the resulting fitted parameters distributions are consistent with the input values.


Example
^^^^^^^

The following table gives an overview over values that WStat takes in different
scenarios

    >>> from gammapy.stats import wstat
    >>> from astropy.table import Table
    >>> table = Table()
    >>> table['mu_sig'] = [0.1, 0.1, 1.4, 0.2, 0.1, 5.2, 6.2, 4.1, 6.4, 4.9, 10.2,
    ...                    16.9, 102.5]
    >>> table['n_on'] = [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 10, 20, 100]
    >>> table['n_off'] = [0, 1, 1, 10 , 10, 0, 5, 5, 20, 40, 2, 70, 10]
    >>> table['alpha'] = [0.01, 0.01, 0.5, 0.1 , 0.2, 0.2, 0.2, 0.01, 0.4, 0.4,
    ...                   0.2, 0.1, 0.6]
    >>> table['wstat'] = wstat(n_on=table['n_on'],
    ...                        n_off=table['n_off'],
    ...                        alpha=table['alpha'],
    ...                        mu_sig=table['mu_sig'])
    >>> table['wstat'].format = '.3f'
    >>> table.pprint()
    mu_sig n_on n_off alpha wstat
    ------ ---- ----- ----- ------
       0.1    0     0  0.01  0.200
       0.1    0     1  0.01  0.220
       1.4    0     1   0.5  3.611
       0.2    0    10   0.1  2.306
       0.1    0    10   0.2  3.846
       5.2    5     0   0.2  0.008
       6.2    5     5   0.2  0.736
       4.1    5     5  0.01  0.163
       6.4    5    20   0.4  7.125
       4.9    5    40   0.4 14.578
      10.2   10     2   0.2  0.034
      16.9   20    70   0.1  0.656
     102.5  100    10   0.6  0.663

Notes
^^^^^

All above formulae are equivalent to what is given on the `XSpec manual
statistics page`_ with the substitutions:

.. math::
    \mu_{\mathrm{sig}} = t_s \cdot m_i \\
    \mu_{\mathrm{bkg}} = t_b \cdot m_b \\
    \alpha = t_s / t_b  \\

Further references
------------------

* `Sherpa statistics page`_
* `XSpec manual statistics page`_
