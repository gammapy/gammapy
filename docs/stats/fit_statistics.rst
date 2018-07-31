.. include:: ../references.txt

.. _fit-statistics:

Fit statistics
==============

Introduction
------------

This page describes common fit statistics used in gamma-ray astronomy. Results
were tested against results from the `Sherpa`_ and `XSpec`_ X-ray analysis
packages.

All functions compute per-bin statistics. If you want the summed statistics for
all bins, call sum on the output array yourself. Here's an example for the
`~gammapy.stats.cash` statistic::

    >>> from gammapy.stats import cash
    >>> data = [3, 5, 9]
    >>> model = [3.3, 6.8, 9.2]
    >>> cash(data, model)
    array([ -0.56353481,  -5.56922612, -21.54566271])
    >>> cash(data, model).sum()
    -27.678423645645118

Gaussian data
-------------

TODO

Poisson data
------------

TODO

.. _wstat:

Poisson data with background measurement
----------------------------------------

If you not only have a  measurement of counts  :math:`n_{\mathrm{on}}` in the
signal region, but also a measurement :math:`n_{\mathrm{off}}` in a background
region you can write down the likelihood formula as

.. math::

    L (n_{\mathrm{on}}, n_{\mathrm{off}}, \alpha; \mu_{\mathrm{sig}},
    \mu_{\mathrm{bkg}}) = \frac{(\mu_{\mathrm{sig}}+\alpha
    \mu_{\mathrm{bkg}})^{n_{\mathrm{on}}}}{n_{\mathrm{on}} !}
    \exp{(-(\mu_{\mathrm{sig}}+\alpha \mu_{\mathrm{bkg}}))}\times
    \frac{(\mu_{\mathrm{bkg}})^{n_{\mathrm{off}}}}{n_{\mathrm{off}}
    !}\exp{(-\mu_{\mathrm{bkg}})},

where :math:`\mu_{\mathrm{sig}}` is the number of expected counts in the signal
regions, and :math:`\mu_{\mathrm{bkg}}` is the number of expected counts in the
background region, as defined in the :ref:`stats-introduction`. By taking two
time the negative log likelihood and neglecting model independent and thus
constant terms, we define the **WStat**.

.. math::

    W = 2 \big(\mu_{\mathrm{sig}} + (1 + \alpha)\mu_{\mathrm{bkg}}
    - n_{\mathrm{on}} \log{(\mu_{\mathrm{sig}} + \alpha \mu_{\mathrm{bkg}})}
    - n_{\mathrm{off}} \log{(\mu_{\mathrm{bkg}})}\big)

In the most general case, where :math:`\mu_{\mathrm{src}}` and
:math:`\mu_{\mathrm{bkg}}` are free the minimum of :math:`W` is at

.. math::

    \mu_{\mathrm{sig}} = n_{\mathrm{on}} - \alpha\,n_{\mathrm{off}}   \\
    \mu_{\mathrm{bkg}} = n_{\mathrm{off}}


Profile Likelihood
^^^^^^^^^^^^^^^^^^

Most of the times you probably won't have a model in order to get
:math:`\mu_{\mathrm{bkg}}`. The strategy in this case is to treat
:math:`\mu_{\mathrm{bkg}}` as so-called nuisance parameter, i.e. a free
parameter that is of no physical interest.  Of course you don't want an
additional free parameter for each bin during a fit. Therefore one calculates an
estimator for :math:`\mu_{\mathrm{bkg}}` by analytically minimizing the
likelihood function. This is called 'profile likelihood'.

.. math::
    \frac{\mathrm d \log L}{\mathrm d \mu_{\mathrm{bkg}}} = 0

This yields a quadratic equation for :math:`\mu_{\mathrm{bkg}}`

.. math::
    \frac{\alpha\,n_{\mathrm{on}}}{\mu_{\mathrm{sig}}+\alpha
    \mu_{\mathrm{bkg}}} + \frac{n_{\mathrm{off}}}{\mu_{\mathrm{bkg}}} - (\alpha
    + 1) = 0

with the solution

.. math::

    \mu_{\mathrm{bkg}} = \frac{C + D}{2\alpha(\alpha + 1)}

where

.. math::

    C = \alpha(n_{\mathrm{on}} + n_{\mathrm{off}}) - (\alpha+1)\mu_{\mathrm{sig}} \\
    D^2 = C^2 + 4 (\alpha+1)\alpha n_{\mathrm{off}} \mu_{\mathrm{sig}}

Goodness of fit
^^^^^^^^^^^^^^^

The best-fit value of the WStat as defined now contains no information about the
goodness of the fit. We consider the likelihood of the data
:math:`n_{\mathrm{on}}` and :math:`n_{\mathrm{off}}` under the expectation of
:math:`n_{\mathrm{on}}` and :math:`n_{\mathrm{off}}`,

.. math::

    L (n_{\mathrm{on}}, n_{\mathrm{off}}; n_{\mathrm{on}}, n_{\mathrm{off}}) =
    \frac{n_{\mathrm{on}}^{n_{\mathrm{on}}}}{n_{\mathrm{on}} !}
    \exp{(-n_{\mathrm{on}})}\times
    \frac{n_{\mathrm{off}}^{n_{\mathrm{off}}}}{n_{\mathrm{off}} !}
    \exp{(-n_{\mathrm{off}})}

and add twice the log likelihood

.. math::

     2 \log L (n_{\mathrm{on}}, n_{\mathrm{off}}; n_{\mathrm{on}},
     n_{\mathrm{off}}) = 2 (n_{\mathrm{on}} ( \log{(n_{\mathrm{on}})} - 1 ) +
     n_{\mathrm{off}} ( \log{(n_{\mathrm{off}})} - 1))

to WStat. In doing so, we are computing the likelihood ratio:

.. math::

    -2 \log \frac{L(n_{\mathrm{on}},n_{\mathrm{off}},\alpha;
    \mu_{\mathrm{sig}},\mu_{\mathrm{bkg}})}
    {L(n_{\mathrm{on}},n_{\mathrm{off}};n_{\mathrm{on}},n_{\mathrm{off}})}

Intuitively, this log-likelihood ratio should asymptotically behave like a
chi-square with ``m-n`` degrees of freedom, where ``m`` is the number of
measurements and ``n`` the number of model parameters.

Final result
^^^^^^^^^^^^

.. math::

    W = 2 \big(\mu_{\mathrm{sig}} + (1 + \alpha)\mu_{\mathrm{bkg}} -
    n_{\mathrm{on}} - n_{\mathrm{off}} - n_{\mathrm{on}}
    (\log{(\mu_{\mathrm{sig}} + \alpha \mu_{\mathrm{bkg}}) -
    \log{(n_{\mathrm{on}})}}) - n_{\mathrm{off}} (\log{(\mu_{\mathrm{bkg}})} -
    \log{(n_{\mathrm{off}})})\big)

Special cases
^^^^^^^^^^^^^

The above formula is undefined if :math:`n_{\mathrm{on}}` or
:math:`n_{\mathrm{off}}` are equal to zero, because of the :math:`n\log{{n}}`
terms, that were introduced by adding the goodness of fit terms. These cases are
treated as follows.

If :math:`n_{\mathrm{on}} = 0` the likelihood formulae read

.. math::

    L (0, n_{\mathrm{off}}, \alpha; \mu_{\mathrm{sig}}, \mu_{\mathrm{bkg}}) =
    \exp{(-(\mu_{\mathrm{sig}}+\alpha \mu_{\mathrm{bkg}}))}\times
    \frac{(\mu_{\mathrm{bkg}})^{n_{\mathrm{off}}}}{n_{\mathrm{off}}
    !}\exp{(-\mu_{\mathrm{bkg}})},

and

.. math::

    L (0, n_{\mathrm{off}}; 0, n_{\mathrm{off}}) =
    \frac{n_{\mathrm{off}}^{n_{\mathrm{off}}}}{n_{\mathrm{off}} !}
    \exp{(-n_{\mathrm{off}})}

WStat is derived by taking 2 times the negative log likelihood and adding the
goodness of fit term as ever

.. math::

    W = 2 \big(\mu_{\mathrm{sig}} + (1 + \alpha)\mu_{\mathrm{bkg}} -
    n_{\mathrm{off}} - n_{\mathrm{off}} (\log{(\mu_{\mathrm{bkg}})} -
    \log{(n_{\mathrm{off}})})\big)

Note that this is the limit of the original Wstat formula for
:math:`n_{\mathrm{on}} \rightarrow 0`.

The analytical result for
:math:`\mu_{\mathrm{bkg}}` in this case reads:

.. math::

    \mu_{\mathrm{bkg}} = \frac{n_{\mathrm{off}}}{\alpha + 1}

When inserting this into the WStat we find the simplified expression.

.. math::

    W = 2\big(\mu_{\mathrm{sig}} + n_{\mathrm{off}} \log{(1 + \alpha)}\big)

If :math:`n_{\mathrm{off}} = 0` Wstat becomes

.. math::

    W = 2 \big(\mu_{\mathrm{sig}} + (1 + \alpha)\mu_{\mathrm{bkg}} -
    n_{\mathrm{on}} - n_{\mathrm{on}} (\log{(\mu_{\mathrm{sig}} + \alpha
    \mu_{\mathrm{bkg}}) - \log{(n_{\mathrm{on}})}})

and

.. math::

    \mu_{\mathrm{bkg}} = \frac{n_{\mathrm{on}}}{1+\alpha} -
    \frac{\mu_{\mathrm{sig}}}{\alpha}

For :math:`\mu_{\mathrm{sig}} > n_{\mathrm{on}} (\frac{\alpha}{1 + \alpha})`,
:math:`\mu_{\mathrm{bkg}}` becomes negative which is unphysical.

Therefore we distinct two cases. The physical one where

:math:`\mu_{\mathrm{sig}} < n_{\mathrm{on}} (\frac{\alpha}{1 + \alpha})`.

is straightforward and gives

.. math::

    W = -2\big(\mu_{\mathrm{sig}} \left(\frac{1}{\alpha}\right) +
    n_{\mathrm{on}} \log{\left(\frac{\alpha}{1 + \alpha}\right)\big)}

For the unphysical case, we set :math:`\mu_{\mathrm{bkg}}=0` and arrive at

.. math::

    W = 2\big(\mu_{\mathrm{sig}} + n_{\mathrm{on}}(\log{(n_{\mathrm{on}})} -
    \log{(\mu_{\mathrm{sig}})} - 1)\big)


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
