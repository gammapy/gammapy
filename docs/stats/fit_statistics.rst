.. _fit-statistics:

Fit statistics
==============

Introduction
------------

This page describes common fit statistics used in gamma-ray astronomy.
Results were tested against results from the
`Sherpa <http://cxc.harvard.edu/sherpa/>`_ and
`XSpec <https://heasarc.gsfc.nasa.gov/xanadu/xspec/>`_
X-ray analysis packages.

.. Likelihood defined per bin -> take sum
.. Stat = -2 log (L)
.. Code example

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
If you not only have a  measurement of counts  ``n_on`` in the signal region,
but also a measurement ``n_off`` in a background region you can write down the
likelihood formula as 

.. math::

    L (n_{on}, n_{off}, \alpha; \mu_{sig}, \mu_{bkg}) =
         \frac{(\mu_{sig}+\alpha \mu_{bkg})^{n_{on}}}{n_{on} !}
        \exp{(-(\mu_{sig}+\alpha \mu_{bkg}))}\times 
        \frac{(\mu_{bkg})^{n_{off}}}{n_{off} !}\exp{(-\mu_{bkg})},

where :math:`\mu_{sig}` is the number of expected counts in the signal regions,
and :math:`\mu_{bkg}` is the number of expected counts in the background
region, as defined in the :ref:`stats-introduction`. By taking two time the
negative log likelihood and neglecting model independent and thus constant
terms, we define the **WStat**.

.. math::

    W = 2 (\mu_{sig} + (1 + \alpha)\mu_{bkg}
    - n_{on} \log{(\mu_{sig} + \alpha \mu_{bkg})}
    - n_{off} \log{(\mu_{bkg})})

Most of the times you probably won't have a model in order to get
:math:`\mu_{bkg}`. The strategy in this case is to treat :math:`\mu_{bkg}` as
so-called nuisance parameter, i.e. a free parameter that is of no physical
interest.  Of course you don't want an additional free parameter for each bin
during a fit. Therefore one calculates an estimator for :math:`\mu_{bkg}` by
analytically minimizing the likelihood function. This is called 'profile
likelihood'.

.. math::
    \frac{\mathrm d \log L}{\mathrm d \mu_{bkg}} = 0
    
This yields a quadratic equation for :math:`\mu_{bkg}` 

.. math::
    \frac{\alpha n_{on}}{mu_{sig}+\alpha \mu_{bkg}} +
    \frac{n_{off}}{\mu_{bkg}} - (\alpha + 1) = 0

with the solution

.. math::

    \mu_{bkg} = \frac{C + D}{2\alpha(\alpha + 1)}

where

.. math::

    C = \alpha(n_{on} + n_{off}) - (\alpha+1)\mu_{sig} \\
    D^2 = C^2 + 4 (\alpha+1)\alpha n_{off} \mu_{sig}


The best-fit value of the WStat as defined now contains no information about
the goodness of the fit. In order to provide such an estimate, we can add a
constant term to the WStat, namely twice the log likelihood of the data
``n_on`` and ``n_off`` under the expectation of ``n_on`` and ``n_off``,

.. math::

     2 \log L (n_{on}, n_{off}; n_{on}, n_{off}) =
         2 (n_{on} ( \log n_{on} - 1 ) + n_{off} ( \log n_{off} - 1))


In doing so, we are computing the likelihood ratio:

.. math::

    -2 \log \frac{L(n_{on},n_{off},\alpha; \mu_{sig},\mu_{bkg})}
        {L(n_{on},n_{off};n_{on},n_{off})}

Intuitively, this log-likelihood ratio should asymptotically behave like a
chi-square with ``m-n`` degrees of freedom, where ``m`` is the number of
measurements and ``n`` the number of model parameters.

Hence, we rewrite WStat as:

.. math::

    W = 2 (\mu_{sig} + (1 + \alpha)\mu_{bkg} - n_{on} - n_{off}
    - n_{on} (\log{(\mu_{sig} + \alpha \mu_{bkg}) - \log{(n_{on})}})
    - n_{off} (\log{(\mu_{bkg})} - \log{(n_{off})}))


TODO: Explain how to handle corner cases


Further references
------------------
* `Sherpa statistics page <http://cxc.cfa.harvard.edu/sherpa/statistics>`_ 
* `XSpec manual statistics page
  <http://heasarc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html>`_
 
