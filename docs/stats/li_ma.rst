.. li_ma:

Li \& Ma methods of estimate errors and significance
====================================================

Li, T.-P. and Ma, Y.-Q. [LiMa1983]_ have derived procedures to derive significance of
a gamma-ray signal based on a likelihood ratio method. This method, verified by Monte-Carlo
simulations, is very accurate in the Gaussian regime and also in the Poisson regime. This
methodology have been extended to compute errors on excess, as well as the asymmetric errors
(positive and negative).

The functions ``gammapy.stats.lm_*`` give you access to a numerical solution to
the Li&Ma formulae.

The significance estimate of Li \& Ma (eq. 17)
--------------------------------------------

The standard use of their work is the computation of the significance of an excess.
The method used the likelihood ratio :math:`\lambda = \frac{L(X|{H_1})}{L(X|H_0)}`,
where the null hypothesis :math:`H_0` assumes no signal and the hypothesis :math:`H_1`
assumes a gamma-ray excess, using the Poisson statistics. Assymptotically, :math:`-2 \ln \lambda` follows a :math:`\chi^2`
distribution with 1 degree of freedom.

Extensive Monte-Carlo simulations have been used to check the statistical behaviour of the Li&Ma formula for different regime (weak and large
counts, small and big :math:`\alpha = \frac{Acceptance_{ON}}{Acceptance_{OFF}}` values).

Let's note that the Li&Ma formula assumes a correct estimation of the :math:`\alpha` parameter. If a
large statistical error exists on it, the :math:`\sigma` estimation is biased and its
distribution does not follow anymore a Gaussian with a width of 1. Its distribution is enlarged
significantly.

.. code-block:: python

    >>> from gammapy.stats.li_ma import lm_significance_on_off
    >>> non = 8
    >>> noff = 60
    >>> alpha = 0.1
    >>> lm_significance_on_off(non, noff, alpha)
    0.7368243826529305

    >>> non = [8, 13]
    >>> noff = [60, 45]
    >>> alpha = [0.1, 1.]
    >>> lm_significance_on_off(non, noff, alpha)
    array([ 0.73682438, -4.32226689])

An other equivalent way to have this estimate if the following:

.. code-block:: python

    >>> from gammapy.stats.poisson import significance_on_off
    >>> non = 8
    >>> noff = 60
    >>> alpha = 0.1
    >>> lm_significance_on_off(non, noff, alpha)
    0.7368243826529305

The errors estimate based on Li \& Ma
--------------------------------------

The likelihood estimation of Li&Ma has been used to estimate errors on an excess, as
well as its positive and negative estimations. Its implementation uses the computation of the
likelihood value, via ``gammapy.stats.lm_loglikelihood``. The downward, upward and averaged
excess uncertainties of an on-off observation can be then computed as follow:

.. code-block:: python

    >>> from gammapy.stats.li_ma import lm_dexcess_down, lm_dexcess_up
    >>> non = 12
    >>> noff = 40
    >>> alpha = 0.25
    >>> lm_dexcess_down(non, noff, alpha)
    3.56689453125
    >>> lm_dexcess_up(non, noff, alpha)
    4.0869140625

And the average error is then:

.. code-block:: python

    >>> from gammapy.stats.li_ma import lm_dexcess
    >>> non = 12
    >>> noff = 40
    >>> alpha = 0.25
    >>> lm_dexcess(non, noff, alpha)
    3.826904296875


An other equivalent way to have this estimate if the following:

.. code-block:: python

    >>> from gammapy.stats.poisson import excess_error
    >>> non = 8
    >>> noff = 60
    >>> alpha = 0.1
    >>> excess_error(non, noff, alpha, method='lima')
    2.947509765625
