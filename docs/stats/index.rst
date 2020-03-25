.. _stats:

******************
stats - Statistics
******************

.. currentmodule:: gammapy.stats

.. _stats-introduction:

Introduction
============

`gammapy.stats` holds statistical estimators, fit statistics and algorithms
commonly used in gamma-ray astronomy.

It is mostly concerned with the evaluation of one or several observations that
count events in a given region and time window, i.e. with Poisson-distributed
counts measurements.

For on-off methods we will use the following variable names following the
notation in [Cousins2007]_:

================= ====================== ====================================================
Variable          Dataset attribute name Definition
================= ====================== ====================================================
``n_on``          ``counts``             Total observed counts in the on region
``n_off``         ``counts_off``         Total observed counts in the off region
``mu_on``         ``npred``              Total expected counts in the on region
``mu_off``        ``npred_off``          Total expected counts in the off region
``mu_sig``        ``npred_sig``          Signal expected counts in the on region
``mu_bkg``        ``npred_bkg``          Background expected counts in the on region
``a_on``          ``acceptance``         Relative background exposure in the on region
``a_off``         ``acceptance_off``     Relative background exposure in the off region
``alpha``         ``alpha``              Background efficiency ratio ``a_on`` / ``a_off``
``n_bkg``         ``background``         Background estimate in the on region
================= ====================== ====================================================

The following formulae show how an on-off measurement :math:`(n_{on}, n_{off})`
is related to the quantities in the above table:

.. math::
    n_{on} \sim Pois(\mu_{on})\text{ with }\mu_{on} = \mu_s + \mu_b
    n_{off} \sim Pois(\mu_{off})\text{ with }\mu_{off} = \mu_b / \alpha\text{ with }\alpha = a_{on} / a_{off}

With the background estimate in the on region

.. math::
   n_{bkg} = \alpha\ n_{off},

the maximum likelihood estimate of a signal excess is

.. math::
    n_{excess} = n_{on} - n_{bkg}.

When the background is known and there is only an "on" region (sometimes also
called "source region"), we use the variable names ``n_on``, ``mu_on``,
``mu_sig`` and ``mu_bkg``.

These are references describing the available methods: [LiMa1983]_, [Cash1979]_,
[Stewart2009]_, [Rolke2005]_, [Feldman1998]_, [Cousins2007]_.

Getting Started
===============

General notions
---------------

Counts and fit statistics
^^^^^^^^^^^^^^^^^^^^^^^^^

Gamma-ray measurements are counts, ``n_on``, containing both signal and background events.

Estimation of number of signal events or of quantities in physical models is done through
Poisson likelihood functions, the fit statistics. In gammapy, they are all log-likelihood
functions normalized like chi-squares, i.e. if :math:`L` is the likelihood function used,
they follow the expression :math:`2 \times log L`.

When the expected number of background events, ``mu_bkg`` is known, the statistic function
is ``Cash`` (see :ref:`cash`). When the number of background events is unknown, one has to
use a background estimate ``n_bkg`` taken from an OFF measurement where only background events
are expected. In this case, the statistic function is ``WStat`` (see :ref:`wstat`).

These statistic functions are at the heart of the model fitting approach in gammapy. They are
used to estimate the best fit values of model parameters and their associated confidence intervals.

They are used also to estimate the excess counts significance, i.e. the probability that
a given number of measured events ``n_on`` actually contains some signal events :math:`n_{excess}`,
as well as the errors associated to this estimated number of signal counts.

Estimating Delta TS
^^^^^^^^^^^^^^^^^^^

A classical approach to modeling and fitting relies on hypothesis testing. One wants to estimate whether
an hypothesis :math:`H_1` is statistically preferred over the reference, or null-hypothesis, :math:`H_0`.

The maximum log-likelihood ratio test provides a way to estimate the p-value of the data following :math:`H_1`
rather than :math:`H_0`, when the two hypotheses are nested.
We note this ratio :math:`\lambda = \frac{max L(X|{H_1})}{max L(X|H_0)}`

The Wilks theorem shows that under some hypothesis, :math:`-2 \log \lambda` assymptotically follows a :math:`\chi^2`
distribution with :math:`n_{dof}` degrees of freedom, where :math:`n_{dof}` is the difference of free parameters
between :math:`H_1` and :math:`H_0`.

With the definition the fit statistics :math:`-2 \log \lambda` is simply the difference of the fit statistic values for
the two hypotheses, the delta TS (for test statistic).

Counts Statistics
-----------------

To estimate the excess counts significance and errors, gammapy uses two classes for Poisson counts with
and without known background: `~gammapy.stats.CashCountsStatistic` and `~gammapy.stats.WStatCountsStatistic`

We show below how to use them.

Excess Significance
^^^^^^^^^^^^^^^^^^^

To measure the significance of an excess, one can directly use the delta TS of the measurement with and
without the excess. Taking the square root of the result yields the so-called Li & Ma significance
[LiMa1983]_ (see equation 17).

As an example, assume you measured :math:`n_{on} = 18` counts in a region where
you suspect a source might be present and :math:`n_{off} = 97` counts in a
background control region where you assume no source is present and that is
:math:`a_{off}/a_{on}=10` times larger than the on-region.


Here's how you compute the statistical significance of your detection:

.. code-block:: python

    >>> from gammapy.stats import WStatCountsStatistic
    >>> stat = WStatCountsStatistic(n_on=18, n_off=97, alpha=1. / 10)
    >>> stat.excess
    8.299999999999999
    >>> stat.significance
    2.2421704424844875

.. plot:: stats/plot_wstat_significance.py

Conversely, if you know that the expected number of background events is 9.5, you can use
the Cash statistic and obtain the Li & Ma significance for known background:

.. code-block:: python

    >>> from gammapy.stats import CashCountsStatistic
    >>> stat = CashCountsStatistic(n_on=18, mu_bkg=9.5)
    >>> stat.excess
    8.5
    >>> stat.significance
    2.4508934155585176

.. plot:: stats/plot_cash_significance.py

Excess errors
^^^^^^^^^^^^^

You can also compute the confidence interval for the true excess based on the statistic value:

If you are interested in 68% (1 :math:`\sigma`) confidence errors:

.. code-block:: python

    >>> from gammapy.stats import CashCountsStatistic
    >>> stat = CashCountsStatistic(n_on=18, mu_bkg=9.5)
    >>> stat.compute_errn()
    -3.91606323
    >>> stat.compute_errp()
    4.5823187389960225

.. plot:: stats/plot_cash_errors.py


Note that confidence intervals can be computed in a more robust manner following [Feldman1998]_.
See :ref:`feldman_cousins`.


Using `gammapy.stats`
=====================

.. toctree::
    :maxdepth: 1

    feldman_cousins
    fit_statistics
    wstat_derivation

Reference/API
=============

.. automodapi:: gammapy.stats
    :no-inheritance-diagram:
    :include-all-objects:
