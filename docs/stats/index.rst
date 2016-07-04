.. _stats:

************************************
Statistics tools (``gammapy.stats``)
************************************

.. currentmodule:: gammapy.stats

.. _stats-introduction:

Introduction
============

`gammapy.stats` holds statistical estimators,
fit statistics and algorithms commonly used in gamma-ray astronomy.

It is mostly concerned with the evaluation of one or several observations
that count events in a given region and time window, i.e. with
Poisson-distributed counts measurements.

For on-off methods we will use the following variable names
following the notation in [Cousins2007]_:

================= ====================================================
Variable          Definition
================= ====================================================
``n_on``          Total observed counts in the on region
``n_off``         Total observed counts in the off region
``mu_on``         Total expected counts in the on region
``mu_off``        Total expected counts in the off region
``mu_sig``        Signal expected counts in the on region
``mu_bkg``        Background expected counts in the on region
``a_on``          Relative background efficiency in the on region
``a_off``         Relative background efficiency in the off region
``alpha``         Background efficiency ratio ``a_on`` / ``a_off``
``n_bkg``         Background estimate in the on region
================= ====================================================

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

When the background is known and there is only an "on" region (sometimes also called "source region"),
we use the variable names ``n_on``, ``mu_on``, ``mu_sig`` and ``mu_bkg``.


These are references describing the available methods:
[LiMa1983]_, [Cash1979]_, [Stewart2009]_, [Rolke2005]_, [Feldman1998]_, [Cousins2007]_.

Getting Started
===============

Li \& Ma Significance
---------------------

As an example, assume you measured :math:`n_{on} = 18` counts in a region where
you suspect a source might be present and :math:`n_{off} = 97` counts in a
background control region where you assume no source is present and that is
:math:`a_{off}/a_{on}=10` times larger than the on-region.

Here's how you compute the statistical significance of your detection
with the Li \& Ma formula:

.. code-block:: python

   >>> from gammapy.stats import significance_on_off
   >>> significance_on_off(n_on=18, n_off=97, alpha=1. / 10, method='lima')
   2.2421704424844875

Confidence Intervals
--------------------

Assume you measured 6 counts in a Poissonian counting experiment with an
expected background :math:`b = 3`. Here's how you compute the 90% upper limit
on the signal strength :math:`\\mu`:

.. code-block:: python

   import numpy as np
   from scipy import stats
   import gammapy.stats as gstats

   x_bins = np.arange(0, 100)
   mu_bins = np.linspace(0, 50, 50 / 0.005 + 1, endpoint=True)

   matrix = [stats.poisson(mu + 3).pmf(x_bins) for mu in mu_bins]
   acceptance_intervals = gstats.fc_construct_acceptance_intervals_pdfs(matrix, 0.9)
   LowerLimitNum, UpperLimitNum, _ = gstats.fc_get_limits(mu_bins, x_bins, acceptance_intervals)
   mu_upper_limit = gstats.fc_find_limit(6, UpperLimitNum, mu_bins)

The result is ``mu_upper_limit == 8.465``.

Using `gammapy.stats`
=====================

.. toctree::
   :maxdepth: 1

   feldman_cousins
   fit_statistics

Reference/API
=============

.. automodapi:: gammapy.stats
    :no-inheritance-diagram:
