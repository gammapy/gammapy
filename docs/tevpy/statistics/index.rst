********************************
Statistics tools (`tevpy.stats`)
********************************

Introduction
============

The `~tevpy.stats` package holds statistical estimators,
fit statistics and algorithms commonly used in gamma-ray astronomy.

It is mostly concerned with the evaluation of one or several observations
that count events in a given region and time window, i.e. with
Poisson-distributed counts measurements.

As an example, assume you have measured :math:`n_{on} = 18` counts in a
region where you suspect a source might be present and :math:`n_{off} = 97`
counts in a background control region where you assume no source is present
and that is :math:`a_{off}/a_{on}=10` times larger than the on-region.

Here's how you compute the statistical significance of your detection 
with the Li \& Ma formula:

   >>> from tevpy.stats import significance_on_off
   >>> significance_on_off(n_on=18, n_off=97, alpha=1. / 10, method='lima')
   2.2421704424844875

Note that throughout this package the parameter `alpha = a_on / a_off`
is used and not the `area_factor = a_off / a_on`.

TODO: Give references to Li & Ma, Rolke and Feldman-Cousins.


Reference/API
=============

.. automodapi:: tevpy.statistics.poisson
    :no-inheritance-diagram:

.. automodapi:: tevpy.statistics.fit_statistics
    :no-inheritance-diagram:

.. automodapi:: tevpy.statistics.utils
    :no-inheritance-diagram:
    