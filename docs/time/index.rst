.. _time:

********************
time - Time analysis
********************

.. currentmodule:: gammapy.time

Introduction
============

`gammapy.time` contains classes and methods for time-based analysis, e.g. for AGN, binaries
or pulsars studies. The main classes are `~gammapy.time.LightCurve`, which is a container for
light curves, and `~gammapy.time.LightCurveEstimator`, which extracts a light curve from a list
of datasets. A number of functions to test for variability and periodicity are available in
`~gammapy.time.variability` and `~gammapy.time.periodicity`. Finally, `gammapy.utils.time`
contains low-level helper functions for time conversions.





Variability and periodicity tests
=================================

A few utility functions to perform timing tests are available in `~gammapy.time`.

`~gammapy.time.compute_chisq` performs a chisquare test for variable source flux::

     >>> from gammapy.time import chisquare
     >>> print(compute_chisq(lc['FLUX']))

`~gammapy.time.compute_fvar` calculates the fractional variance excess::

     >>> from gammapy.time import fvar
     >>> print(compute_fvar(lc['FLUX'], lc['FLUX_ERR']))

`~gammapy.time` also provides methods for period detection in time series, i.e. light
curves of :math:`\gamma`-ray sources.  `~gammapy.time.robust_periodogram` performs a
periodogram analysis where the unevenly sampled time series is contaminated by outliers,
i.e. due to the source's high states. This is demonstrated on the :ref:`period` page.

Tutorials
=========

The main tutorial demonstrates how to extract light curves from 1D and 3D datasets:

* `Light Curve tutorial <../notebooks/light_curve.html>`__

Light curve extraction on small time bins (i.e. smaller than the observation scale) for flares
is demonstrated in the following tutorial:

* `Flare tutorial <../notebooks/light_curve_flare.html>`__

Using `gammapy.time`
====================

.. toctree::
   :maxdepth: 1

   period

Reference/API
=============

.. automodapi:: gammapy.time
    :no-inheritance-diagram:
    :include-all-objects:
