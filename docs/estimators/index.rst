.. include:: ../references.txt

.. _estimators:

**********************************
estimators - High level estimators
**********************************

.. currentmodule:: gammapy.estimators

Introduction
============
The `gammapy.estimators` submodule contains algorithms and classes
for high level flux and significance estimation such as flux maps,
flux points, flux profiles and flux light curves.


`FluxPointsEstimator`
`FluxMapEstimator`
`FluxProfileEstimator`
`LightCurveEstimator` `FluxCurveEstimator`





`FluxPoints`
`FluxProfile`
`LightCurve`
`FluxMap`


All estimators can compute the same quantities:

================= ====================================
Quantity          Definition
================= ====================================
e_ref
e_min
e_max
ref_dnde
norm
stat

norm_err
norm_errp
norm_errn
norm_ul
sqrt_ts
ts
null_value
norm_scan
stat_scan
counts
excess
npred
================= ====================================


Getting Started
===============
An `Estimator` takes a reduced dataset as input.


.. toctree::
    :maxdepth: 1

    detect
    lightcurve


Reference/API
=============

.. automodapi:: gammapy.estimators
    :no-inheritance-diagram:
    :include-all-objects:



