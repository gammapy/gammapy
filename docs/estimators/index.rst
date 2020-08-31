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
flux points, flux profiles and flux light curves. All estimators
feature a common API and allow to estimate fluxes in energy bands.

The core of any estimator algorithm is hypothesis testing: a reference
model is tested against a null hypothesis. From the best fit reference
model a flux is derived and a corresponding significance from the
difference in fit statistics to the null hypothesis.

The technical implementation follows the concept of a reference
best fit model, which is then scaled in amplitude by fitting a `norm`
parameter. The fitting is done by grouping the data in spatial, time
and energy bins.

Based on this algorithm all estimators compute the same basic quantities:

================= =================================================
Quantity          Definition
================= =================================================
e_ref			  Reference energy
e_min			  Minimum energy
e_max			  Maximum energy
norm			  Norm with respect to the reference spectral model
norm_err		  Error on the norm derived from the Hessian matrix
ts				  Difference in fit statistics (`stat - null_value` )
sqrt_ts			  Square root of TS, corresponds to signficance (Wilk's theorem)
================= ==================================================

In addition the following optional quantities can be computed:

================= =================================================
Quantity          Definition
================= =================================================
norm_errp		  Positive error of the norm
norm_errn	      Negative error of the norm
norm_ul			  Upper limit of the norm
norm_scan		  Norm scan
stat_scan		  Fit statistics scan
stat			  Fit statistics value of the best fit model
null_value		  Fit statistics value of the null hypothesis. Rename?
================= ==================================================


In addition a reference spectral model is given. Using this reference
spectral model the norm values can be converted to different flux types
such as `dnde`, `flux`, `e2dnde` and `eflux`, whatever users are interested
in.

Optionally a likelihood (fit statistics) profile can be computed for
most estimators.


Getting Started
===============
An `Estimator` takes a reduced dataset and model definition as input.


.. toctree::
    :maxdepth: 1

    detect
    lightcurve


Reference/API
=============

.. automodapi:: gammapy.estimators
    :no-inheritance-diagram:
    :include-all-objects:



