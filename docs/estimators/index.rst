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
feature a common API and allow to estimate fluxes in bands of reconstructed
energy.

The core of any estimator algorithm is hypothesis testing: a reference
model or counts excess is tested against a null hypothesis. From the
best fit reference model a flux is derived and a corresponding :math:`\sqrt{\Delta TS}`
value from the difference in fit statistics to the null hypothesis,
assuming one degree of freedom (:ref:`Estimating Delta TS`). In this case
:math:`\sqrt{\Delta TS}` represents an approximation of the
"classical significance".

In general the flux can be estimated using methods:

1. Based on model fitting: given a (global) best fit model with multiple model components,
the flux of the component of interest is re-fitted in the chosen energy, time or spatial
region. The new flux is given as a ``norm`` with respect to the global reference model.
Optionally other component parameters in the global model can be re-optimised.

2. Based on excess: in the case of having one energy bin, neglecting the PSF and not re-optimising
other parameters, once can estimate the flux based on excess and derive the significance
analytically from the classical Li & Ma solution.


The technical implementation follows the concept of a reference
best fit model. Given a global best fit model, the source of interest
(for which flux points are computed) is scaled in amplitude by fitting a ``norm``
parameter. The fitting is done by grouping the data in time
and reconstructed energy bins (reference?).

Based on this algorithm most estimators compute the same basic quantities:

================= =================================================
Quantity          Definition
================= =================================================
e_ref			  Reference energy
e_min			  Minimum energy
e_max			  Maximum energy
norm			  Norm with respect to the reference spectral model
norm_err		  Symmetric rrror on the norm derived from the Hessian matrix
ts				  Difference in fit statistics (`stat_sum - null_value` )
sqrt_ts			  Square root of TS, corresponds to significance (Wilk's theorem)
================= =================================================

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
null_value		  Fit statistics value of the null hypothesis
================= =================================================


To compute the assymetric errors as well as upper limits one can
specify the arguments ``n_sigma`` and ``n_sigma_ul``. The ``n_sigma``
arguments are translated into a TS value assuming ``ts = sigma ** 2``.

In addition to the norm values a reference spectral model is given.
Using this reference spectral model the norm values can be converted
to the following different SED types:

================= =================================================
Quantity          Definition
================= =================================================
dnde 		      Differential flux at ``e_ref``
flux 			  Integrated flux between ``e_min`` and ``e_max``
eflux			  Integrated energy flux between ``e_min`` and ``e_max``
================= =================================================

The same can be applied for the error and upper limit information.
More information can be found on the `likelihood SED type page`_.


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


.. _`likelihood SED type page`: https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/binned_likelihoods/index.html


