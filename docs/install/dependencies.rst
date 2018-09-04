.. include:: ../references.txt

.. _install-dependencies:

Dependencies
============

The philosophy of Gammapy is to build on the existing scientific Python stack.
This means that you need to install those dependencies to use Gammapy.

We are aware that too many dependencies is an issue for deployment and
maintenance. That's why currently Gammapy only has two core dependencies ---
Numpy and Astropy. We are considering making Scipy and reproject and PyYAML core dependencies.

In addition there are about a dozen optional dependencies that are OK to import
from Gammapy because they are potentially useful (not all of those are actually
currently imported).

Before the Gammapy 1.0 release we will re-evaluate and clarify the Gammapy
dependencies.

The required core dependencies of Gammapy are:

* `Numpy`_ - the fundamental package for scientific computing with Python
* `Astropy`_ - the core package for Astronomy in Python
* `regions`_ - Astropy regions package. Planned for inclusion in Astropy core as ``astropy.regions``.
* `click`_ for making command line tools

Optional dependencies of Gammapy:

* `PyYAML`_ for `YAML <http://en.wikipedia.org/wiki/YAML>`__ data handling (config and results files)
* `scipy library`_ for numerical methods
* `reproject`_ for image reprojection
* `iminuit`_ for fitting by optimization
* `uncertainties`_ for linear error propagation
* `matplotlib`_ for plotting
* `emcee`_ for fitting by MCMC sampling
* `healpy`_ for `HEALPIX`_ data handling
* `naima`_ for SED modeling
