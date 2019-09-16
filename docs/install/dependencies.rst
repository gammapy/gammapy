.. include:: ../references.txt

.. _install-dependencies:

Gammapy Dependencies
====================

The latest stable version of Gammapy is listed at https://gammapy.org

Gammapy works with Python 3.6 or later.

Linux and Mac OS are fully supported.

Gammapy itself, and most analyses, work on Windows. However, two optional
dependencies don't support Windows yet: Sherpa (an optional fitting backend) and
healpy (needed to work with HEALPix maps, which is common for all-sky analyses).

Gammapy is a Python package built on Numpy and Astropy, as well as a few other
required dependencies. For certain functionality, optional dependencies are
used. The recommended way to install Gammapy (see :ref:`install`) is via a conda
environment which includes all required and optional dependencies.

Below is a complete list of dependencies, with links to the documentation pages
for each project with further information.

Required dependencies
---------------------

The required core dependencies of Gammapy are:

* `Numpy`_ - the fundamental package for scientific computing with Python
* `scipy library`_ for numerical methods
* `Astropy`_ - the core package for Astronomy in Python
* `regions`_ - Astropy regions package. Planned for inclusion in Astropy core as ``astropy.regions``.
* `click`_ for making command line tools
* `PyYAML`_ for `YAML <http://en.wikipedia.org/wiki/YAML>`__ data handling (config and results files)

Optional dependencies
---------------------

Optional dependencies of Gammapy:

* `reproject`_ for image reprojection
* `iminuit`_ for fitting by optimization
* `uncertainties`_ for linear error propagation
* `matplotlib`_ for plotting
* `emcee`_ for fitting by MCMC sampling
* `healpy`_ for `HEALPIX`_ data handling
* `naima`_ for SED modeling
* `Sherpa`_ for modelling and fitting
