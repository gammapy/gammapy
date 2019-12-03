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
used. The recommended way to install Gammapy is via a conda environment which
includes all required and optional dependencies (see :ref:`install`).

Note that when you install Gammapy with conda (or actually any alternative
distribution channel), you have a full package manager at your fingertips. You
can conda or pip install any extra Python package you like (e.g. `pip install
pyjokes <https://pyjok.es/>`__), upgrade or downgrade packages to other versions
(very rarely needed) or uninstall any package you don't like (almost never
useful, unless you run out of disk space).

Required dependencies
---------------------

Required dependencies are automatically installed when using e.g. ``conda
install gammapy -c conda-forge`` or ``pip install gammapy``.

* numpy_ - array and math functions
* scipy_ - numerical methods (interpolation, integration, convolution)
* Astropy_ - core package for Astronomy in Python
* regions_ - Astropy sky region package
* click_ - used for the ``gammapy`` command line tool
* PyYAML_ - support for YAML_ format (config and results files)
* pydantic_ - support config file validation

Optional dependencies
---------------------

The optional dependencies listed here are the packages listed in the conda
environment specification (see :ref:`install`). This is a mix of packages that
make it convenient to use Gammapy (e.g. ``ipython`` or ``jupyter``), that add
extra functionality (e.g. ``matplotlib`` to make plots, ``naima`` for physical
SED modeling), and partly packages that aren't used within Gammapy, only for
example data download (``parfive``) or in one of the tutorials (``sherpa``).

* ipython_, jupyter_ and jupyterlab_ for interactive analysis
* matplotlib_ for plotting
* pandas_ for working with tables (not used within Gammapy)
* healpy_ for `HEALPIX`_ data handling
* iminuit_ for fitting by optimization
* Sherpa_ for modeling and fitting
* naima_ for SED modeling
* emcee_ for fitting by MCMC sampling
* corner_ for MCMC corner plots
* parfive_ for example data and tutorial notebook download

Versions
--------

Every stable version of Gammapy is compatible with a range of versions of it's
dependencies. E.g. Gammapy v0.13 was fully tested and is known to be compatible
with Python 3.5, 3.6 and 3.7. Most likely it will be compatible with Python 3.8
when it comes out, or also with Python 4.0 if that ever comes out, but we can't
know for sure or guarantee that. With other dependencies it's similar, e.g.
Gammapy v0.13 was fully tested and known to work with Astropy 2.0 to 3.2, there
were certain functions in Gammapy that didn't work with older Astropy versions
than 2.0, and for Astropy 4.0 or newer versions likely everything will work, but
we can't test or guarantee it before that comes out.

So for now, we have decided to ship and recommend the use of Gammapy conda
environments with fixed and recent versions of all dependencies. This will give
users a well-tested and known good reproducible execution environment (on any of
the supported platforms: Linux, macOS and Windows). As an example, see
`gammapy-0.13-environment.yml`_.
