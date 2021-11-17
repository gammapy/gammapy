.. include:: ../references.txt

.. _install-dependencies:

Dependencies
============
Gammapy is a Python package built on Numpy, Scipy and Astropy, as well as a few other
required dependencies. For certain functionality, optional dependencies are
used. The recommended way to install Gammapy is via a conda environment which
includes all required and optional dependencies (see :ref:`install`).

Note that when you install Gammapy with conda (or actually any alternative
distribution channel), you have a full package manager at your fingertips. You
can conda or pip install any extra Python package you like (e.g. `pip install
pyjokes <https://pyjok.es/>`__), upgrade or downgrade packages to other versions
(very rarely needed) or uninstall any package you don't like (almost never
useful, unless you run out of disk space).

Gammapy itself, and most analyses, work on Windows. However, two optional
dependencies don't support Windows yet: Sherpa (an optional fitting backend) and
healpy (needed to work with HEALPix maps, which is common for all-sky analyses).


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
make it convenient to use Gammapy (e.g. ``ipython`` or ``jupyter``), or that add
extra functionality (e.g. ``matplotlib`` to make plots, ``naima`` for physical
SED modeling).

* ipython_, jupyter_ and jupyterlab_ for interactive analysis
* matplotlib_ for plotting
* pandas_ for working with tables (not used within Gammapy)
* healpy_ for `HEALPIX`_ data handling
* iminuit_ for fitting by optimization
* Sherpa_ for modeling and fitting
* naima_ for SED modeling
* emcee_ for fitting by MCMC sampling
* corner_ for MCMC corner plots
