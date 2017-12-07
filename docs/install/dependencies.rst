.. include:: ../references.txt

.. _install-dependencies:

Dependencies
============

.. note:: The philosophy of Gammapy is to build on the existing scientific Python stack.
   This means that you need to install those dependencies to use Gammapy.

   We are aware that too many dependencies is an issue for deployment and maintenance.
   That's why currently Gammapy only has two core dependencies --- Numpy and Astropy.
   We are considering making Sherpa, Scipy, scikit-image, photutils, reproject and naima
   core dependencies.

   In addition there are about a dozen optional dependencies that are OK to import
   from Gammapy because they are potentially useful (not all of those are
   actually currently imported).

   Before the Gammapy 1.0 release we will re-evaluate and clarify the Gammapy dependencies.

The required core dependencies of Gammapy are:

* `Numpy`_ - the fundamental package for scientific computing with Python
* `Astropy`_ - the core package for Astronomy in Python
* `regions`_ - Astropy regions package. Planned for inclusion in Astropy core as `astropy.regions`.
* `click`_ for making command line tools

We're currently using

* `PyYAML`_ for `YAML <http://en.wikipedia.org/wiki/YAML>`__ data handling (config and results files)

Currently optional dependencies that are being considered as core dependencies:

* `Sherpa`_ for modeling / fitting
* `scipy library`_ for numerical methods
* `scikit-image`_ for some image processing tasks
* `photutils`_ for image photometry
* `reproject`_ for image reprojection
* `naima`_ for SED modeling

Allowed optional dependencies:

* `matplotlib`_ for plotting
* `aplpy`_ for sky image plotting (provides a high-level API)
* `pandas`_ CSV read / write; DataFrame
* `scikit-learn`_ for some data analysis tasks
* `GammaLib`_ and `ctools`_ for simulating data and likelihood fitting
* `ROOT`_ and `rootpy`_ conversion helper functions (still has some Python 3 issues)
* `uncertainties`_ for linear error propagation
* `astroplan`_ for observation planning and scheduling
* `iminuit`_ for fitting by optimization
* `emcee`_ for fitting by MCMC sampling
* `h5py`_ for `HDF5 <http://en.wikipedia.org/wiki/Hierarchical_Data_Format>`__ data handling
* `healpy`_ for `HEALPIX <http://healpix.jpl.nasa.gov/>`__ data handling
* `nbsphinx`_ for transformation of Jupyter notebooks into fixed-text documentation 
