.. include:: ../references.txt

.. _install:

************
Installation
************

* Gammapy works with legacy Python (version 2.7) as well as Python 3 (version 3.4 or above).
* The core dependencies are Numpy and Astropy, as well as for now the regions package
  (until it is merged into Astropy core).
* The main optional dependencies are PyYAML, Scipy, healpy, uncertainties and Sherpa
  (only imported when used).
* Linux and Mac OS are fully supported.
* Windows is fully supported by Gammapy, but not by Sherpa (which is used as a backend for
  modeling and fitting in Gammapy), so on Windows only part of the Gammapy functionality is available.
* You can always check what the latest stable release of Gammapy is here: https://pypi.python.org/pypi/gammapy

Quick install guide
===================

Stable version
--------------

You can install the latest stable version of Gammapy with conda::

  conda install -c openastronomy gammapy

or with pip::

  pip install gammapy

or with Macports (a package manager for Mac OS)::

  sudo port install gammapy


Gammapy is not yet available in the Linux distributions, i.e. at this time you can't
install it with e.g. ``apt-get`` or ``yum``.

Development version
-------------------

To install the development version of Gammapy::

  git clone https://github.com/gammapy/gammapy.git
  cd gammapy
  pip install .

of if you're using conda, you can install the development version like this::

  git clone https://github.com/gammapy/gammapy.git
  cd gammapy
  conda install -f environment.yml
  source activate gammapy-dev

Verify
------

To verify that Gammapy is installed and available, type this::

    $ python

    >>> import gammapy
    >>> print(gammapy.__version__)

To find out

Need help?
==========

If you're not sure how to best install Gammapy on your machine (e.g. whether to use
conda or pip or Macports ...), we recommend that you give conda a try first.
It's a binary package manager (so generally installation is fast), and allows you to
install any software in your home folder (without needing ``sudo``) and works the
same on Linux, Mac OS and Windows. Many Gammapy users and developers use conda
and are happy with it.

If you have any questions or issues, don't hesitate to ask on the `Gammapy mailing list`_!

Detailed information
====================

Once you've made your choice how to install Gammapy, you can find detailed information
on the following sub-pages:

.. toctree::
  :maxdepth: 1

  conda
  pip
  macports
  other
  check
  dependencies


If you'd like to make a code contribution to Gammapy, please see the developer documentation
for information how to get set up (e.g. to code, run tests, generate docs locally).
