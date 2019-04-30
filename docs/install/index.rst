.. include:: ../references.txt

.. note::

    The recommended way to install Gammapy is to use conda and to follow
    the simple instructions in :ref:`getting-started`.

    This section contains some additional information concerning the conda
    installation as well as information on alternative ways to install Gammapy and
    it's dependencies like Numpy and Astropy, e.g. using pip or Macports or your
    Linux package manager.

.. _install:

************
Installation
************

* Gammapy works with Python 3.5 or later.
  The last release to support Python 2.7 was Gammapy v0.10 in January 2019.
* The core dependencies are Numpy and Astropy, as well as for now the regions
  package (until it is merged into Astropy core).
* The main optional dependencies are PyYAML, Scipy, healpy, uncertainties and
  Sherpa (only imported when used).
* Linux and Mac OS are fully supported.
* Windows is fully supported by Gammapy, but not by Sherpa (which is used as a
  backend for modeling and fitting in Gammapy), so on Windows only part of the
  Gammapy functionality is available.
* You can always check what the latest stable release of Gammapy is here:
  https://pypi.org/project/gammapy

Quick install guide
===================

Stable version
--------------

You can install the latest stable version of Gammapy with conda::

    conda install -c conda-forge gammapy

or with pip::

    python -m pip install gammapy

or with Macports (a package manager for Mac OS)::

    sudo port install gammapy

Gammapy is not yet available in the Linux distributions, i.e. at this time you
can't install it with e.g. ``apt-get`` or ``yum``.

Development version
-------------------

You can install the development version of Gammapy like this::

    python -m pip install --user git+https://github.com/gammapy/gammapy.git

This will ``git clone`` the Gammapy repository from Github into a temp folder
and then build and install Gammapy from there.

If there are any errors related to Cython, Numpy or Astropy, you should install
those first and try again::

    conda install -c cython numpy astropy click regions
    python -m pip install --user git+https://github.com/gammapy/gammapy.git

How to get set up for Gammapy development is described here: :ref:`dev_setup`

Verify
------

To verify that Gammapy is installed and available, and to check it's version and
where it's located, type this::

    $ python

    >>> import gammapy
    >>> print(gammapy.__version__)
    >>> print(gammapy.__path__)

Need help?
==========

If you're not sure how to best install Gammapy on your machine (e.g. whether to
use conda or pip or Macports ...), we recommend that you give conda a try first.
It's a binary package manager (so generally installation is fast), and allows
you to install any software in your home folder (without needing ``sudo``) and
works the same on Linux, Mac OS and Windows. Many Gammapy users and developers
use conda and are happy with it.

There is a nice blog post `Installing Python Packages from a Jupyter Notebook`_
that explains how to install Python packages in general with ``pip`` and
``conda``, specifically from inisde Jupyter notebooks, but also in general it's
a good introduction how Python package installation with ``pip`` and ``conda``
work.

If you have any questions or issues, don't hesitate to ask on the `Gammapy
mailing list`_!

Detailed information
====================

Once you've made your choice how to install Gammapy, you can find detailed
information on the following sub-pages:

.. toctree::
    :maxdepth: 1

    conda
    pip
    macports
    other
    check
    dependencies

If you'd like to make a code contribution to Gammapy, please see the developer
documentation for information how to get set up (e.g. to code, run tests,
generate docs locally).
