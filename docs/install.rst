.. include:: references.txt

.. _install:

Installation
============

Here we provide short installation instructions for Gammapy and its dependencies.

Gammapy works (and is continually tested) with Python 2 and 3 on Linux, Mac OS X and Windows.

More specifically, in the Python 2 series we only support (i.e. test in continuous integration)
Python 2.7, and in the Python 3 series we support version 3.4 or later.
Gammapy will (probably) not work with older Python versions,
such as 2.6 or 3.2 (see :ref:`development-python2and3` if you care why).

Due to the large variety of systems, package managers and setups in us it's not
possible to give a detailed description for every option.

Using `conda`_ is a good option to get everything installed from scratch within minutes.
It works on any Linux, Mac or Windows machine and doesn't require root access.

If you get stuck, have a look at the extensive installation instructions for Astropy
at http://www.astropy.org/ or ask on the `Gammapy mailing list`_.

The main way to improve the instructions is via your feedback!

Conda
-----

To install the latest Gammapy **stable** version as well as the most common
optional dependencies for Gammapy, first install `Anaconda <http://continuum.io/downloads>`__
and then run these commands:

.. code-block:: bash

    conda config --add channels astropy --add channels sherpa
    conda install gammapy naima \
        scipy matplotlib ipython-notebook \
        cython click

We strongly recommend that you install the optional dependencies of Gammapy to have the full
functionality available:

.. code-block:: bash

    conda install \
        scikit-image scikit-learn h5py pandas \
        aplpy wcsaxes photutils reproject

    pip install iminuit

Sherpa is the only Gammapy dependency that's not yet available on Python 3, so if you want
to use Sherpa for modeling / fitting, install Anaconda Python 2 and

.. code-block:: bash

    conda install sherpa

For a quick (depending on your download and disk speed, usually a few minutes),
non-interactive install of `Miniconda <http://conda.pydata.org/miniconda.html>`__
and Gammapy from scratch, use the commands from this script:
`gammapy-conda-install.sh <https://raw.githubusercontent.com/gammapy/gammapy/master/gammapy-conda-install.sh>`__.

Executing it like this should also work:

.. code-block:: bash

    bash "$(curl -fsSL https://raw.githubusercontent.com/gammapy/gammapy/master/gammapy-conda-install.sh)"

To update to the latest version:

.. code-block:: bash

    conda update --all
    conda update gammapy

Overall ``conda`` is a great cross-platform package manager, you can quickly learn how to use
it by reading the docs `here <http://conda.pydata.org/docs/>`__.

pip
---

To install the latest Gammapy **stable** version (see `Gammapy page on PyPI`_) using `pip`_:

.. code-block:: bash

   $ pip install gammapy

To install the current Gammapy **development** version using `pip`_:

.. code-block:: bash

   $ pip install git+https://github.com/gammapy/gammapy.git#egg=gammapy

setup.py
--------

To download the latest stable version of Gammapy, download it from
https://pypi.python.org/pypi/gammapy, if you have the
`wget <http://www.gnu.org/software/wget/>`_ tool available you can do this
from the command line:

.. code-block:: bash

   $ wget https://pypi.python.org/packages/source/g/gammapy/gammapy-0.3.tar.gz
   $ tar zxf gammapy-0.3.tar.gz
   $ cd gammapy-0.3

To download the latest development version of Gammapy:

.. code-block:: bash

   $ git clone https://github.com/gammapy/gammapy.git
   $ cd gammapy

Either way, you now can install, test or build the documentation:

.. code-block:: bash

   $ python setup.py install
   $ python setup.py test
   $ python setup.py build_sphinx

Also you have easy access to the Python scripts from the tutorials and examples: 

.. code-block:: bash

   $ cd docs/tutorials
   $ cd examples

If you want to contribute to Gammapy, but are not familiar with Python or
git or Astropy yet, please have a look at the  
`Astropy developer documentation <http://docs.astropy.org/en/latest/#developer-documentation>`__.

Other package managers
----------------------

Besides conda, Gammapy and some of the optional dependencies (Sherpa, Astropy-affiliated packages)
as not yet available in other package managers (such as
e.g. `apt-get <https://en.wikipedia.org/wiki/Advanced_Packaging_Tool>`__
or `yum <https://en.wikipedia.org/wiki/Yellowdog_Updater,_Modified>`__ on Linux
or `Macports <https://www.macports.org/>`__
or `homebrew <http://brew.sh/>`__ on Mac.

So installing Gammapy this way is not recommended at this time.
(The recommended method is conda as mentioned above).

Still, it's possible and common on systems where users have root access
to install some of the dependencies using those package managers, and then
to use `pip`_ to do the rest of the installation.

So as a convenience, here we show the commands to install those packages that are available,
so that you don't have to look up the package names.

We do hope this situation will improve in the future as more astronomy packages become
available in those distributions and versions are updated.

apt-get
+++++++

`apt-get <https://en.wikipedia.org/wiki/Advanced_Packaging_Tool>`__ is a popular package manager on Linux,
e.g. on Debian or Ubuntu.

The following packages are available:

.. code-block:: bash

    sudo apt-get install \
        python3-pip python3-scipy python3-matplotlib python3-skimage python3-sklearn \
        python3-pandas python3-h5py python3-yaml ipython3-notebook python3-uncertainties \
        python3-astropy python3-click

The following packages have to be installed with pip:

.. code-block:: bash

    pip3 install --user \
        gammapy naima photutils reproject wcsaxes gwcs astroplan \
        iminuit emcee healpy

Sherpa currently doesn't work on Python 3.
You could try to use Python 2 and pip-installing Sherpa (don't know if that works).

A Debian package for Sherpa is in preparation: https://github.com/sherpa/sherpa/issues/75

A Debian package for Gammapy is in preparation: https://github.com/gammapy/gammapy/issues/324

As far as I can see there's no HEALPIX or healpy package.

yum
+++

`yum <https://en.wikipedia.org/wiki/Yellowdog_Updater,_Modified>`__ is a popular package manager on Linux,
e.g. on Scientific linux or Red Hat Linux.

If you are a ``yum`` user, please contribute the equivalent commands
(see e.g. the Macports section below).

Macports
++++++++

`Macports <https://www.macports.org/>`__ is a popular package manager on Mac.

The following packages are available via Macports:

.. code-block:: bash

    export PY=py35
    sudo port install \
        $PY-pip $PY-scipy $PY-matplotlib $PY-scikit-image $PY-scikit-learn \
        $PY-pandas $PY-emcee $PY-h5py $PY-yaml $PY-ipython $PY-uncertainties \
        $PY-healpy $PY-astropy $PY-click $PY-cython


If you want some other Python version, use a different suffix (e.g. ``py27`` or ``py35``).
Having multiple Python versions simultaneously works well, but is only really useful for developers.

.. code-block:: bash

    pip install --user \
        gammapy naima photutils reproject wcsaxes gwcs astroplan \
        iminuit


Homebrew
++++++++

`Homebrew <http://brew.sh/>`_ is a popular package manager on Mac.

If you are a ``brew`` user, please contribute the equivalent commands
(see e.g. the Macports section above).


Check Gammapy installation
--------------------------

To check if Gammapy is correctly installed, start up python or ipython,
import Gammapy and run the unit tests:

.. code-block:: bash

   $ python -c 'import gammapy; gammapy.test()'

To check if the Gammapy command line tools are on your ``$PATH`` try this:

.. code-block:: bash

   $ gammapy-info --tools

To check which dependencies of Gammapy you have installed:

.. code-block:: bash

   $ gammapy-info --dependencies

.. _install-issues:

Common issues
-------------

If you have an issue with Gammapy installation or usage, please check
this list. If your issue is not adressed, please send an email to the
mailing list.

- Q: I get an error mentioning something (e.g. Astropy) isn't available,
  but I did install it.

  A: Check that you're using the right ``python`` and that your
  ``PYTHONPATH`` isn't pointing to places that aren't appropriate
  for this Python (usually it's best to not set it at all)
  using these commands:

  .. code-block:: bash

      which python
      echo $PYTHONPATH
      python -c 'import astropy'

.. _install-dependencies:

Dependencies
------------

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

We're currently using

* `click`_ for making command line tools
* `PyYAML`_ for `YAML <http://en.wikipedia.org/wiki/YAML>`__ data handling (config and results files
* `flask`_ and some Flask plugins for Gammapy web apps


Currently optional dependencies that are being considered as core dependencies:

* `Sherpa`_ for modeling / fitting (doesn't work with Python 3 yet)
* `scipy library`_ for numerical methods
* `scikit-image`_ for some image processing tasks
* `photutils`_ for image photometry
* `reproject`_ for image reprojection
* `naima`_ for SED modeling

Allowed optional dependencies:

* `matplotlib`_ for plotting
* `wcsaxes`_ for sky image plotting (provides a low-level API)
* `aplpy`_ for sky image plotting (provides a high-level API)
* `pandas`_ CSV read / write; DataFrame
* `scikit-learn`_ for some data analysis tasks
* `GammaLib`_ and `ctools`_ for simulating data and likelihood fitting
* `ROOT`_ and `rootpy`_ conversion helper functions (still has some Python 3 issues)
* `uncertainties`_ for linear error propagation
* `gwcs`_ for generalised world coordinate transformations
* `astroplan`_ for observation planning and scheduling
* `iminuit`_ for fitting by optimization
* `emcee`_ for fitting by MCMC sampling
* `h5py`_ for `HDF5 <http://en.wikipedia.org/wiki/Hierarchical_Data_Format>`__ data handling
* `healpy`_ for `HEALPIX <http://healpix.jpl.nasa.gov/>`__ data handling


Fermi ScienceTools
------------------

The `Fermi ScienceTools <http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`__
ships with it's own Python 2.7 interpreter.

If you want to use Astropy or Gammapy with that Python, you have to install it using
that Python interpreter, other existing Python interpreters or installed packages
can't be used (when they have C extensions, like Astropy does).

Fermi ScienceTools version ``v10r0p5`` (released Jun 24, 2015) includes
Python 2.7.8, Numpy 1.9.1, Scipy 0.14.0, matplotlib 1.1.1, PyFITS 3.1.2.
Unfortunately pip, ipython or Astropy are not included.

So first in stall `pip`_ (see
`instructions <https://pip.pypa.io/en/latest/installing.html#install-pip>`__),
and then

.. code-block:: bash

   $ pip install ipython astropy gammapy

If this doesn't work (which is not uncommon, this is known to fail to compile the C
extensions of Astropy on some platforms), ask your Python-installation-savvy co-worker
or on the Astropy or Gammapy mailing list.
