.. include:: references.txt

.. _install:

Installation
============

Here we provide short installation instructions for Gammapy and its dependencies.

Due to the large variety of systems, package managers and setups in us it's not
possible to give a detailed description for every option.

Using `conda`_ is a good option to get everything installed from scratch within minutes.
It works on any Linux, Mac or Windows machine and doesn't require root access.

If you get stuck, have a look at the extensive installation instructions for Astropy
at http://www.astropy.org/ or ask on the `Gammapy mailing list`_.

Install Gammapy using conda
---------------------------

To install the latest Gammapy **stable** version as well as the most common
optional dependencies for Gammapy, first install `Anaconda <http://continuum.io/downloads>`__
and then run these commands:

.. code-block:: bash

    conda config --add channels astropy --add channels sherpa
    conda install gammapy

For a super-quick (depending on your download speed, usually a few minutes),
non-interactive install of `Miniconda <http://conda.pydata.org/miniconda.html>`__
and Gammapy from scratch, you can also download and execute the
`gammapy-conda-install.sh <https://raw.githubusercontent.com/gammapy/gammapy/master/gammapy-conda-install.sh>`__
script like this:

.. code-block:: bash

    bash "$(curl -fsSL https://raw.githubusercontent.com/gammapy/gammapy/master/gammapy-conda-install.sh)"


To update to the latest version:

.. code-block:: bash

    conda update --all
    conda update gammapy

Overall ``conda`` is a great cross-platform package manager, you can quickly learn how to use
it by reading the docs `here <http://conda.pydata.org/docs/>`__.

Install Gammapy using pip
-------------------------

To install the latest Gammapy **stable** version (see `Gammapy page on PyPI`_) using `pip`_:

.. code-block:: bash

   $ pip install gammapy

To install the current Gammapy **development** version using `pip`_:

.. code-block:: bash

   $ pip install git+https://github.com/gammapy/gammapy.git#egg=gammapy

Install Gammapy manually
------------------------

To download the latest stable version of Gammapy, download it from
https://pypi.python.org/pypi/gammapy, if you have the
`wget <http://www.gnu.org/software/wget/>`_ tool available you can do this
from the command line:

.. code-block:: bash

   $ wget https://pypi.python.org/packages/source/g/gammapy/gammapy-0.2.tar.gz
   $ tar zxf gammapy-0.2.tar.gz
   $ cd gammapy-0.2

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

Install Gammapy dependencies using other package managers
---------------------------------------------------------

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

`apt-get <https://en.wikipedia.org/wiki/Advanced_Packaging_Tool>`__ is a popular package manager on Linux.

.. code-block:: bash

    sudo apt-get install TODO
    sudo pip install TODO


yum
+++

`yum <https://en.wikipedia.org/wiki/Yellowdog_Updater,_Modified>`__ is a popular package manager on Linux.

.. code-block:: bash

    sudo yum install TODO
    sudo pip install TODO


Macports
++++++++

`Macports <https://www.macports.org/>`__ is a popular package manager on Mac.

.. code-block:: bash

    sudo port install py34-astropy py34-pip py34-matplotlib
    sudo pip install TODO


Homebrew
++++++++

`Homebrew <http://brew.sh/>`_ is a popular package manager on Mac.

.. code-block:: bash

    sudo brew install TODO
    sudo pip install TODO


Check if your Gammapy installation is OK
----------------------------------------

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

Gammapy works with Python 2 and 3.

More specifically, in the Python 2 series we only support Python 2.7,
and in the Python 3 series we support version 3.3 or later.
Gammapy will not work with Python 2.6 or 3.2
(see :ref:`development-python2and3` if you care why).

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
* `imfun`_ for a trous wavelet decomposition
* `uncertainties`_ for Gaussian error propagation
* `gwcs`_ for generalised world coordinate transformations
* `astroplan`_ for observation planning and scheduling
* `iminuit`_ for fitting by optimization (doesn't work with Python 3 yet)
* `emcee`_ for fitting by MCMC sampling
* `h5py`_ for `HDF5 <http://en.wikipedia.org/wiki/Hierarchical_Data_Format>`__ data handling
* `PyYAML`_ for `YAML <http://en.wikipedia.org/wiki/YAML>`__ data handling
* `healpy`_ for `HEALPIX <http://healpix.jpl.nasa.gov/>`__ data handling

Actually at this point we welcome experimentation, so you can use cool new technologies
to implement some functionality in Gammapy if you like, e.g.

* `Numba <http://numba.pydata.org/>`__
* `Bokeh <http://bokeh.pydata.org/en/latest/>`__
* `Blaze <http://blaze.pydata.org/en/latest/>`__


How to make Astropy / Gammapy work with the CIAO Sherpa Python?
---------------------------------------------------------------

Note: CIAO 4.7 (released 16 December 2014)
includes Python 2.7.6, Numpy 1.8.1, IPython 2.0.0 and no PyFITS, Scipy or Matplotlib.

Note: It looks like Sherpa installation is improving ... e.g. it's now available
as a binary install via Conda:
http://cxc.harvard.edu/sherpa/contrib.html#pysherpa

Some parts of Gammapy use the `Sherpa`_ Python modeling / fitting package
from the `CIAO`_ Chandra satellite X-ray data analysis package. 

Building Sherpa and all the required libraries from source is very difficult.
You should install the binary version of CIAO as described
`here <http://cxc.cfa.harvard.edu/ciao/download>`__,
make sure you include Sherpa and exclude the Chandra CALDB.
But then the Sherpa Python and Numpy will not work with the existing
Python, Numpy, Astropy, Gammapy, ... on your system.

You have to re-install Astropy, Gammapy and any other Python packages
that you want to use in the same script as Sherpa into the CIAO Python.
Sometimes this just works, but sometimes you'll run into compilation errors
when e.g. the C extensions in ``astropy.wcs`` or ``astropy.io.fits`` are compiled.

Here's a few tricks that might help you make it work.

* Execute the  
  `ciao-python-fix <http://cxc.cfa.harvard.edu/ciao/threads/ciao_install/index.html#ciao_python_fix>`__
  script after installing CIAO:

.. code-block:: bash

   $ cd $CIAO_DIR
   $ bash bin/ciao-python-fix

* Set ``LDFLAGS`` and use ``ciaorun`` before installing a Python package with C extensions:

.. code-block:: bash

   $ export LDFLAGS="-L${ASCDS_INSTALL}/ots/lib" 
   $ ciaorun python setup.py install

* Add these folders to your ``PATH`` and ``PYTHONPATH`` so that the right command line tools or
  Python packages are picked up instead of the ones in other system folders:

.. code-block:: bash

   $ export PATH=$CIAO_DIR/ots/bin:$PATH
   $ export PYTHONPATH=$CIAO_DIR/ots/lib/python2.7/site-packages/:$PYTHONPATH

How to make Astropy / Gammapy work with the Fermi ScienceTools Python?
----------------------------------------------------------------------

Note: ``ScienceTools-v9r33p0-fssc-20140520`` (v9r33p0, released Jun 03, 2014)
includes Python 2.7.2, Numpy 1.6.1, Scipy 0.10.1, Matplotlib 1.1.1, PyFITS 3.1.2 and no IPython.

Try installing `pip`_ into that Python and then:

.. code-block:: bash

   $ pip install astropy
   $ pip install gammapy

If this doesn't work (which is not uncommon, this is known to fail to compile the C extensions of Astropy
on some platforms), ask your Python-installation-savvy co-worker or on the Astropy or Gammapy mailing list.
