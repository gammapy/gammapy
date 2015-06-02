.. include:: references.txt

.. _install:

Installation
============

Here we provide short installation instructions for Gammapy and its dependencies
using the `pip`_ and `conda`_ tools.

If you get stuck, have a look at the extensive installation instructions for Astropy
at http://www.astropy.org/ or ask on the `Gammapy mailing list`_.

Install Gammapy using pip
-------------------------

To install the latest Gammapy **stable** version using `pip`_:

.. code-block:: bash

   $ pip install gammapy

To install the current Gammapy **development** version using `pip`_:

.. code-block:: bash

   $ pip install git+https://github.com/gammapy/gammapy.git#egg=gammapy

Download and install Gammapy manually
-------------------------------------

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

Install Gammapy and its dependencies using Conda
------------------------------------------------

In the past it has often been hard to install Python packages and all their dependencies.
Not any more ... using `conda`_ you can install Gammapy and most of its dependencies on
any Linux machine or Mac in 5 minutes (without needing root access on the machine).

Go to http://conda.pydata.org/miniconda.html and download the installer for your system.
Or directly use `wget <https://www.gnu.org/software/wget/>`__ from the terminal:

For Linux:

.. code-block:: bash

   $ wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

For Mac:

.. code-block:: bash

   $ wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh

Then install binary packages using ``conda`` and source packages using ``pip`` by
copy & pasting the following lines into your terminal:

.. code-block:: bash

   bash miniconda.sh -b -p $PWD/miniconda
   export PATH="$PWD/miniconda/bin:$PATH"
   conda config --set always_yes yes --set changeps1 no
   conda update -q conda
   conda install pip scipy matplotlib scikit-image scikit-learn astropy h5py pandas
   pip install reproject aplpy wcsaxes naima astroplan gwcs photutils
   pip install gammapy
 
Overall ``conda`` is a great cross-platform package manager, you can quickly learn how to use
it by reading the docs here: http://conda.pydata.org/docs/


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


BSD or GPL license?
-------------------

Gammapy is BSD licensed (same license as Numpy, Scipy, Matplotlib, scikit-image, Astropy, photutils, yt, ...).

We prefer this over the GPL3 or LGPL license because it means that the packages we are most likely to
share code with have the same license, e.g. we can take a function or class and "upstream" it, i.e. contribute
it e.g. to Astropy or Scipy if it's generally useful.

Some optional dependencies of Gammapy (i.e. other packages like Sherpa or Gammalib or ROOT that we import in some
places) are GPL3 or LGPL licensed.

Now the GPL3 and LGPL license contains clauses that other package that copy or modify it must be released under
the same license.
We take the standpoint that Gammapy is independent from these libraries, because we don't copy or modify them.
This is a common standpoint, e.g. ``astropy.wcs`` is BSD licensed, but uses the LGPL-licensed WCSLib.

Note that if you distribute Gammapy together with one of the GPL dependencies,
the whole distribution then falls under the GPL license.
