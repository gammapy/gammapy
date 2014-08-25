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

   $ wget https://pypi.python.org/packages/source/g/gammapy/gammapy-0.1.tar.gz  
   $ tar zxf gammapy-0.1.tar.gz
   $ cd gammapy-0.1

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

Go to http://conda.pydata.org/miniconda.html and download the installer for your system,
e.g. currently for Linux and Mac you can use:

.. code-block:: bash

   $ wget http://repo.continuum.io/miniconda/Miniconda3-3.6.0-Linux-x86_64.sh -O miniconda.sh
   $ wget http://repo.continuum.io/miniconda/Miniconda3-3.6.0-MacOSX-x86_64.sh -O miniconda.sh

Then install binary packages using ``conda`` and source packages using ``pip`` by
copy & pasting the following lines into your terminal:

.. code-block:: bash

   bash miniconda.sh -b -p $PWD/miniconda
   export PATH="$PWD/miniconda/bin:$PATH"
   conda config --set always_yes yes --set changeps1 no
   conda update -q conda
   conda install pip scipy matplotlib scikit-image scikit-learn astropy
   pip install reproject aplpy
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

.. _install-requirements:

Requirements
------------

Gammapy works with Python 2 and 3.

More specifically, in the Python 2 series we only support Python 2.7,
and in the Python 3 series we support version 3.3 or later.
Gammapy will not work with Python 2.6 or 3.2
(see :ref:`development-python2and3` if you care why).

Optional dependencies (imported and used only where needed):

* `scipy library <http://scipy.org/scipylib/index.html>`_ for numerical methods
* `matplotlib`_ for plotting
* `wcsaxes`_ for sky image plotting (provides a low-level API)
* `aplpy`_ for sky image plotting (provides a high-level API)
* `pandas`_ CVS read / write; DataFrame
* `scikit-image`_ for some image processing tasks
* `scikit-learn`_ for some data analysis tasks
* `GammaLib`_ and `ctools`_ for simulating data and likelihood fitting
* `Sherpa`_ for modeling / fitting
* `ROOT`_ and `rootpy`_ conversion helper functions
* `photutils`_ for image photometry
* `imfun`_ for a trous wavelet decomposition
* `uncertainties`_ for Gaussian error propagation
* `imageutils`_ for image utility functions (temporary, will become `astropy.image`)
* `reproject`_ for image reprojection (temporary, will become `astropy.reproject`)

.. note:: We didn't put any effort into minimizing the number of dependencies ...
   I'll limit the number of optional packages if people complain about installation woes.

How to make Astropy / Gammapy work with the CIAO Sherpa Python?
---------------------------------------------------------------

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

Try installing `pip`_ into that Python and then:

.. code-block:: bash

   $ pip install astropy
   $ pip install gammapy

If this doesn't work (which is not uncommon, this is known to fail to compile the C extensions of Astropy
on some platforms), ask your Python-installation-savvy co-worker or on the Astropy or Gammapy mailing list.
