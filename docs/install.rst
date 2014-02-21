.. include:: references.txt

.. _install:

Installation
============

``gammapy`` works with Python 2 and Python 3 (specifically 2.6, 2.7 and 3.2 or later).

To install the latest ``gammapy`` stable version the easiest way is using the `pip <http://www.pip-installer.org/>`_ installer::

   pip install gammapy

To install the latest developer version of ``gammapy``, use::

   git clone https://github.com/gammapy/gammapy.git
   cd gammapy
   python setup.py install

To check if ``gammapy`` is correctly installed, start up python or ipython, import ``gammapy`` and run the unit tests::

   >>> import gammapy
   >>> gammapy.test()

To check if the ``gammapy`` command line tools are on your ``$PATH`` try this::

   $ gp-info --tools

Requirements
------------

To install and use this package you need `Astropy`_ version 0.3 or later.

Optional dependencies (imported and used only where needed):

* `scipy library <http://scipy.org/scipylib/index.html>`_ for numerical methods
* `matplotlib <http://matplotlib.org>`_ for plotting
* `pandas <http://pandas.pydata.org>`_ CVS read / write; DataFrame
* `scikit-image`_ for image processing
* `GammaLib`_ and `ctools`_ 
* `Sherpa`_ for modeling / fitting
* `ROOT`_ and `rootpy`_ conversion helper functions
* `photutils`_ for image photometry
* `Kapteyn`_ for reprojecting images
* `aplpy`_ for plotting astro images
* `imfun`_ for a trous wavelet decomposition
* `uncertainties`_ for Gaussian error propagation

.. note:: I didn't put any effort into minimizing the number of dependencies,
   since ``gammapy`` is a prototype. Should it develop into a package that is actually used
   by a few people I'll limit the optional packages to what is actually necessary.

Sherpa
------

Some parts of gammapy use the `Sherpa`_ Python modeling / fitting package
from the `CIAO`_ Chandra satellite X-ray data analysis package. 

Building Sherpa and all the required libraries from source is very difficult.
You should install the binary version of CIAO as described
`here <http://cxc.cfa.harvard.edu/ciao/download>`__,
make sure you include Sherpa and exclude the Chandra CALDB.
But then the Sherpa Python and numpy will not work with the existing
Python, numpy, astropy, gammapy, ... on your system.

You have to re-install Astropy, gammapy and any other Python packages
that you want to use in the same script as Sherpa into the CIAO Python.
Sometimes this just works, but sometimes you'll run into compilation errors
when e.g. the C extensions in ``astropy.wcs`` or ``astropy.io.fits`` are compiled.

Here's a few tricks that might help you make it work.

* Execute the  
  `ciao-python-fix <http://cxc.cfa.harvard.edu/ciao/threads/ciao_install/index.html#ciao_python_fix>`__
  script after installing CIAO::

   cd $CIAO_DIR
   bash bin/ciao-python-fix

* Set ``LDFLAGS`` and use ``ciaorun`` before installing a Python package with C extensions::

   export LDFLAGS="-L${ASCDS_INSTALL}/ots/lib" 
   ciaorun python setup.py install

* Add these folders to your ``PATH`` and ``PYTHONPATH`` so that the right command line tools or
  Python packages are picked up instead of the ones in other system folders::

   export PATH=$CIAO_DIR/ots/bin:$PATH
   export PYTHONPATH=$CIAO_DIR/ots/lib/python2.7/site-packages/:$PYTHONPATH
