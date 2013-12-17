.. _install:

Installation
============

`gammapy` works with Python 2 and Python 3 (specifically 2.6, 2.7 and 3.3 or later).

To install the latest `gammapy` stable version the easiest way is using the `pip <http://www.pip-installer.org/>`_ installer::

   pip install gammapy

To install the latest developer version of `gammapy`, use::

   git clone https://github.com/gammapy/gammapy.git
   cd astropy
   python setup.py install

To check if `gammapy` is correctly installed, start up python or ipython, import `gammapy` and run the unit tests::

   >>> import gammapy
   >>> gammapy.test()

To check if the `gammapy` command line tools are on your `$PATH` try this::

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

.. note:: I didn't put any effort into minimizing the number of dependencies,
   since `gammapy` is a prototype. Should it develop into a package that is actually used
   by a few people I'll limit the optional packages to what is actually necessary.

Sherpa
------

Some parts of gammapy use the `Sherpa`_ Python modeling / fitting package
from the `CIAO`_ Chandra satellite X-ray data analysis package. 

Building Sherpa and all the required libraries from source is very difficult.
You should install the binary version of CIAO as described
`here <http://cxc.cfa.harvard.edu/ciao/>`__,
make sure you include Sherpa and exclude the Chandra CALDB.
But then the Sherpa Python and numpy will not work with the existing
Python, numpy, astropy, gammapy, ... on your system.
You have to re-install Astropy, gammapy and any other Python packages
that you want to use in the same script as Sherpa into the CIAO Python.

Here's a useful trick that you can try if you get error messages
trying to install Astropy or gammapy::

   export LDFLAGS="-L${ASCDS_INSTALL}/ots/lib" 
   ciaorun python setup.py install

And if for some reason ``import astropy`` or ``import gammapy`` picks up
the system Python version even though you have installed them into the
Sherpa Python, try this:: 

   export PYTHONPATH=$PYTHONPATH:$CIAO_DIR/ots/lib/python2.7/site-packages/

.. _scikit-image: http://scikit-image.org
.. _GammaLib: http://gammalib.sourceforge.net
.. _ctools: http://cta.irap.omp.eu/ctools
.. _Astropy: http://astropy.org
.. _photutils: http://photutils.readthedocs.org
.. _ROOT: http://root.cern.ch/
.. _rootpy: http://rootpy.org
.. _Kapteyn: http://www.astro.rug.nl/software/kapteyn/
.. _Sherpa: http://cxc.cfa.harvard.edu/sherpa/
.. _CIAO: http://cxc.cfa.harvard.edu/ciao/
.. _imfun: http://code.google.com/p/image-funcut/
.. _aplpy: http://aplpy.github.io

