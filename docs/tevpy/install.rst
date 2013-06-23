.. _install:

Installation
============

To install the latest `tevpy` stable version the easiest way is using the `pip <http://www.pip-installer.org/>`_ installer::

   pip install tevpy

To install the latest developer version of `tevpy`, use::

   git clone https://github.com/gammapy/tevpy.git
   cd astropy
   python setup.py install

To check if `tevpy` is correctly installed, start up python or ipython, import `tevpy` and run the unit tests::

   >>> import tevpy
   >>> tevpy.test()

To check if the `tevpy` command line tools are on your `$PATH` try this::

   $ lookup_map_values --help

Requirements
------------

To install and use this package you need `Astropy`_.  

Optional dependencies (imported and used only where needed):

* some of the `scipy stack <http://scipy.org>`_ packages:
  `scipy library <http://scipy.org/scipylib/index.html>`_,
  `matplotlib <http://matplotlib.org>`_,
  `pandas <http://pandas.pydata.org>`_.
* `GammaLib`_ and `ctools`_
* `Sherpa`_
* `ROOT`_ and `rootpy`_
* `photutils`_
* `Kapteyn`_

.. note:: I didn't put any effort into minimizing the number of dependencies,
   since `tevpy` is a prototype. Should it develop into a package that is actually used
   by a few people I'll limit the optional packages to what is actually necessary.

.. _GammaLib: http://gammalib.sourceforge.net
.. _ctools: http://cta.irap.omp.eu/ctools
.. _Astropy: http://astropy.org
.. _photutils: http://photutils.readthedocs.org
.. _ROOT: http://root.cern.ch/
.. _rootpy: http://rootpy.org
.. _Kapteyn: http://www.astro.rug.nl/software/kapteyn/
.. _Sherpa: http://cxc.cfa.harvard.edu/sherpa/
   