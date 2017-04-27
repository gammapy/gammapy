.. _tutorials:

**********************
Tutorials and Examples
**********************

We currently have three places for Gammapy tutorials and examples:

1. Python scripts with Sphinx documentation in the ``docs/tutorials`` folder in the ``gammapy`` repo.
2. Python scripts without inline comments only in the ``examples`` folder in the ``gammapy`` repo.
3. IPython notebooks in the ``notebooks`` folder in the ``gammapy-extra`` repo.

Each of these solutions has advantages / disadvantages, we might consolidate this in the future as
the tooling to convert between these formats improves.

.. _tutorials-sphinx:

Python example scripts with Sphinx documentation
================================================

The tutorials show some real-world usage examples of the Gammapy Python package and / or command line tools.

.. toctree::
   :maxdepth: 1

   crab_mwl_sed/index
   npred/index
   catalog/index
   flux_point/index
   background/index
   fermi_psf/index

.. _tutorials-examples:

Python example scripts
======================

The ``examples`` folder in the ``gammapy`` repo contains small Python scripts
illustrating how to use Gammapy.

.. note:: For now the full list of examples is only available here:
   https://github.com/gammapy/gammapy/tree/master/examples
   
   We plan to integrate that into the online Sphinx docs ...
   please help if you know how to do this:
   https://github.com/gammapy/gammapy/issues/172

.. _tutorials-notebook:

IPython notebooks
=================

The IPython notebooks are in the `gammapy-extra <https://github.com/gammapy/gammapy-extra>`__ repo,
see `here <https://nbviewer.ipython.org/github/gammapy/gammapy-extra/blob/master/notebooks/Index.ipynb>`__.

External
========

Here's some links to good external resources to learn Python, Numpy, Scipy, Astropy, ...

Python is very popular in astronomy (and data science in general).
It's possible to learn the basics within a day and to become productive within a week.

* Tom Robitaille Python for scientists workshop –– http://www2.mpia-hd.mpg.de/~robitaille/PY4SCI_SS_2015/
* Scientific Python lecture notes –– https://scipy-lectures.github.io/
* Practical Python for astronomers — http://python4astronomers.github.io
* MPIK Astropy workshop — https://astropy4mpik.readthedocs.io/
* CEA Python for astronomers workshop –– https://github.com/kosack/CEAPythonWorkshopForAstronomers
* ctools — http://cta.irap.omp.eu/ctools-devel/
* Astropy tutorials — http://www.astropy.org/astropy-tutorials/
* Naima examples — http://naima.readthedocs.io/en/latest/
* Fermi ScienceTools analysis threads ––– http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/
* CIAO Sherpa threads ––– http://cxc.harvard.edu/sherpa/threads/index.html
* `Astropy tutorial by Axel Donath from December 2013
  <https://nbviewer.ipython.org/github/adonath/gamma_astropy_talk/blob/master/gamma_astropy_talk.ipynb>`_
