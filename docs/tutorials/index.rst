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
   gammapy-pfmap/index
   gammapy-pfspec/index
   background/index

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
see `here <http://nbviewer.ipython.org/github/gammapy/gammapy-extra/blob/master/notebooks/Index.ipynb>`__. 

External
========

TODO: make a separate sub-page for presentations / posters / papers about Gammapy and move this:

* `Astropy tutorial by Axel Donath from December 2013
  <http://nbviewer.ipython.org/github/adonath/gamma_astropy_talk/blob/master/gamma_astropy_talk.ipynb>`_
