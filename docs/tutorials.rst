.. include:: references.txt

.. _tutorials:

Tutorial notebooks
==================

What is this?
-------------

-  This is an overview of tutorial `Jupyter <http://jupyter.org/>`__
   notebooks for `Gammapy <http://gammapy.org>`__, a Python package for
   gamma-ray astronomy.
-  The notebooks complement the Gammapy Sphinx-based documentation at
   http://docs.gammapy.org
-  The notebooks and example datasets are available at
   https://github.com/gammapy/gammapy-extra

Set up
------

The Gammapy installation instructions are here: :ref:`install`

One quick way to get set up, that works the same on Linux, Mac and Windows is
this:

* Install Anaconda or Miniconda (see https://www.anaconda.com/download/ )
* Get the following repository that contains the Gammapy tutorial notebooks::

    git clone https://github.com/gammapy/gammapy-extra.git
    export GAMMAPY_EXTRA=$PWD/gammapy-extra

* Create a Python conda environment that contains all software used in the tutorials::

    cd gammapy-extra
    conda env create -f environment.yml

* If you already have that environment, but want to update::

    conda env update -f environment.yml

* Activate the environment and start Jupyter::

    source activate gammapy-tutorial
    cd notebooks
    jupyter notebook

* Select and start the notebook you want in your webbrowser.

If you have any questions, ask for help. See http://gammapy.org/contact.html

Execute tutorials online
------------------------

.. image:: http://mybinder.org/badge.svg

You can execute the notebooks on-line.
Just click on the *launch binder* badge placed at the top of each of the notebooks below.

Note that this is a free, temporary notebook server. You cannot save your work there and retrieve it afterwards.
For that, install Gammapy on your machine and work there.

The basics
----------

Gammapy is a Python package built on Numpy and Astropy, and the tutorials are
Jupyter notebooks. If you're already familar with those, you can skip to the
next section and start learning about Gammapy.

To learn the basics, here are a few good resources.

Python
++++++

- `A Whirlwind tour of Python <http://nbviewer.jupyter.org/github/jakevdp/WhirlwindTourOfPython/blob/master/Index.ipynb>`__ (learn Python)

Scientific Python
+++++++++++++++++

- `Python data science handbook <http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb>`__ (learn IPython, Numpy, matplotlib)

Astropy
+++++++

- `Astropy introduction for Gammapy users <notebooks/astropy_introduction.html>`__  | *astropy_introduction.ipynb*
- `Astropy Hands On (1st ASTERICS-OBELICS International School) <https://github.com/Asterics2020-Obelics/School2017/blob/master/astropy/astropy_hands_on.ipynb>`__

Other useful resources:

- http://www.astropy.org/astropy-tutorials
- http://astropy.readthedocs.io
- https://python4astronomers.github.io/

Notebooks
---------
.. toctree::
   :hidden:

   notebooks/first_steps.ipynb
   notebooks/intro_maps.ipynb
   notebooks/cta_1dc_introduction.ipynb
   notebooks/cta_data_analysis.ipynb
   notebooks/analysis_3d.ipynb
   notebooks/simulate_3d.ipynb
   notebooks/detect_ts.ipynb
   notebooks/image_fitting_with_sherpa.ipynb
   notebooks/spectrum_models.ipynb
   notebooks/spectrum_pipe.ipynb
   notebooks/spectrum_analysis.ipynb
   notebooks/spectrum_fitting_with_sherpa.ipynb
   notebooks/sed_fitting_gammacat_fermi.ipynb
   notebooks/fermi_lat.ipynb

For a quick introduction to Gammapy, go here:

- `First steps with Gammapy <notebooks/first_steps.html>`__  | *first_steps.ipynb*
- `Introduction to gammapy.maps <notebooks/intro_maps.html>`__  | *intro_maps.ipynb*

Interested to do a first analysis of simulated CTA data?

- `CTA first data challenge (1DC) with Gammapy <notebooks/cta_1dc_introduction.html>`__ | *cta_1dc_introduction.ipynb*
- `CTA data analysis with Gammapy <notebooks/cta_data_analysis.html>`__ | *cta_data_analysis.ipynb*

3-dimensional cube analysis:

- `3D analysis <notebooks/analysis_3d.html>`__ | *analysis_3d.ipynb*
- `3D simulation and fitting <notebooks/simulate_3d.html>`__ | *simulate_3d.ipynb*
- `Fermi-LAT data with Gammapy <notebooks/fermi_lat.html>`__ | *fermi_lat.ipynb*

2-dimensional sky image analysis:

- `Image analysis with Gammapy (individual steps) <notebooks/image_analysis.html>`__ (H.E.S.S. data example) | *image_analysis.ipynb*
- `Source detection with Gammapy <notebooks/detect_ts.html>`__ (Fermi-LAT data example) | *detect_ts.ipynb*
- `CTA 2D source fitting with Sherpa <notebooks/image_fitting_with_sherpa.html>`__ | *image_fitting_with_sherpa.ipynb*

1-dimensional spectral analysis:

- `Spectral models in Gammapy <notebooks/spectrum_models.html>`__ | *spectrum_models.ipynb*
- `Spectral analysis with Gammapy (run pipeline) <notebooks/spectrum_pipe.html>`__ (H.E.S.S. data example) | *spectrum_pipe.ipynb*
- `Spectral analysis with Gammapy (individual steps) <notebooks/spectrum_analysis.html>`__ (H.E.S.S. data example) | *spectrum_analysis.ipynb*
- `Fitting gammapy spectra with sherpa <notebooks/spectrum_fitting_with_sherpa.html>`__ | *spectrum_fitting_with_sherpa.ipynb*
- `Flux point fitting with Gammapy <notebooks/sed_fitting_gammacat_fermi.html>`__ | *sed_fitting_gammacat_fermi.ipynb*

Extra topics
------------
.. toctree::
    :hidden:

    notebooks/hgps.ipynb
    notebooks/source_population_model.ipynb
    notebooks/cwt.ipynb
    notebooks/astro_dark_matter.ipynb

These notebooks contain examples on some more specialised functionality in Gammapy.

- `H.E.S.S. Galactic plane survey (HGPS) data <notebooks/hgps.html>`__ | *hgps.ipynb*
- `Astrophysical source population modeling with Gammapy <notebooks/source_population_model.html>`__ | *source_population_model.ipynb*
- `Continuous wavelet transform on gamma-ray images <notebooks/cwt.html>`__ | *cwt.ipynb*
- `Dark matter spatial and spectral models <notebooks/astro_dark_matter.html>`__ | *astro_dark_matter.ipynb*
