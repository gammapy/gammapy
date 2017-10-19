
Gammapy tutorial notebooks
==========================

What is this?
-------------

-  This is an overview of tutorial `Jupyter <http://jupyter.org/>`__
   notebooks for `Gammapy <http://gammapy.org>`__, a Python package for
   gamma-ray astronomy.
-  The notebooks complement the Gammapy Sphinx-based documentation at
   http://docs.gammapy.org
-  You can read a static HTML version of these notebooks (i.e. code
   can't be executed) online
   `here <http://nbviewer.ipython.org/github/gammapy/gammapy-extra/tree/master/notebooks/>`__.
-  The notebooks and example datasets are available at
   https://github.com/gammapy/gammapy-extra

Set up
------

If you want to execute the notebooks locally, to play around with the
examples, or to try to do one of the exercises, you have to first
install Gammapy and get the ``gammapy-extra`` repository. This is
described in `Gammapy tutorial
setup <notebooks/tutorial_setup.ipynb>`__.

The basics
----------

Gammapy is a Python package built on Numpy and Astropy, and the
tutorials are Jupyter notebooks. If you're already familar with those,
you can skip to the next section and start learning about Gammapy.

If you're new to Python or scientific Python, we can recommmend the
following resources:

-  `A Whirlwind tour of
   Python <http://nbviewer.jupyter.org/github/jakevdp/WhirlwindTourOfPython/blob/master/Index.ipynb>`__
   (learn Python)
-  `Python data science
   handbook <http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb>`__
   (learn IPython, Numpy, matplotlib)
-  `Scipy lectures <http://www.scipy-lectures.org/>`__ (another intro to
   scientific Python)

If you're new to Astropy, here's some good resources:

-  http://www.astropy.org/astropy-tutorials
-  http://astropy.readthedocs.io
-  https://python4astronomers.github.io/

Notebooks
---------

If you're new to Astro (haven't used ``Table``, ``SkyCoord`` and
``Time`` much), start here:

-  `Astropy introduction for Gammapy
   users <notebooks/astropy_introduction.ipynb>`__
-  `Astropy Hands On (1st ASTERICS-OBELICS International
   School) <https://github.com/Asterics2020-Obelics/School2017/blob/master/astropy/astropy_hands_on.ipynb>`__

For a quick introduction to Gammapy, go here:

-  `First steps with Gammapy <notebooks/first_steps.ipynb>`__

Interested to do a first analysis of simulated CTA data?

-  `CTA first data challenge (1DC) with
   Gammapy <notebooks/cta_1dc_introduction.ipynb>`__
-  `CTA data analysis with
   Gammapy <notebooks/cta_data_analysis.ipynb>`__

To learn how to work with gamma-ray data with Gammapy:

-  `IACT DL3 data with Gammapy <notebooks/data_iact.ipynb>`__ (H.E.S.S.
   data example)
-  `Fermi-LAT data with Gammapy <notebooks/data_fermi_lat.ipynb>`__
   (Fermi-LAT data example)

2-dimensional sky image analysis:

-  `Image analysis with Gammapy (run
   pipeline) <notebooks/image_pipe.ipynb>`__ (H.E.S.S. data example)
-  `Image analysis with Gammapy (individual
   steps) <notebooks/image_analysis.ipynb>`__ (H.E.S.S. data example)
-  `Source detection with Gammapy <notebooks/detect_ts.ipynb>`__
   (Fermi-LAT data example)

1-dimensional spectral analysis:

-  `Spectral models in Gammapy <notebooks/spectrum_models.ipynb>`__
-  `Spectral analysis with Gammapy (run
   pipeline) <notebooks/spectrum_pipe.ipynb>`__ (H.E.S.S. data example)
-  `Spectral analysis with Gammapy (individual
   steps) <notebooks/spectrum_analysis.ipynb>`__ (H.E.S.S. data example)
-  `Spectrum simulation and fitting <notebooks/cta_simulation.ipynb>`__
   (CTA data example with AGN / EBL)
-  `Flux point fitting with
   Gammapy <notebooks/sed_fitting_gammacat_fermi.ipynb>`__
-  `Light curve estimation with Gammapy <notebooks/light_curve.ipynb>`__
   (H.E.S.S. fake data example TBC)

3-dimensional cube analysis:

-  `Cube analysis with Gammapy (part
   1) <notebooks/cube_analysis_part1.ipynb>`__ - compute cubes and mean
   PSF / EDISP
-  `Cube analysis with Gammapy (part
   2) <notebooks/cube_analysis_part2.ipynb>`__ - likelihood fit

Time-related analysis:

-  `Time analysis with Gammapy <notebooks/time_analysis.ipynb>`__ (not
   written yet)

Extra topics
~~~~~~~~~~~~

These notebooks contain examples on some more specialised functionality
in Gammapy.

Most users will not need them. It doesn't make much sense that you read
through all of them, but maybe browse the list and see if there's
something that could be interesting for your work (or contribute to
Gammapy if something is missing!).

-  `Template background model production with
   Gammapy <notebooks/background_model.ipynb>`__
-  `Continuous wavelet transform on gamma-ray
   images <notebooks/cwt.ipynb>`__
-  `Interpolation using the NDDataArray
   class <notebooks/nddata_demo.ipynb>`__
-  `Rapid introduction on using numpy, scipy,
   matplotlib <notebooks/using_numpy.ipynb>`__

Work in progress
~~~~~~~~~~~~~~~~

The following notebooks are work in progress or broken.

Please help make these better, or write new, better ones!

-  TODO: joint Fermi-IACT analysis with Gammpy
-  TODO: simulate Fermi and CTA sky with Gammapy

-  `Astrophysical source population modeling with
   Gammapy <notebooks/source_population_model.ipynb>`__
-  `notebooks/source\_catalogs.ipynb <notebooks/source_catalogs.ipynb>`__
   : Working with gamma-ray source catalogs
-  `notebooks/diffuse\_model\_computation.ipynb <notebooks/diffuse_model_computation.ipynb>`__
   : Diffuse model computation
-  `notebooks/fermi\_vela\_model.ipynb <notebooks/fermi_vela_model.ipynb>`__
   : Fermi Vela model
-  `Simulating and analysing sources and diffuse
   emission <notebooks/source_diffuse_estimation.ipynb>`__
