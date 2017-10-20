
Gammapy tutorial notebooks
==========================

What is this?
-------------

-  This is an overview of tutorial `Jupyter <http://jupyter.org/>`
   notebooks for `Gammapy <http://gammapy.org>`, a Python package for
   gamma-ray astronomy.
-  The notebooks complement the Gammapy Sphinx-based documentation at
   http://docs.gammapy.org
-  You can read a static HTML version of these notebooks (i.e. code
   can't be executed) online
   `here <http://nbviewer.ipython.org/github/gammapy/gammapy-extra/tree/master/notebooks/>`.
-  The notebooks and example datasets are available at
   https://github.com/gammapy/gammapy-extra

Set up
------

If you want to execute the notebooks locally, to play around with the
examples, or to try to do one of the exercises, you have to first
install Gammapy and get the ``gammapy-extra`` repository. This is
described in `Gammapy tutorial
setup <notebooks/tutorial_setup.html>`__.

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
   users <notebooks/astropy_introduction.html>`__
-  `Astropy Hands On (1st ASTERICS-OBELICS International
   School) <https://github.com/Asterics2020-Obelics/School2017/blob/master/astropy/astropy_hands_on.ipynb>`__

For a quick introduction to Gammapy, go here:

-  `First steps with Gammapy <notebooks/first_steps.html>`__

Interested to do a first analysis of simulated CTA data?

-  `CTA first data challenge (1DC) with
   Gammapy <notebooks/cta_1dc_introduction.html>`__
-  `CTA data analysis with
   Gammapy <notebooks/cta_data_analysis.html>`__

To learn how to work with gamma-ray data with Gammapy:

-  `IACT DL3 data with Gammapy <notebooks/data_iact.html>`__ (H.E.S.S.
   data example)
-  `Fermi-LAT data with Gammapy <notebooks/data_fermi_lat.html>`__
   (Fermi-LAT data example)

2-dimensional sky image analysis:

-  `Image analysis with Gammapy (run
   pipeline) <notebooks/image_pipe.html>`__ (H.E.S.S. data example)
-  `Image analysis with Gammapy (individual
   steps) <notebooks/image_analysis.html>`__ (H.E.S.S. data example)
-  `Source detection with Gammapy <notebooks/detect_ts.html>`__
   (Fermi-LAT data example)

1-dimensional spectral analysis:

-  `Spectral models in Gammapy <notebooks/spectrum_models.html>`__
-  `Spectral analysis with Gammapy (run
   pipeline) <notebooks/spectrum_pipe.html>`__ (H.E.S.S. data example)
-  `Spectral analysis with Gammapy (individual
   steps) <notebooks/spectrum_analysis.html>`__ (H.E.S.S. data example)
-  `Spectrum simulation and fitting <notebooks/cta_simulation.html>`__
   (CTA data example with AGN / EBL)
-  `Flux point fitting with
   Gammapy <notebooks/sed_fitting_gammacat_fermi.html>`__
-  `Light curve estimation with Gammapy <notebooks/light_curve.html>`__
   (H.E.S.S. fake data example TBC)

3-dimensional cube analysis:

-  `Cube analysis with Gammapy (part
   1) <notebooks/cube_analysis_part1.html>`__ - compute cubes and mean
   PSF / EDISP
-  `Cube analysis with Gammapy (part
   2) <notebooks/cube_analysis_part2.html>`__ - likelihood fit

Time-related analysis:

-  `Time analysis with Gammapy <notebooks/time_analysis.html>`__ (not
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
   Gammapy <notebooks/background_model.html>`__
-  `Continuous wavelet transform on gamma-ray
   images <notebooks/cwt.html>`__
-  `Interpolation using the NDDataArray
   class <notebooks/nddata_demo.html>`__
-  `Rapid introduction on using numpy, scipy,
   matplotlib <notebooks/using_numpy.html>`__

Work in progress
~~~~~~~~~~~~~~~~

The following notebooks are work in progress or broken.

Please help make these better, or write new, better ones!

-  TODO: joint Fermi-IACT analysis with Gammpy
-  TODO: simulate Fermi and CTA sky with Gammapy

-  `Astrophysical source population modeling with
   Gammapy <notebooks/source_population_model.html>`__
-  `Source catalogs <notebooks/source_catalogs.html>`__
   : Working with gamma-ray source catalogs
-  `Diffuse model computation <notebooks/diffuse_model_computation.html>`__
   : Diffuse model computation
-  `Fermi Vela model <notebooks/fermi_vela_model.html>`__
   : Fermi Vela model
-  `Simulating and analysing sources and diffuse
   emission <notebooks/source_diffuse_estimation.html>`__

List of notebooks
~~~~~~~~~~~~~~~~~~
.. toctree::
   :titlesonly:
   :glob:

   notebooks/*
