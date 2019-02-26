.. include:: references.txt

.. _tutorials:

Tutorial notebooks
==================

This page lists the Gammapy tutorials that are available as `Jupyter`_ notebooks.

You can read them here, or execute them using a temporary cloud server in Binder.

To execute them locally, you have to first install Gammapy locally and download
the tutorial notebooks and example datasets. The setup steps are described in
:ref:`getting-started`. Once Gammapy installed, remember that you can always use
``gammapy info`` to check your setup.

.. _tutorials_notebooks:

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
   notebooks/hess.ipynb
   notebooks/detect_ts.ipynb
   notebooks/image_fitting_with_sherpa.ipynb
   notebooks/spectrum_models.ipynb
   notebooks/spectrum_pipe.ipynb
   notebooks/spectrum_analysis.ipynb
   notebooks/spectrum_fitting_with_sherpa.ipynb
   notebooks/sed_fitting_gammacat_fermi.ipynb
   notebooks/fermi_lat.ipynb
   notebooks/light_curve.ipynb
   notebooks/cta_sensitivity.ipynb
   notebooks/spectrum_simulation.ipynb
   notebooks/spectrum_simulation_cta.ipynb
   notebooks/astropy_introduction.ipynb
   notebooks/image_analysis.ipynb


For a quick introduction to Gammapy, go here:

- `First steps with Gammapy <notebooks/first_steps.html>`__  | *first_steps.ipynb*
- `Introduction to gammapy.maps <notebooks/intro_maps.html>`__  | *intro_maps.ipynb*

Interested to do a first analysis of simulated CTA data?

- `CTA first data challenge (1DC) with Gammapy <notebooks/cta_1dc_introduction.html>`__ | *cta_1dc_introduction.ipynb*
- `CTA data analysis with Gammapy <notebooks/cta_data_analysis.html>`__ | *cta_data_analysis.ipynb*

To get started with H.E.S.S. data analysis see here:

- `H.E.S.S. with Gammapy <notebooks/hess.html>`__ | *hess.ipynb*

3-dimensional cube analysis:

- `3D analysis <notebooks/analysis_3d.html>`__ | *analysis_3d.ipynb*
- `3D simulation and fitting <notebooks/simulate_3d.html>`__ | *simulate_3d.ipynb*
- `Fermi-LAT data with Gammapy <notebooks/fermi_lat.html>`__ | *fermi_lat.ipynb*

2-dimensional sky image analysis:

- `Source detection with Gammapy <notebooks/detect_ts.html>`__ (Fermi-LAT data example) | *detect_ts.ipynb*
- `CTA 2D source fitting with Gammapy <notebooks/image_analysis.html>`__ (DC 1 example) | *image_analysis.ipynb*
- `CTA 2D source fitting with Sherpa <notebooks/image_fitting_with_sherpa.html>`__ | *image_fitting_with_sherpa.ipynb*


1-dimensional spectral analysis:

- `Spectral models in Gammapy <notebooks/spectrum_models.html>`__ | *spectrum_models.ipynb*
- `Spectral analysis with Gammapy (run pipeline) <notebooks/spectrum_pipe.html>`__ (H.E.S.S. data example) | *spectrum_pipe.ipynb*
- `Spectral analysis with Gammapy (individual steps) <notebooks/spectrum_analysis.html>`__ (H.E.S.S. data example) | *spectrum_analysis.ipynb*
- `Fitting gammapy spectra with sherpa <notebooks/spectrum_fitting_with_sherpa.html>`__ | *spectrum_fitting_with_sherpa.ipynb*
- `Flux point fitting with Gammapy <notebooks/sed_fitting_gammacat_fermi.html>`__ | *sed_fitting_gammacat_fermi.ipynb*

Time-dependent analysis:

- `Light curves <notebooks/light_curve.html>`__ | *light_curve.ipynb*

Sensitivity:

- `Compute the CTA sensitivity <notebooks/cta_sensitivity.html>`__ | *cta_sensitivity.ipynb*

.. _tutorials_extras:

Extra topics
------------
.. toctree::
    :hidden:

    notebooks/hgps.ipynb
    notebooks/source_population_model.ipynb
    notebooks/cwt.ipynb
    notebooks/astro_dark_matter.ipynb
    notebooks/background_model.ipynb

These notebooks contain examples on some more specialised functionality in Gammapy.

- `H.E.S.S. Galactic plane survey (HGPS) data <notebooks/hgps.html>`__ | *hgps.ipynb*
- `Astrophysical source population modeling with Gammapy <notebooks/source_population_model.html>`__ | *source_population_model.ipynb*
- `Continuous wavelet transform on gamma-ray images <notebooks/cwt.html>`__ | *cwt.ipynb*
- `Dark matter spatial and spectral models <notebooks/astro_dark_matter.html>`__ | *astro_dark_matter.ipynb*
- `Make template background model <notebooks/background_model.html>`__ | *background_model.ipynb*

.. _tutorials_basics:

Basics
------

Gammapy is a Python package built on Numpy and Astropy, so for now you have to learn
a bit of Python, Numpy and Astropy to be able to use Gammapy.
To make plots you have to learn a bit of matplotlib.

We plan to add a very simple to use high-level interface to Gammapy where you just have to
adjust a config file, but that isn't available yet.

Here are some great resources:

- Python: `A Whirlwind tour of Python <https://nbviewer.jupyter.org/github/jakevdp/WhirlwindTourOfPython/blob/master/Index.ipynb>`__
- IPython, Jupyter, Numpy, matplotlib: `Python data science handbook <http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb>`__
- `Astropy introduction for Gammapy users <notebooks/astropy_introduction.html>`__  | *astropy_introduction.ipynb*
- `Astropy Hands On (1st ASTERICS-OBELICS International School) <https://github.com/Asterics2020-Obelics/School2017/blob/master/astropy/astropy_hands_on.ipynb>`__

Other useful resources:

- http://www.astropy.org/astropy-tutorials
- http://astropy.readthedocs.io
- https://python4astronomers.github.io
