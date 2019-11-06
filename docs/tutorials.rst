.. include:: references.txt

.. _tutorials:

Tutorials
=========

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
   notebooks/maps.ipynb
   notebooks/models.ipynb
   notebooks/cta_1dc_introduction.ipynb
   notebooks/cta_data_analysis.ipynb
   notebooks/analysis_3d.ipynb
   notebooks/analysis_3d_joint.ipynb
   notebooks/simulate_3d.ipynb
   notebooks/hess.ipynb
   notebooks/detect_ts.ipynb
   notebooks/image_fitting_with_sherpa.ipynb
   notebooks/spectrum_analysis.ipynb
   notebooks/spectrum_fitting_with_sherpa.ipynb
   notebooks/sed_fitting_gammacat_fermi.ipynb
   notebooks/fermi_lat.ipynb
   notebooks/light_curve.ipynb
   notebooks/cta_sensitivity.ipynb
   notebooks/spectrum_simulation.ipynb
   notebooks/image_analysis.ipynb
   notebooks/joint_1d_3d_analysis.ipynb


For a quick introduction to Gammapy, go here:

- `First steps with Gammapy <notebooks/first_steps.html>`__  | *first_steps.ipynb*
- `Maps <notebooks/maps.html>`__  | *maps.ipynb*
- `Models <notebooks/models.html>`__  | *models.ipynb*

Interested to do a first analysis of simulated CTA data?

- `CTA first data challenge (1DC) with Gammapy <notebooks/cta_1dc_introduction.html>`__ | *cta_1dc_introduction.ipynb*
- `CTA data analysis with Gammapy <notebooks/cta_data_analysis.html>`__ | *cta_data_analysis.ipynb*

To get started with H.E.S.S. data analysis see here:

- `H.E.S.S. with Gammapy <notebooks/hess.html>`__ | *hess.ipynb*

3-dimensional cube analysis:

- `3D analysis <notebooks/analysis_3d.html>`__ | *analysis_3d.ipynb*
- `Joint 3D analysis <notebooks/analysis_3d_joint.html>`__ | *analysis_3d_joint.ipynb*
- `3D simulation and fitting <notebooks/simulate_3d.html>`__ | *simulate_3d.ipynb*
- `Fermi-LAT data with Gammapy <notebooks/fermi_lat.html>`__ | *fermi_lat.ipynb*
- `Joint 3D and 1D analysis <notebooks/joint_1d_3d_analysis.html>`__ | *joint_1d_3d_analysis.ipynb*

2-dimensional sky image analysis:

- `Source detection with Gammapy <notebooks/detect_ts.html>`__ (Fermi-LAT data example) | *detect_ts.ipynb*
- `CTA 2D source fitting with Gammapy <notebooks/image_analysis.html>`__ (DC 1 example) | *image_analysis.ipynb*
- `CTA 2D source fitting with Sherpa <notebooks/image_fitting_with_sherpa.html>`__ | *image_fitting_with_sherpa.ipynb*


1-dimensional spectral analysis:

- `Spectral simulation with Gammapy <notebooks/spectrum_simulation.html>`__ | *spectrum_simulation.ipynb*
- `Spectral analysis with Gammapy  <notebooks/spectrum_analysis.html>`__ (H.E.S.S. data example) | *spectrum_analysis.ipynb*
- `Fitting Gammapy spectra with sherpa <notebooks/spectrum_fitting_with_sherpa.html>`__ | *spectrum_fitting_with_sherpa.ipynb*
- `Flux point fitting with Gammapy <notebooks/sed_fitting_gammacat_fermi.html>`__ | *sed_fitting_gammacat_fermi.ipynb*

Time-dependent analysis:

- `Light curves <notebooks/light_curve.html>`__ | *light_curve.ipynb*

Sensitivity:

- `Compute the CTA sensitivity <notebooks/cta_sensitivity.html>`__ | *cta_sensitivity.ipynb*

.. _tutorials_scripts:

Scripts
-------

TODO: show a few examples how to use Gammapy from Python scripts.

::

    cd $GAMMAPY_DATA/../scripts-0.13
    python cta_1dc_survey_map.py

- TODO: Make a CTA 1DC survey counts map
- TODO: some other long-running analysis or simulation

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
    notebooks/mcmc_sampling.ipynb
    notebooks/pulsar_analysis.ipynb

These notebooks contain examples on some more specialised functionality in Gammapy.

- `H.E.S.S. Galactic plane survey (HGPS) data <notebooks/hgps.html>`__ | *hgps.ipynb*
- `Astrophysical source population modeling with Gammapy <notebooks/source_population_model.html>`__ | *source_population_model.ipynb*
- `Continuous wavelet transform on gamma-ray images <notebooks/cwt.html>`__ | *cwt.ipynb*
- `Dark matter spatial and spectral models <notebooks/astro_dark_matter.html>`__ | *astro_dark_matter.ipynb*
- `Make template background model <notebooks/background_model.html>`__ | *background_model.ipynb*
- `MCMC sampling of Gammapy models using the emcee package <notebooks/mcmc_sampling.html>`__ | *mcmc_sampling.ipynb*
- `Pulsar analysis with Gammapy <notebooks/pulsar_analysis.html>`__ | *pulsar_analysis.ipynb*

.. _tutorials_basics:

Basics
------

Gammapy is a Python package built on `Numpy`_ and `Astropy`_, so to use it effectively,
you have to learn the basics. To make plots you have to learn a bit of `matplotlib`_.

Here are some great hands-on tutorials to get started quickly:

- Python: `A Whirlwind tour of Python <https://nbviewer.jupyter.org/github/jakevdp/WhirlwindTourOfPython/blob/master/Index.ipynb>`__
- IPython, Jupyter, Numpy, matplotlib: `Python data science handbook <http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb>`__
- Astropy: `Astropy Hands-On Tutorial <https://github.com/Asterics2020-Obelics/School2019/tree/master/astropy>`__
