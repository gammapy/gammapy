.. include:: ../references.txt

.. _tutorials:

Tutorials
=========

This page lists the Gammapy tutorials that are available as `Jupyter`_ notebooks.

You can read them here, or execute them using a temporary cloud server in Binder.

To execute them locally, you have to first install Gammapy locally (see
:ref:`install`) and download the tutorial notebooks and example datasets (see
:ref:`getting-started`). Once Gammapy installed, remember that you can always
use ``gammapy info`` to check your setup.

Gammapy is a Python package built on `Numpy`_ and `Astropy`_, so to use it
effectively, you have to learn the basics. Many good free resources are
available, e.g. `A Whirlwind tour of Python`_, the `Python data science
handbook`_ and the `Astropy Hands-On Tutorial`_.

.. _tutorials_notebooks:

Getting started
---------------

.. toctree::
   :hidden:

   ../notebooks/analysis_1.ipynb
   ../notebooks/analysis_2.ipynb

The following tutorials show how to use gammapy to perform a complete data analysis,
here a simple 3D cube analysis of the Crab. They show the gammapy workflow from data selection
to data reduction and finally modeling and fitting.

First, we show how to do it with the high level interface in configuration-driven approach.
The second tutorial exposes the same analysis, this time using the medium level API, showing
what is happening 'under-the-hood':

- `Configuration driven analysis <../notebooks/analysis_1.html>`__ | *analysis_1.ipynb*
- `Lower level analysis <../notebooks/analysis_2.html>`__ | *analysis_2.ipynb*


Core tutorials
--------------

.. toctree::
   :hidden:

   ../notebooks/cta.ipynb
   ../notebooks/hess.ipynb
   ../notebooks/fermi_lat.ipynb
   ../notebooks/cta_data_analysis.ipynb
   ../notebooks/analysis_3d.ipynb
   ../notebooks/simulate_3d.ipynb
   ../notebooks/spectrum_analysis.ipynb
   ../notebooks/sed_fitting.ipynb
   ../notebooks/light_curve.ipynb
   ../notebooks/light_curve_flare.ipynb
   ../notebooks/spectrum_simulation.ipynb
   ../notebooks/modeling_2D.ipynb
   ../notebooks/ring_background.ipynb

The following tutorials expose common analysis tasks.

*Accessing and exploring DL3 data*

- `Overview <../notebooks/overview.html>`__  | *overview.ipynb*
- `CTA with Gammapy <../notebooks/cta.html>`__ | *cta.ipynb*
- `H.E.S.S. with Gammapy <../notebooks/hess.html>`__ |  *hess.ipynb*

*1-dim spectral analysis*

- `Spectral analysis <../notebooks/spectrum_analysis.html>`__ | *spectrum_analysis.ipynb*
- `Flux point fitting <../notebooks/sed_fitting.html>`__ | *sed_fitting.ipynb*

*2-dim sky image analysis*

- `Ring background map creation <../notebooks/ring_background.html>`__ | *ring_background.ipynb*
- `2D map fitting <../notebooks/modeling_2D.html>`__ | *modeling_2D.ipynb*

*3-dim sky cube analysis*

- `CTA data analysis <../notebooks/cta_data_analysis.html>`__ | *cta_data_analysis.ipynb*
- `3D analysis <../notebooks/analysis_3d.html>`__ | *analysis_3d.ipynb*

*Time-dependent analysis*

- `Light curves <../notebooks/light_curve.html>`__ | *light_curve.ipynb*
- `Light curves for flares <../notebooks/light_curve_flare.html>`__ | *light_curve_flare.ipynb*

*Simulations*

- `1D spectrum simulation <../notebooks/spectrum_simulation.html>`__ | *spectrum_simulation.ipynb*
- `3D map simulation <../notebooks/simulate_3d.html>`__ | *simulate_3d.ipynb*

Advanced tutorials
------------------

.. toctree::
   :hidden:

   ../notebooks/analysis_mwl.ipynb
   ../notebooks/extended_source_spectral_analysis.ipynb
   ../notebooks/detect.ipynb
   ../notebooks/cta_sensitivity.ipynb
   ../notebooks/modeling_2D.ipynb
   ../notebooks/ring_background.ipynb
   ../notebooks/overview.ipynb
   ../notebooks/maps.ipynb
   ../notebooks/modeling.ipynb
   ../notebooks/models.ipynb
   ../notebooks/catalog.ipynb

The following tutorials expose how to perform more complex analyses or they demonstrate how to use the
Gammapy API.

*Source detection*

- `Source detection and significance maps <../notebooks/detect.html>`__ | *detect.ipynb*

*Spectral analysis*

- `Spectral analysis of extended sources <../notebooks/extended_source_spectral_analysis.html>`__ | *extended_source_spectral_analysis.ipynb*

*Multi-instrument analysis*

- `Multi instrument joint 3D and 1D analysis <../notebooks/analysis_mwl.html>`__ | *analysis_mwl.ipynb*
- `A Fermi-LAT analysis with Gammapy <../notebooks/fermi_lat.html>`__ | *fermi_lat.ipynb*

*Sensitivity estimation*

- `Point source sensitivity <../notebooks/cta_sensitivity.html>`__ | *cta_sensitivity.ipynb*

*Modeling and fitting in gammapy*

- `Modeling and Fitting <../notebooks/modeling.html>`__  | *modeling.ipynb*
- `Models <../notebooks/models.html>`__  | *models.ipynb*

*Working with catalogs*

- `Source catalogs <../notebooks/catalog.html>`__  | *catalog.ipynb*

*Working with gammapy maps*

- `Maps <../notebooks/maps.html>`__  | *maps.ipynb*


.. _tutorials_scripts:

Scripts
-------

Examples how to run Gammapy via Python scripts:

.. toctree::
    :maxdepth: 1

    survey_map


.. _tutorials_extras:

Extra topics
------------

.. toctree::
    :hidden:

    ../notebooks/astro_dark_matter.ipynb
    ../notebooks/background_model.ipynb
    ../notebooks/mcmc_sampling.ipynb
    ../notebooks/pulsar_analysis.ipynb

These notebooks contain examples on some more specialised functionality in Gammapy.

- `Dark matter spatial and spectral models <../notebooks/astro_dark_matter.html>`__ | *astro_dark_matter.ipynb*
- `Make template background model <../notebooks/background_model.html>`__ | *background_model.ipynb*
- `MCMC sampling of Gammapy models using the emcee package <../notebooks/mcmc_sampling.html>`__ | *mcmc_sampling.ipynb*
- `Pulsar analysis with Gammapy <../notebooks/pulsar_analysis.html>`__ | *pulsar_analysis.ipynb*
