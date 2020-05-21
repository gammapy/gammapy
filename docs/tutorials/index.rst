.. include:: ../references.txt

.. _tutorials:

Tutorials
=========

This page lists the Gammapy tutorials that are available as `Jupyter`_ notebooks.

You can read them here, or execute them using a temporary cloud server in Binder.

To execute them locally, you have to first install Gammapy locally (see
:ref:`install`) and download the tutorial notebooks and example datasets (see
:ref:`getting-started`). Once Gammapy is installed, remember that you can always
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

   analysis_1.ipynb
   analysis_2.ipynb

The following tutorials show how to use gammapy to perform a complete data analysis,
here a simple 3D cube analysis of the Crab. They show the gammapy workflow from data selection
to data reduction and finally modeling and fitting.

First, we show how to do it with the high level interface in configuration-driven approach.
The second tutorial exposes the same analysis, this time using the medium level API, showing
what is happening 'under-the-hood':

- `Configuration driven analysis <analysis_1.html>`__ | *analysis_1.ipynb*
- `Lower level analysis <analysis_2.html>`__ | *analysis_2.ipynb*


Core tutorials
--------------

.. toctree::
   :hidden:

   cta.ipynb
   hess.ipynb
   fermi_lat.ipynb
   cta_data_analysis.ipynb
   analysis_3d.ipynb
   simulate_3d.ipynb
   spectrum_analysis.ipynb
   sed_fitting.ipynb
   light_curve.ipynb
   light_curve_flare.ipynb
   light_curve_simulation.ipynb
   spectrum_simulation.ipynb
   modeling_2D.ipynb
   ring_background.ipynb
   event_sampling.ipynb

The following tutorials expose common analysis tasks.

*Accessing and exploring DL3 data*

- `Overview <overview.html>`__  | *overview.ipynb*
- `CTA with Gammapy <cta.html>`__ | *cta.ipynb*
- `H.E.S.S. with Gammapy <hess.html>`__ |  *hess.ipynb*

*1-dim spectral analysis*

- `Spectral analysis <spectrum_analysis.html>`__ | *spectrum_analysis.ipynb*
- `Flux point fitting <sed_fitting.html>`__ | *sed_fitting.ipynb*

*2-dim sky image analysis*

- `Ring background map creation <ring_background.html>`__ | *ring_background.ipynb*
- `2D map fitting <modeling_2D.html>`__ | *modeling_2D.ipynb*

*3-dim sky cube analysis*

- `CTA data analysis <cta_data_analysis.html>`__ | *cta_data_analysis.ipynb*
- `3D analysis <analysis_3d.html>`__ | *analysis_3d.ipynb*

*Time-dependent analysis*

- `Light curves <light_curve.html>`__ | *light_curve.ipynb*
- `Light curves for flares <light_curve_flare.html>`__ | *light_curve_flare.ipynb*
- `Simulating and fiting a time varying source <light_curve_simulation.html>`__ | *light_curve_simulation.ipynb*

*Simulations*

- `1D spectrum simulation <spectrum_simulation.html>`__ | *spectrum_simulation.ipynb*
- `3D map simulation <simulate_3d.html>`__ | *simulate_3d.ipynb*
- `Event sampling <event_sampling.html>`__ | *event_sampling.ipynb*


Advanced tutorials
------------------

.. toctree::
   :hidden:

   analysis_mwl.ipynb
   extended_source_spectral_analysis.ipynb
   detect.ipynb
   cta_sensitivity.ipynb
   modeling_2D.ipynb
   ring_background.ipynb
   exclusion_mask.ipynb
   overview.ipynb
   maps.ipynb
   modeling.ipynb
   models.ipynb
   catalog.ipynb

The following tutorials expose how to perform more complex analyses or they demonstrate how to use the
Gammapy API.

*Exclusion masks*

- `How to create an exclusion mask <exclusion_mask.html>`__ | *exclusion_mask.ipynb*

*Source detection*

- `Source detection and significance maps <detect.html>`__ | *detect.ipynb*

*Spectral analysis*

- `Spectral analysis of extended sources <extended_source_spectral_analysis.html>`__ | *extended_source_spectral_analysis.ipynb*

*Multi-instrument analysis*

- `Multi instrument joint 3D and 1D analysis <analysis_mwl.html>`__ | *analysis_mwl.ipynb*
- `A Fermi-LAT analysis with Gammapy <fermi_lat.html>`__ | *fermi_lat.ipynb*

*Sensitivity estimation*

- `Point source sensitivity <cta_sensitivity.html>`__ | *cta_sensitivity.ipynb*

*Modeling and fitting in gammapy*

- `Modeling and Fitting <modeling.html>`__  | *modeling.ipynb*
- `Models <models.html>`__  | *models.ipynb*

*Working with catalogs*

- `Source catalogs <catalog.html>`__  | *catalog.ipynb*

*Working with gammapy maps*

- `Maps <maps.html>`__  | *maps.ipynb*


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

    astro_dark_matter.ipynb
    background_model.ipynb
    mcmc_sampling.ipynb
    pulsar_analysis.ipynb

These notebooks contain examples on some more specialised functionality in Gammapy.

- `Dark matter spatial and spectral models <astro_dark_matter.html>`__ | *astro_dark_matter.ipynb*
- `Make template background model <background_model.html>`__ | *background_model.ipynb*
- `MCMC sampling of Gammapy models using the emcee package <mcmc_sampling.html>`__ | *mcmc_sampling.ipynb*
- `Pulsar analysis with Gammapy <pulsar_analysis.html>`__ | *pulsar_analysis.ipynb*
