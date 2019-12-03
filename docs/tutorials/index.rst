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

Notebooks
---------

.. toctree::
   :hidden:

   ../notebooks/analysis_1.ipynb
   ../notebooks/analysis_2.ipynb
   ../notebooks/cta.ipynb
   ../notebooks/hess.ipynb
   ../notebooks/fermi_lat.ipynb
   ../notebooks/cta_data_analysis.ipynb
   ../notebooks/analysis_3d.ipynb
   ../notebooks/analysis_mwl.ipynb
   ../notebooks/simulate_3d.ipynb
   ../notebooks/detect.ipynb
   ../notebooks/spectrum_analysis.ipynb
   ../notebooks/sed_fitting.ipynb
   ../notebooks/light_curve.ipynb
   ../notebooks/light_curve_flare.ipynb
   ../notebooks/cta_sensitivity.ipynb
   ../notebooks/spectrum_simulation.ipynb
   ../notebooks/image_analysis.ipynb
   ../notebooks/overview.ipynb
   ../notebooks/maps.ipynb
   ../notebooks/modeling.ipynb
   ../notebooks/models.ipynb
   ../notebooks/catalog.ipynb

**Getting started**

The following tutorials show the same 3D cube analysis of the Crab with the high level
interface and with the lower level API:

- `First analysis <../notebooks/analysis_1.html>`__ | *analysis_1.ipynb*
- `Second analysis <../notebooks/analysis_2.html>`__ | *analysis_2.ipynb*

**What data can I analyse?**

- `CTA with Gammapy <../notebooks/cta.html>`__ | *cta.ipynb*
- `H.E.S.S. with Gammapy <../notebooks/hess.html>`__ |  *hess.ipynb*
- `Fermi-LAT with Gammapy <../notebooks/fermi_lat.html>`__ | *fermi_lat.ipynb*

**Analyses**

*3-dim sky cube analysis*

- `CTA data analysis with Gammapy <../notebooks/cta_data_analysis.html>`__ | *cta_data_analysis.ipynb*
- `3D analysis <../notebooks/analysis_3d.html>`__ | *analysis_3d.ipynb*
- `Multi instrument joint 3D and 1D analysis <../notebooks/analysis_mwl.html>`__ | *analysis_mwl.ipynb*

*2-dim sky image analysis*

- `2D map analysis <../notebooks/image_analysis.html>`__ | *image_analysis.ipynb*
- `Source detection <../notebooks/detect.html>`__ | *detect.ipynb*

*1-dim spectral analysis*

- `Spectral analysis  <../notebooks/spectrum_analysis.html>`__ | *spectrum_analysis.ipynb*
- `Flux point fitting <../notebooks/sed_fitting.html>`__ | *sed_fitting.ipynb*

*Time-dependent analysis*

- `Light curves <../notebooks/light_curve.html>`__ | *light_curve.ipynb*
- `Light curves for flares <../notebooks/light_curve_flare.html>`__ | *light_curve_flare.ipynb*

**Simulations, Sensitivity, Observability**

- `3D map simulation <../notebooks/simulate_3d.html>`__ | *simulate_3d.ipynb*
- `1D spectrum simulation <../notebooks/spectrum_simulation.html>`__ | *spectrum_simulation.ipynb*
- `Point source sensitivity <../notebooks/cta_sensitivity.html>`__ | *cta_sensitivity.ipynb*

**Gammapy package**

- `Overview <../notebooks/overview.html>`__  | *overview.ipynb*
- `Maps <../notebooks/maps.html>`__  | *maps.ipynb*
- `Modeling and Fitting <../notebooks/modeling.html>`__  | *modeling.ipynb*
- `Models Gallery <../notebooks/models.html>`__  | *models.ipynb*
- `Source catalogs <../notebooks/catalog.html>`__  | *catalog.ipynb*

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
