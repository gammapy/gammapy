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

The following tutorials show how to use gammapy to perform a complete data analysis,
here a simple 3D cube analysis of the Crab. They show the gammapy workflow from data selection
to data reduction and finally modeling and fitting.

First, we show how to do it with the high level interface in configuration-driven approach.
The second tutorial exposes the same analysis, this time using the medium level API, showing
what is happening 'under-the-hood':

.. nbgallery::

    analysis_1.ipynb
    analysis_2.ipynb


Core tutorials
--------------

The following tutorials expose common analysis tasks.

*Accessing and exploring DL3 data*

.. nbgallery::

    overview.ipynb
    cta.ipynb
    hess.ipynb

*1-dim spectral analysis*

.. nbgallery::

    spectrum_analysis.ipynb
    sed_fitting.ipynb

*2-dim sky image analysis*

.. nbgallery::

    ring_background.ipynb
    modeling_2D.ipynb

*3-dim sky cube analysis*

.. nbgallery::

    cta_data_analysis.ipynb
    analysis_3d.ipynb

*Time-dependent analysis*

.. nbgallery::

    light_curve.ipynb
    light_curve_flare.ipynb
    light_curve_simulation.ipynb

*Simulations*

.. nbgallery::

    spectrum_simulation.ipynb
    simulate_3d.ipynb
    event_sampling.ipynb

Advanced tutorials
------------------

The following tutorials expose how to perform more complex analyses or they demonstrate how to use the
Gammapy API.

.. nbgallery::

    exclusion_mask.ipynb
    detect.ipynb
    extended_source_spectral_analysis.ipynb
    analysis_mwl.ipynb
    fermi_lat.ipynb
    cta_sensitivity.ipynb
    modeling.ipynb
    models.ipynb
    catalog.ipynb
    maps.ipynb


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

These notebooks contain examples on some more specialised functionality in Gammapy.

.. nbgallery::

    astro_dark_matter.ipynb
    background_model.ipynb
    mcmc_sampling.ipynb
    pulsar_analysis.ipynb
