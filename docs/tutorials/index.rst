.. include:: ../references.txt

.. _tutorials:

Tutorials
=========

This page lists the Gammapy tutorials that are available as `Jupyter`_ notebooks.
You can read them here, or execute them using a temporary cloud server in Binder.

To execute them locally, you have to first install Gammapy locally (see
:ref:`installation`) and download the tutorial notebooks and example datasets (see
:ref:`getting-started`). Once Gammapy is installed, remember that you can always
use ``gammapy info`` to check your setup.

Gammapy is a Python package built on `Numpy`_ and `Astropy`_, so to use it
effectively, you have to learn the basics. Many good free resources are
available, e.g. `A Whirlwind tour of Python`_, the `Python data science
handbook`_ and the `Astropy Hands-On Tutorial`_.

.. _tutorials_notebooks:

Introduction
------------

The following three tutorials show different ways of how to use Gammapy to perform a complete data analysis,
from data selection to data reduction and finally modeling and fitting.

The first tutorial is an overview on how to perform a standard analysis workflow using the high level interface
in a configuration-driven approach, whilst the second deals with the same use-case using the low level API
and showing what is happening *under-the-hood*. The third tutorial shows a glimpse of how to handle different
basic data structures like event lists, source catalogs, sky maps, spectral models and flux points tables.

.. nbgallery::

   starting/analysis_1.ipynb
   starting/analysis_2.ipynb
   starting/overview.ipynb

Data exploration
----------------

These three tutorials show how to perform data exploration with Gammapy, providing an introduction to the CTA,
H.E.S.S. and Fermi-LAT data and instrument response functions (IRFs). You will be able to explore and filter
event lists according to different criteria, as well as to get a quick look of the multidimensional IRFs files.

.. nbgallery::

   data/cta.ipynb
   data/hess.ipynb
   data/fermi_lat.ipynb


Data analysis
-------------

The following set of tutorials are devoted to data analysis, and grouped according to the specific covered use
cases in spectral analysis and flux fitting, image and cube analysis modelling and fitting, as well as
time-dependent analysis with light-curves.

1D Spectral
~~~~~~~~~~~

.. nbgallery::

   analysis/1D/spectral_analysis.ipynb
   analysis/1D/spectral_analysis_rad_max.ipynb
   analysis/1D/sed_fitting.ipynb
   analysis/1D/extended_source_spectral_analysis.ipynb
   analysis/1D/spectrum_simulation.ipynb
   analysis/1D/cta_sensitivity.ipynb

2D Image
~~~~~~~~

.. nbgallery::

   analysis/2D/ring_background.ipynb
   analysis/2D/modeling_2D.ipynb
   analysis/2D/detect.ipynb


3D Cube
~~~~~~~

.. nbgallery::

   analysis/3D/cta_data_analysis.ipynb
   analysis/3D/analysis_3d.ipynb
   analysis/3D/simulate_3d.ipynb
   analysis/3D/analysis_mwl.ipynb
   analysis/3D/event_sampling.ipynb
   analysis/3D/flux_profiles.ipynb

Time
~~~~

.. nbgallery::

   analysis/time/light_curve_simulation.ipynb
   analysis/time/light_curve.ipynb
   analysis/time/light_curve_flare.ipynb
   analysis/time/pulsar_analysis.ipynb


Package / API
-------------

The following tutorials demonstrate different dimensions of the Gammapy API or
expose how to perform more specific use cases.

.. nbgallery::

   api/makers.ipynb
   api/catalog.ipynb
   api/models.ipynb
   api/model_management.ipynb
   api/fitting.ipynb
   api/maps.ipynb
   api/mask_maps.ipynb
   api/astro_dark_matter.ipynb
   api/datasets.ipynb


.. _tutorials_scripts:

Scripts
-------

For interactive use, IPython and Jupyter are great, and most Gammapy examples use those.
However, for long-running, non-interactive tasks like data reduction or survey maps,
you might prefer a Python script.

The following example shows how to run Gammapy within a Python script.

.. toctree::

   survey_map
