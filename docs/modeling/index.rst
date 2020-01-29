.. include:: ../references.txt

.. _modeling:

*****************************
modeling - Models and fitting
*****************************

.. currentmodule:: gammapy.modeling

Introduction
============

`gammapy.modeling` contains all the functionality related to modeling and fitting
data. This includes spectral, spatial and temporal model classes, as well as the fit
and parameter API. An overview of all the available models can be found in the :ref:`model-gallery`.
In general the models are grouped into the following categories:

- `~gammapy.modeling.models.SpectralModel`: models to describe spectral shapes of sources
- `~gammapy.modeling.models.SpatialModel`: models to describe spatial shapes (morphologies) of sources
- `~gammapy.modeling.models.TemporalModel`: models to describe temporal flux evolution of sources, such as light and phase curves
- `~gammapy.modeling.models.SkyModel` and `~gammapy.modeling.models.SkyDiffuseCube`: model to combine the spectral and spatial model components

The models follow a naming scheme which contains the category as a suffix to the class
name.


Tutorials
=========


:ref:`tutorials` that show examples using ``gammapy.modeling``:

- `Models Tutorial <../notebooks/models.html>`__
- `Modeling and Fitting <../notebooks/modeling.html>`__
- `analysis_3d.html <../notebooks/analysis_3d.html>`__
- `spectrum_analysis.html <../notebooks/spectrum_analysis.html>`__


Reference/API
=============

.. automodapi:: gammapy.modeling
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.modeling.models
    :no-inheritance-diagram:
    :include-all-objects:
