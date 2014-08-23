.. _image-plotting:

Image plotting
==============

`gammapy.image` provides a few helper functions and classes to create
publication-quality images.

Colormaps
---------

The following example shows how to plot images using colormaps that are commonly
used in gamma-ray astronomy (`~gammapy.image.colormap_hess` and `~gammapy.image.colormap_milagro`).

.. plot:: image/colormap_example.py

Multi-panel Galactic plane survey image plots
---------------------------------------------

The following example shows how to plot a very wide Galactic plane survey image
by splitting it into multiple panels using the `~gammapy.image.GalacticPlaneSurveyPanelPlot` class.

.. plot:: image/survey_example.py
