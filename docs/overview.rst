.. include:: references.txt

.. _overview:

Overview
========

This page gives an overview of the main concepts in Gammapy. It is a theoretical
introduction to Gammapy, explaining what data, packages, classes and methods are
involved in a data analysis with Gammapy, proviging many links to other
documentation pages and tutorials, but not giving code examples here.

For a hands-on introduction how to use Gammapy, see :ref:`install`,
:ref:`getting-started` and the many :ref:`tutorials`. Specifically the `Overview
tutorial notebook <notebooks/overview.html>`__ can be seen as an example-based
Gammapy overview that parallels this more theoretical description.

Introduction
------------

Gammapy is an open-source Python package for gamma-ray astronomy built on Numpy
and Astropy. It is a prototype for the Cherenkov Telescope Array (CTA) science
tools, and can also be used to analyse data from existing gamma-ray telescopes
if their data is available in the standard FITS format (gadf_).

To use Gammapy you need a basic knowledge of Python, Numpy, Astropy, as well as
matplotlib for plotting. Many standard gamma-ray analyses can be done with few
lines of configuration and code, so you can get pretty far by copy and pasting
and adapting the working examples from the Gammapy documentation. But
eventually, if you want to script more complex analyses, or inspect analysis
results or intermediate analysis products, you need to acquire a basic to
intermediate Python skill level. See :ref:`tutorials_basics` for links to
excellent tutorials on Python, Numpy and Astropy.


Gammapy workflow
----------------

- From `gammapy.data.Datastore` to `gammapy.data.Observations` to `gammapy.modeling.Datasets` to `gammapy.modeling.Fit`
- How `gammapy.analysis.Analysis` orchestrates this.

DL3 data format
---------------

- `gammapy.data.EventList` (link to relevant doc)
- `gammapy.irf` (link to relevant doc)

TODO: mention Fermi

To learn more, see :ref:`gammapy.data <data>` and :ref:`gammapy.irf <irf>`.

Data reduction
--------------

- Producing reduced data to prepare for modeling and fitting.
- Rapid description of the typical data reduction chains (possibly link to some notebooks)

Datasets
--------

Datasets in Gammapy contain reduced data, models, and the likelihood function
fit statistic for a given set of model parameters. All datasets contain a
`~gammapy.modeling.models.SkyModels` container with one or more
`~gammapy.modeling.models.SkyModel` objects that represent additive emission
components.

To model and fit data in Gammapy, you have to create a
`gammapy.modeling.Datasets` container object with one or multiple
`gammapy.modeling.Dataset` objects. Gammapy has built-in support to create and
analyse the following datasets:

- `gammapy.cube.MapDataset`
- `gammapy.cube.MapDatasetOnOff`
- `gammapy.spectrum.SpectrumDataset`
- `gammapy.spectrum.SpectrumDatasetOnOff`
- `gammapy.spectrum.FluxPointsDataset`

The map datasets represent 3D cubes (`gammpy.maps.WcsNDMap` objects) with two
spatial and one energy axis. For 2D images the same map objects and map datasets
are used, an energy axis is present but only has one energy bin. The
`~gammapy.cube.MapDataset` contains a counzts map, background is modeled with a
`~gammapy.modeling.models.BackgroundModel`, and the fit statistic used is
``cash``. The `~gammapy.cube.MapDatasetOnOff` contains on and off count maps,
background is implicitly modeled via the off counts map, and the ``wstat`` fit
statistic.

The spectrum datasets represent 1D spectra (`gammapy.spectrum.CountsSpectrum`
objects) with an energy axis. There are no spatial axes or information, the 1D
spectra are obtained for a given on region. The
`~gammapy.spectrum.SpectrumDataset` contains a counts spectrum, background is
modeled with a `~gammapy.spectrum.CountsSpectrum` (TODO: why not a background
model like for maps?), and the fit statistic used is ``cash``. The
`~gammapy.spectrum.SpectrumDatasetOnOff` contains on on and off count spectra,
background is implicitly modeled via the off counts spectrum, and the ``wstat``
fit statistic.

The `gammapy.spectrum.FluxPointsDataset` contains `gammapy.spectrum.FluxPoints`
and a spectral model, the fit statistic used is ``chi2``.

To learn more about datasets, see :ref:`gammapy.cube <cube>`,
:ref:`gammapy.spectrum <spectrum>`, :ref:`gammapy.modeling <modeling>`.

Modeling and Fitting
--------------------

See `gammapy.modeling` and `gammapy.modeling.models`.

- Forward-folding and Likelihood-based analyses (link to stats)
- Models (link to model.ipynb)
- Typical operations with links to notebooks

Time analysis
-------------

Light curves are represented as `gammapy.time.LightCurve` objects, a wrapper
class around `astropy.table.Table`. To compute light curves, use the
`gammapy.time.LightCurveEstimator`.

To learn more about time, see :ref:`gammapy.time <time>`.

Simulation
----------

Gammapy currently supports binned simulation, Poisson fluctuation of predicted
counts maps. The following tutorials illustrate how to use that to predict
observability, significance and sensitivity, using CTA examples:

- `3D map simulation <notebooks/simulate_3d.html>`__
- `1D spectrum simulation <notebooks/spectrum_simulation.html>`__
- `Point source sensitivity <notebooks/cta_sensitivity.html>`__

Development of event sampling is work in progress, coming soon.

Other
-----

Parts of Gammapy not mentioned in this overview yet:

- :ref:`gammapy.data <data>`
- :ref:`gammapy.irf <irf>`
- :ref:`gammapy.maps <maps>`
- :ref:`gammapy.catalog <catalog>`
- :ref:`gammapy.astro <astro>`
- :ref:`gammapy.stats <stats>`
- :ref:`gammapy.detect <detect>`
- :ref:`gammapy.scripts <CLI>` (``gammapy`` command line tool)

