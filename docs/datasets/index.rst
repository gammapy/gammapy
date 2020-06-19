.. include:: ../references.txt

.. _datasets:

***************************
datasets - Reduced datasets
***************************

.. currentmodule:: gammapy.datasets

Introduction
============

The `gammapy.datasets` sub-package contains classes to handle reduced
gamma-ray data for modeling and fitting.

The `Dataset` class bundles reduced data, IRFs and model to perform
likelihood fitting and joint-likelihood fitting.
All datasets contain a `~gammapy.modeling.models.Models` container with one or more
`~gammapy.modeling.models.SkyModel` objects that represent additive emission
components.

To model and fit data in Gammapy, you have to create a
`~gammapy.datasets.Datasets` container object with one or multiple
`~gammapy.datasets.Dataset` objects. Gammapy has built-in support to create and
analyse the following datasets: `~gammapy.datasets.MapDataset`,
`~gammapy.datasets.MapDatasetOnOff`, `~gammapy.datasets.SpectrumDataset`,
`~gammapy.datasets.SpectrumDatasetOnOff` and
`~gammapy.datasets.FluxPointsDataset`.

The map datasets represent 3D cubes (`~gammapy.maps.WcsNDMap` objects) with two
spatial and one energy axis. For 2D images the same map objects and map datasets
are used, an energy axis is present but only has one energy bin. The
`~gammapy.datasets.MapDataset` contains a counzts map, background is modeled with a
`~gammapy.modeling.models.BackgroundModel`, and the fit statistic used is
``cash``. The `~gammapy.datasets.MapDatasetOnOff` contains on and off count maps,
background is implicitly modeled via the off counts map, and the ``wstat`` fit
statistic.

The spectrum datasets represent 1D spectra (`~gammapy.maps.RegionNDMap`
objects) with an energy axis. There are no spatial axes or information, the 1D
spectra are obtained for a given on region. The
`~gammapy.datasets.SpectrumDataset` contains a counts spectrum, background is
modeled with a `~gammapy.maps.RegionNDMap`, and the fit statistic used is
``cash``. The `~gammapy.datasets.SpectrumDatasetOnOff` contains on on and off
count spectra, background is implicitly modeled via the off counts spectrum, and
the ``wstat`` fit statistic. The `~gammapy.datasets.FluxPointsDataset` contains
`~gammapy.estimatorsFluxPoints` and a spectral model, the fit statistic used is
``chi2``.

Getting Started
===============



Reference/API
=============

.. automodapi:: gammapy.datasets
    :no-inheritance-diagram:
    :include-all-objects:
