.. _time:

********************
time - Time analysis
********************

.. currentmodule:: gammapy.time

Introduction
============

`gammapy.time` contains classes and methods for time-based analysis, e.g. for AGN, binaries
or pulsars studies. The main classes are `~gammapy.time.LightCurve`, which is a container for
light curves, and `~gammapy.time.LightCurveEstimator`, which extracts a light curve from a list
of datasets. A number of functions to test for variability and periodicity are available in
`~gammapy.time.variability` and `~gammapy.time.periodicity`. Finally, `gammapy.utils.time`
contains low-level helper functions for time conversions.

Getting Started
===============

.. _time-lc:

Lightcurve
----------

Gammapy uses a simple container for light curves: the `~gammapy.time.LightCurve` class. It stores
the light curve in the form of a `~astropy.table.Table` and provides a few convenience methods,
to create time objects and plots.

The table structure follows the approach proposed in the gamma-ray-astro-formats_ webpage.

.. _gamma-ray-astro-formats: https://gamma-astro-data-formats.readthedocs.io/en/latest/lightcurves/index.html

First, read a table that contains a lightcurve::

    >>> from astropy.table import Table
    >>> url = 'https://github.com/gammapy/gamma-cat/raw/master/input/data/2006/2006A%2526A...460..743A/tev-000119-lc.ecsv'
    >>> table = Table.read(url, format='ascii.ecsv')

Then, create a ``LightCurve`` object::

    >>> from gammapy.time import LightCurve
    >>> lc = LightCurve(table)

``LightCurve`` gives access to times as `~astropy.time.Time` objects::

    >>> lc.time[:2].iso
    ['2004-05-23 01:47:08.160' '2004-05-23 02:17:31.200']
    >>> lc.plot()

.. _time-lc-estimator:

Light Curve Extraction
----------------------

The extraction of a light curve from gamma-ray data follows the general approach of
data reduction and modeling/fitting. Observations are first reduced to dataset objects
(e.g. `~gammapy.cube.MapDataset` or `~gammapy.spectrum.SpectrumDatasetOnOff`). Then, after
setting the appropriate model the flux is extracted in each time bin with the
`~gammapy.time.LightCurveEstimator`.

To extract the light curve of a source, the `~gammapy.time.LightCurveEstimator`
fits a scale factor on the model component representing the source in each time bin
and returns a `~gammapy.time.LightCurve`. It can work with spectral (1D) datasets as well
as with map (3D) datasets.



.. _time-variability:

Variability and peridocity test
-------------------------------

TODO: Add some rapid discussion of chisquare and fractional variance functions

Tutorials
=========

The main tutorial demonstrates how to extract light curves from 1D and 3D datasets:

* `Light Curve tutorial <../notebooks/light_curve.html>`__

Light curve extraction on small time bins (i.e. smaller than the observation scale) for flares
is demonstrated in the following tutorial:

* `Flare tutorial <../notebooks/light_curve_flare.html>`__

Using `gammapy.time`
====================

.. toctree::
   :maxdepth: 1

   period

Reference/API
=============

.. automodapi:: gammapy.time
    :no-inheritance-diagram:
    :include-all-objects:
