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

Lightcurve
==========

Gammapy uses a simple container for light curves: the `~gammapy.time.LightCurve` class. It stores
the light curve in the form of a `~astropy.table.Table` and provides a few convenience methods,
to create time objects and plots.

The table structure follows the approach proposed in the gamma-ray-astro-formats_ webpage.

.. _gamma-ray-astro-formats: https://gamma-astro-data-formats.readthedocs.io/en/latest/lightcurves/index.html

The following example shows how to read a table that contains a lightcurve and then create a ``LightCurve`` object.
The latter gives access to a number of utilities such as plots and access to times as `~astropy.time.Time` objects::

    >>> from astropy.table import Table
    >>> url = 'https://github.com/gammapy/gamma-cat/raw/master/input/data/2006/2006A%2526A...460..743A/tev-000119-lc.ecsv'
    >>> table = Table.read(url, format='ascii.ecsv')
    >>> from gammapy.time import LightCurve
    >>> lc = LightCurve(table)
    >>> lc.time[:2].iso
    ['2004-05-23 01:47:08.160' '2004-05-23 02:17:31.200']
    >>> lc.plot()


Light Curve Extraction
======================

The extraction of a light curve from gamma-ray data follows the general approach of
data reduction and modeling/fitting. Observations are first reduced to dataset objects
(e.g. `~gammapy.cube.MapDataset` or `~gammapy.spectrum.SpectrumDatasetOnOff`). Then, after
setting the appropriate model the flux is extracted in each time bin with the
`~gammapy.time.LightCurveEstimator`.

To extract the light curve of a source, the `~gammapy.time.LightCurveEstimator`
fits a scale factor on the model component representing the source in each time bin
and returns a `~gammapy.time.LightCurve`. It can work with spectral (1D) datasets as well
as with map (3D) datasets.

Once a `~gammapy.Fit.Datasets` object is build with a model set, one can call the estimator
to compute the light curve in the datasets time intervals::

    >>> lc_estimator = LightCurveEstimator(datasets, source="source")
    >>> lc = lc_estimator.run(e_min=1*u.TeV, emax=10*u.TeV, e_ref=1*u.TeV)

where `source` is the model component describing the source of interest and `datasets` the `~gammapy.modeling.Datasets`
object produced by data reduction.
The light curve notebook shows an example of `observation based light curve
extraction <../notebooks/light_curve.html#Light-Curve-estimation:-by-observation>`__

Similarly, `~gammapy.time.LightCurveEstimator` can be used to extract the light curve in user defined time intervals.
This can be useful to combine datasets to produce light curve by night, week or month::

    >>> lc_estimator = LightCurveEstimator(datasets, source="source", time_intervals=time_intervals)
    >>> lc = lc_estimator.run(e_min=1*u.TeV, emax=10*u.TeV, e_ref=1*u.TeV)

where `time_intervals` is a list of time intervals as `~astropy.time.Time` objects.
The light curve notebook shows an example of `night-wise light curve
extraction <../notebooks/light_curve.html#Night-wise-LC-estimation>`__


Variability and periodicity tests
=================================

A few utility functions to perform timing tests are available in `~gammapy.time`.

`~gammapy.time.compute_chisq` performs a chisquare test for variable source flux::

     >>> from gammapy.time import chisquare
     >>> print(compute_chisq(lc['FLUX']))

`~gammapy.time.compute_fvar` calculates the fractional variance excess::

     >>> from gammapy.time import fvar
     >>> print(compute_fvar(lc['FLUX'], lc['FLUX_ERR']))

`~gammapy.time` also provides methods for period detection in time series, i.e. light
curves of :math:`\gamma`-ray sources.  `~gammapy.time.robust_periodogram` performs a
periodogram analysis where the unevenly sampled time series is contaminated by outliers,
i.e. due to the source's high states. This is demonstrated on the :ref:`period` page.

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
