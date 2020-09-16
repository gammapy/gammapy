.. include:: ../references.txt

.. _lightcurve:

***********
Lightcurves
***********

.. currentmodule:: gammapy.estimators


Lightcurve
==========

Gammapy uses a simple container for light curves: the `~gammapy.estimators.LightCurve` class. It stores
the light curve in the form of a `~astropy.table.Table` and provides a few convenience methods,
to create time objects and plots.

The table structure follows the approach proposed in the gamma-ray-astro-formats_ webpage.

.. _gamma-ray-astro-formats: https://gamma-astro-data-formats.readthedocs.io/en/latest/lightcurves/index.html

The following example shows how to read a table that contains a lightcurve and then create a ``LightCurve`` object.
The latter gives access to a number of utilities such as plots and access to times as `~astropy.time.Time` objects::

    >>> from astropy.table import Table
    >>> url = 'https://github.com/gammapy/gamma-cat/raw/master/input/data/2006/2006A%2526A...460..743A/tev-000119-lc.ecsv'
    >>> table = Table.read(url, format='ascii.ecsv')
    >>> from gammapy.estimators import LightCurve
    >>> lc = LightCurve(table)
    >>> lc.time[:2].iso
    ['2004-05-23 01:47:08.160' '2004-05-23 02:17:31.200']
    >>> lc.plot()


Light Curve Extraction
======================

The extraction of a light curve from gamma-ray data follows the general approach of
data reduction and modeling/fitting. Observations are first reduced to dataset objects
(e.g. `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDatasetOnOff`). Then, after
setting the appropriate model the flux is extracted in each time bin with the
`~gammapy.estimators.LightCurveEstimator`.

To extract the light curve of a source, the `~gammapy.estimators.LightCurveEstimator`
fits a scale factor on the model component representing the source in each time bin
and returns a `~gammapy.estimators.LightCurve`. It can work with spectral (1D) datasets as well
as with map (3D) datasets.

Once a `~gammapy.datasets.Datasets` object is build with a model set, one can call the estimator
to compute the light curve in the datasets time intervals::

    >>> lc_estimator = LightCurveEstimator(datasets, source="source")
    >>> lc = lc_estimator.run(e_min=1*u.TeV, emax=10*u.TeV, e_ref=1*u.TeV)

where `source` is the model component describing the source of interest and `datasets` the `~gammapy.modeling.Datasets`
object produced by data reduction.
The light curve notebook shows an example of `observation based light curve
extraction <../tutorials/light_curve.html#Light-Curve-estimation:-by-observation>`__

Similarly, `~gammapy.estimators.LightCurveEstimator` can be used to extract the light curve in user defined time intervals.
This can be useful to combine datasets to produce light curve by night, week or month::

    >>> lc_estimator = LightCurveEstimator(datasets, source="source", time_intervals=time_intervals)
    >>> lc = lc_estimator.run(e_min=1*u.TeV, emax=10*u.TeV, e_ref=1*u.TeV)

where `time_intervals` is a list of time intervals as `~astropy.time.Time` objects.
The light curve notebook shows an example of `night-wise light curve
extraction <../tutorials/light_curve.html#Night-wise-LC-estimation>`__


Tutorials
=========

The main tutorial demonstrates how to extract light curves from 1D and 3D datasets:

* `Light Curve tutorial <../tutorials/light_curve.html>`__

Light curve extraction on small time bins (i.e. smaller than the observation scale) for flares
is demonstrated in the following tutorial:

* `Flare tutorial <../tutorials/light_curve_flare.html>`__