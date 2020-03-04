.. include:: ../references.txt

.. _lightcurve:

***********
Lightcurves
***********

.. currentmodule:: gammapy.estimators


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


Tutorials
=========

The main tutorial demonstrates how to extract light curves from 1D and 3D datasets:

* `Light Curve tutorial <../notebooks/light_curve.html>`__

Light curve extraction on small time bins (i.e. smaller than the observation scale) for flares
is demonstrated in the following tutorial:

* `Flare tutorial <../notebooks/light_curve_flare.html>`__