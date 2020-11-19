.. include:: references.txt

.. _howto:

How To
======

This page contains short "how to" or "frequently asked question" entries for
Gammapy. Each entry is for a very specific task, with a short answer, and links
to examples and documentation.

If you're new to Gammapy, please read the :ref:`overview` and have a look at the
list of :ref:`tutorials`. The information below is in addition to those pages,
it's not a complete list of how to do everything in Gammapy.

Please give feedback and suggest additions to this page!

Access IACT data
++++++++++++++++

To access IACT data use the `~gammapy.data.DataStore`. You can see how to create
one with the high level interface `~gammapy.analysis.Analysis` `here
<tutorials/analysis_1.html#Setting-the-data-to-use>`__. You can also create it
directly, see `here
<tutorials/analysis_2.html#Defining-the-datastore-and-selecting-observations>`__.

Check IRFs
++++++++++

Gammapy offers a number of methods to explore the content of the various IRFs
contained in an observation. This is usually done thanks to their ``peek()``
methods. See example for CTA `here <tutorials/cta.html#IRFs>`__ and for H.E.S.S.
`here <tutorials/hess.html#DL3-DR1>`__.

Use gammapy for modeling 2D images
++++++++++++++++++++++++++++++++++

Gammapy treats 2D maps as 3D cubes with one bin in energy. To see an example of the relevant data reduction, see
`2-dim sky image analysis <tutorials#core-tutorials>`

Sometimes, you might want to use previously obtained images lacking an energy axis
(eg: reduced using traditional IACT tools) for modeling and fitting inside gammapy.
In this case, it is necessary to attach an `energy` axis on  


Extract 1D spectra
++++++++++++++++++

The `~gammapy.analysis.Analysis` class can perform spectral extraction. The
`~gammapy.analysis.AnalysisConfig` must be defined to produce '1d' datasets.
Alternatively, you can follow the `spectrum extraction notebook
<tutorials/spectrum_analysis.html>`__.

Extract a lightcurve
++++++++++++++++++++

The `Light curve estimation <tutorials/light_curve.html>`__ tutorial shows how
to extract a run-wise lightcurve.

To perform an analysis in a time range smaller than that of an observation, it
is necessary to filter the latter with its `select_time` method. This produces
an new observation containing events in the specified time range. With the new
`~gammapy.data.Observations` it is then possible to perform the usual data
reduction which will produce datasets in the correct time range. The light curve
extraction can then be performed as usual with the
`~gammapy.estimators.LightCurveEstimator`. This is demonstrated in the `Light curve -
Flare <tutorials/light_curve_flare.html>`__ tutorial.

Compute source significance
+++++++++++++++++++++++++++

Estimate the significance of a source, or more generally of an additional model
component (such as e.g. a spectral line on top of a power-law spectrum), is done
via a hypothesis test. You fit two models, with and without the extra source or
component, then use the test statistic values from both fits to compute the
significance or p-value. To obtain the test statistic, call
`~gammapy.modeling.Dataset.stat_sum` for the model corresponding to your two
hypotheses (or take this value from the print output when running the fit), and
take the difference. Note that in Gammapy, the fit statistic is defined as ``S =
- 2 * log(L)`` for likelihood ``L``, such that ``TS = S_0 - S_1``. See
:ref:`overview_datasets` for an overview of fit statistics used.

Compute cumulative significance
+++++++++++++++++++++++++++++++

A classical plot in gamma-ray astronomy is the cumulative significance of a
source as a function of observing time. In Gammapy, you can produce it with 1D
(spectral) analysis. Once datasets are produced for a given ON region, you can
access the total statistics with the ``info_table(cumulative=True)`` method of
`~gammapy.modeling.Datasets`. See example `here
<tutorials/spectrum_analysis.html#Source-statistic>`__.

Detect sources in a map
+++++++++++++++++++++++

Gammapy provides methods to perform source detection in a 2D map. First step is
to produce a significance map, i.e. a map giving the probability that the flux
measured at each position is a background fluctuation. For a
`~gammapy.datasets.MapDataset`, the class `~gammapy.estimators.TSMapEstimator` can be
used. A simple correlated Li & Ma significance can be used, in particular for
ON-OFF datasets. The second step consists in applying a peak finer algorithm,
such as `~gammapy.estimators.utils.find_peaks`. This is demonstrated in the `Source
detection tutorial <tutorials/detect.html>`__.

Astrophysical source modeling
+++++++++++++++++++++++++++++

It is possible to combine Gammapy with astrophysical modeling codes, if they
provide a Python interface. Usually this requires some glue code to be written,
e.g. `~gammapy.modeling.models.NaimaSpectralModel` is an example of a Gammapy
wrapper class around the Naima spectral model and radiation classes, which then
allows modeling and fitting of Naima models within Gammapy (e.g. using CTA,
H.E.S.S. or Fermi-LAT data).

Implement a custom model
++++++++++++++++++++++++
Gammapy allows the flexibility of using user-defined models for analysis.
For an example, see ` Implementing a Custom Model
<tutorials/models.html#Implementing-a-Custom-Model>`__.

Energy Dependent Spatial Models
+++++++++++++++++++++++++++++++
While Gammapy does not ship energy dependent spatial models, it is possible to define
such models within the modeling framework.
For an example, see ` here
<tutorials/models.html#Models-with-Energy-dependent-morphologyl>`__.
