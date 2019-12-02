.. include:: references.txt

.. _howto:

How To
======

Data access and manipulation
----------------------------

How to open a data store and access observations?
+++++++++++++++++++++++++++++++++++++++++++++++++

Gammapy accesses data through a `~gammapy.data.DataStore` object. You can see how to create one with the high
level interface `~gammapy.analysis.Analysis` `here <notebooks/analysis_1.html#Setting-the-data-to-use>`__.
You can also create the object directly, see
this `example <notebooks/analysis_2.html#Defining-the-datastore-and-selecting-observations>`__.

..  How to select observations?
    +++++++++++++++++++++++++++

..  How to filter selected observations?
    ++++++++++++++++++++++++++++++++++++

How to explore the IRFs of an observation?
++++++++++++++++++++++++++++++++++++++++++

Gammapy offers a number of methods to explore the content of the various IRFs contained in an observation.
This is usually done thanks to their `peek()` methods.

You can find `here <notebooks/cta.html#IRFs>`__ an example using the CTA 1DC dataset and
`here <notebooks/hess.html#DL3-DR1>`__ an example using the H.E.S.S. DL3 DR1 data.

Data reduction: spectra
-----------------------

How to extract 1D spectra?
++++++++++++++++++++++++++

The `~gammapy.analysis.Analysis` class can perform spectral extraction. The `~gammapy.analysis.AnalysisConfig`
must be defined to produce '1d' datasets.

Alternatively, you can follow the `spectrum extraction notebook <notebooks/spectrum_analysis.html>`__.

How to compute the cumulative significance of a source?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

A classical plot in gamma-ray astronomy is the cumulative significance of a source as a function
of observing time. In Gammapy, you can produce it with 1D (spectral) analysis. Once datasets are produced
for a given ON region, you can access the total statistics with the `info_table(cumulative=True)` method
of `~gammapy.modeling.Datasets`.

You can find an example usage `here <notebooks/spectrum_analysis.html#Source-statistic>`__.

Data reduction: maps
--------------------

..  How to build maps?
    ++++++++++++++++++

..  How to plot a excess map?
    +++++++++++++++++++++++++

..  How to overlay significance and excess on maps?
    +++++++++++++++++++++++++++++++++++++++++++++++

How to detect sources in a map?
+++++++++++++++++++++++++++++++

Gammapy provides methods to perform source detection in a 2D map. First step is to produce
a significance map, i.e. a map giving the probability that the flux measured at each position
is a background fluctuation. For a `~gammapy.cube.MapDataset`, the class `~gammapy.detect.TSMapEstimator`
can be used. A simple correlated Li & Ma significance can be used, in particular for ON-OFF datasets.
The second step consists in applying a peak finer algorithm, such as `~gammapy.detect.find_peaks`.

This is demonstrated in the `detect notebook <notebooks/detect.html>`__

How to compute the significance of a source?
++++++++++++++++++++++++++++++++++++++++++++

Estimate the significance of a source, or more generally of an additional model component
(such as e.g. a spectral line on top of a power-law spectrum), is done via a hypothesis test.
You fit two models, with and without the extra source or component, then use the test statistic
values from both fits to compute the significance or p-value.

..  TODO: link to notebook
    TODO: update this entry once https://github.com/gammapy/gammapy/issues/2149
    and https://github.com/gammapy/gammapy/issues/1540 are resolved, linking to the documentation
    developed there.


Modeling and fitting
--------------------

..  How to share a model between two datasets?
    ++++++++++++++++++++++++++++++++++++++++++

How to use Gammapy with astrophysical modeling packages?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

It is possible to combine Gammapy with astrophysical modeling codes, if they provide a Python interface.
Usually this requires some glue code to be written, e.g. `~gammapy.modeling.models.NaimaSpectralModel` is
an example of a Gammapy wrapper class around the Naima spectral model and radiation classes, which then
allows modeling and fitting of Naima models within Gammapy (e.g. using CTA, H.E.S.S. or Fermi-LAT data).


.. How to add a user defined model?
    ++++++++++++++++++++++++++++++++
    **TODO: move content from spectrum_simulation.ipynb**

How to extract a lightcurve?
++++++++++++++++++++++++++++

This is demonstrated in the following `notebook <notebooks/Light_curve.html>`__.

How to create a light curve with time intervals shorter than observations?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To perform an analysis in a time range smaller than that of an observation, it is necessary to filter
the latter with its `select_time` method. This produces an new observation containing events in the
specified time range.

With the new `~gammapy.data.Observations` it is then possible to perform the usual data reduction
which will produce datasets in the correct time range. The light curve extraction can then be performed
as usual with the `~gammapy.time.LightCurveEstimator`.

This is demonstrated in the following `notebook <notebooks/Light_curve_flare.html>`__. In particular,
the time selection is performed `here <notebooks/Light_curve_flare.html#Filter-the-observations-list-in-time-intervals>`__



Other Ideas
-----------

If you think an entry is missing, please send us a request to add it.