.. include:: references.txt

.. _howto:

How To
======

Data access and manipulation
----------------------------

How to open a data store and access observations?
+++++++++++++++++++++++++++++++++++++++++++++++++

Gammapy accesses data through a `~gammapy.data.DataStore` object. You can see how to create one with the high
level interface `~gammapy.analysis.Analysis` `here <../notebooks/Analysis_1.html#Setting-the-data-to-use>`__.
You can also create the object directly, see
this `example <../notebooks/Analysis_2.html#Defining-the-datastore-and-selecting-observations>`__.

How to select observations?
+++++++++++++++++++++++++++

How to filter selected observations?
++++++++++++++++++++++++++++++++++++

How to explore the IRFs of an observation?
++++++++++++++++++++++++++++++++++++++++++

Data reduction: spectra
-----------------------

How to extract 1D spectra?
++++++++++++++++++++++++++

The `~gammapy.analysis.Analysis` class can perform spectral extraction. The `~gammapy.analysis.AnalysisConfig`
must be defined to produce '1d' datasets.

Alternatively, you can follow the `spectrum extraction notebook <../notebooks/Spectrum_analysis.html>`__.

How to compute the cumulative significance of a source?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

A classical plot in gamma-ray astronomy is the cumulative significance of a source as a function
of observing time. In Gammapy, you can produce it with 1D (spectral) analysis. Once datasets are produced
for a given ON region, you can access the total statistics with the `~gammapy.modeling.Datasets.info_table()`
method.

You can find an example usage `here <../notebooks/Spectrum_analysis.html#Source-statistic>`__.

Data reduction: maps
--------------------

How to build maps?
++++++++++++++++++

How to plot a excess map?
+++++++++++++++++++++++++

How to overlay significance and excess on maps?
+++++++++++++++++++++++++++++++++++++++++++++++

How to detect sources in a map?
+++++++++++++++++++++++++++++++

Short explanation and link to detect.ipynb

How to compute the significance of a source?
++++++++++++++++++++++++++++++++++++++++++++

Estimate the significance of a source, or more generally of an additional model component
(such as e.g. a spectral line on top of a power-law spectrum), is done via a hypothesis test.
You fit two models, with and without the extra source or component, then use the test statistic
values from both fits to compute the significance or p-value.

**TODO: link to notebook**
TODO: update this entry once https://github.com/gammapy/gammapy/issues/2149
and https://github.com/gammapy/gammapy/issues/1540 are resolved, linking to the documentation
developed there.


Modeling and fitting
--------------------

How to share a model between two datasets?
++++++++++++++++++++++++++++++++++++++++++

How to use Gammapy with astrophysical modeling packages?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

It is possible to combine Gammapy with astrophysical modeling codes, if they provide a Python interface.
Usually this requires some glue code to be written, e.g. `~gammapy.modeling.models.NaimaSpectralModel` is
an example of a Gammapy wrapper class around the Naima spectral model and radiation classes, which then
allows modeling and fitting of Naima models within Gammapy (e.g. using CTA, H.E.S.S. or Fermi-LAT data).

**TODO: give link to example in a notebook.**

How to add a user defined model?
++++++++++++++++++++++++++++++++

**TODO: move content from spectrum_simulation.ipynb**

How to extract a lightcurve?
++++++++++++++++++++++++++++

Link to relevant notebook.

How to create a light curve with time intervals shorter than observations?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Link to relevant notebook.

Do we really want those:

How to test for variability?
++++++++++++++++++++++++++++

How to test for periodicity in a light curve?
+++++++++++++++++++++++++++++++++++++++++++++



Other Ideas
-----------

If you think an entry is missing, please send us a request to add it.