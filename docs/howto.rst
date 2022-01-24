.. include:: references.txt

.. _howto:

How-to
======

This page contains short "how to" or "frequently asked question" entries for
Gammapy. Each entry is for a very specific task, with a short answer, and links
to examples and documentation.

If you're new to Gammapy, please read the :ref:`overview` and have a look at the
list of :ref:`tutorials`. The information below is in addition to those pages,
it's not a complete list of how to do everything in Gammapy.

Please give feedback and suggest additions to this page!


Spell and pronounce Gammapy
+++++++++++++++++++++++++++
The recommended spelling is "Gammapy" as proper name. The recommended
pronunciation is [ɡæməpaɪ] where the syllable "py" is pronounced like
the english word "pie". You can listen to it `here <http://ipa-reader.xyz/?text=ˈ%C9%A1æməpaɪ&voice=Amy>`__.


Access IACT DL3 data
++++++++++++++++++++

To access IACT data in the DL3 format, use the `~gammapy.data.DataStore`. It allows
easy access to observations stored in the DL3 data library. You can create it directly
as in `this example<tutorials/data/hess.html#Datastore>`__.
It is also internally used by the high level interface `~gammapy.analysis.Analysis`. You
can see how to properly set it `here
<tutorials/starting/analysis_1.html#Setting-the-data-to-use>`__.


Select observations
+++++++++++++++++++

The `~gammapy.data.DataStore` provides access to a summary table of all observations available.
It can be used to select observations with various criterion.
You can for instance apply a cone search as shown `here
<tutorials/starting/analysis_2.html#Defining-the-datastore-and-selecting-observations>`__.
You can also select observations based on other information available using the
`~gammapy.data.ObservationTable.select_observations` method.

Make a on-axis equivalent livetime map
++++++++++++++++++++++++++++++++++++++

IACT detection efficiency varies in the FoV. To have an estimate
of the effective exposure with respect to the on-axis one,
it can be useful to build an `on-axis equivalent lifetime map
<tutorials/data/hess.html#On-axis-equivalent-livetime>`__.


Check IRFs
++++++++++

Gammapy offers a number of methods to explore the content of the various IRFs
contained in an observation. This is usually done thanks to their ``peek()``
methods. See example for CTA `here <tutorials/data/cta.html#IRFs>`__ and for H.E.S.S.
`here <tutorials/data/hess.html#DL3-DR1>`__.

Model 2D images
+++++++++++++++

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
<tutorials/analysis/1D/spectrum_analysis.html>`__.

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
Flare <tutorials/time/light_curve_flare.html>`__ tutorial.

Choose units for plotting
+++++++++++++++++++++++++
Units for plotting are handled with a combination of `matplotlib` and `astropy.units`.
For most plotting methods Gammapy forwards additional keywords to the corresponding
matplotlib plot method, including the `xunits` and `yunits` keywords, which allows
you to define the x and y axis units using `astropy.units`. Here is a minimal example:

.. code::

    from gammapy.estimators import FluxPoints
    from astropy import units as u

    filename = "$GAMMAPY_DATA/hawc_crab/HAWC19_flux_points.fits"
    fp = FluxPoints.read(filename)
    fp.plot(sed_type="e2dnde", xunits=u.erg, yunits=u.Unit("erg cm-2 s-1"))

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
<tutorials/analysis/1D/spectrum_analysis.html#Source-statistic>`__.

Detect sources in a map
+++++++++++++++++++++++

Gammapy provides methods to perform source detection in a 2D map. First step is
to produce a significance map, i.e. a map giving the probability that the flux
measured at each position is a background fluctuation. For a
`~gammapy.datasets.MapDataset`, the class `~gammapy.estimators.TSMapEstimator` can be
used. A simple correlated Li & Ma significance can be used, in particular for
ON-OFF datasets. The second step consists in applying a peak finer algorithm,
such as `~gammapy.estimators.utils.find_peaks`. This is demonstrated in the `Source
detection tutorial <tutorials/analysis/2D/detect.html>`__.

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
For an example, see `Implementing a Custom Model
<tutorials/api/models.html#Implementing-a-Custom-Model>`__.

Implement an energy dependent spatial models
++++++++++++++++++++++++++++++++++++++++++++
While Gammapy does not ship energy dependent spatial models, it is possible to define
such models within the modeling framework.
For an example, see `here
<tutorials/api/models.html#Models-with-Energy-dependent-morphologyl>`__.

Reduce memory budget for large datasets
+++++++++++++++++++++++++++++++++++++++

When dealing with surveys and large sky regions, the amount of memory required might become
problematic, in particular because of the default settings of the IRF maps stored in the
`~gammapy.datasets.MapDataset` used for the data reduction. Several options can be used to reduce
the required memory:
- Reduce the spatial sampling of the `~gammapy.irf.PSFMap` and the `~gammapy.irf.EDispKernelMap`
using the `binsz_irf` argument of the `~gammapy.datasets.MapDataset.create` method. This will reduce
the accuracy of the IRF kernels used for model counts predictions.
- Change the default IRFMap axes, in particular the `rad_axis` argument of `~gammapy.datasets.MapDataset.create`
This axis is used to define the geometry of the `~gammapy.irf.PSFMap` and controls the distribution of error angles
used to sample the PSF. This will reduce the quality of the PSF description.
- If one or several IRFs are not required for the study at hand, it is possible not to build them
by removing it from the list of options passed to the `~gammapy.makers.MapDatasetMaker`.

Copy part of a data store
+++++++++++++++++++++++++

To share specific data from a database, it might be necessary to create a new data storage with
a limited set of observations and summary files following the scheme described in gadf_.
This is possible with the method `~gammapy.data.DataStore.copy_obs` provided by the
`~gammapy.data.DataStore`. It allows to copy individual observations files in a given directory
and build the associated observation and HDU tables.

Interpolate maps onto a different geometry
++++++++++++++++++++++++++++++++++++++++++

To interpolate maps onto a different geometry, use `Map.interp_to_geom`,
see `here <tutorials/api/maps.html#Filling-maps-from-interpolation>`__.


Suppress warnings
+++++++++++++++++
In general it is not recommended to suppress warnings from code because they
might point to potential issues or help debugging a non-working script. However
in some cases the cause of the warning is known and the warnings clutter the
logging output. In this case it can be useful to locally suppress a specific
warning like so:

.. code::

    from astropy.io.fits.verify import VerifyWarning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        # do stuff here