.. include:: references.txt

.. _overview:

Overview
========

This page gives an overview of the main concepts in Gammapy. It is a theoretical
introduction to Gammapy, explaining which data, sub-packages, classes and methods
are involved in a data analysis with Gammapy.

.. _data_flow:

.. figure:: _static/data-flow-gammapy.png
    :width: 100%

    Data flow and sub-package structure of Gammapy. The folder icons
    represent the corresponding sub-packages. The direction of the
    the data flow is illustrated with shaded arrows. The top section
    shows the data levels as defined by `CTA`_.


Gammapy is organised in sub-packages. Figure :numref:`data_flow` illustrates
the data flow and sub-package structure of Gammapy.

.. _overview_data:


Data access (DL3)
-----------------

The data analysis starts with data level 3 FITS files consisting of event lists,
instrument response information
(effective area, point spread function, energy dispersion, background) and
extra information concerning the observation (pointing direction, time),
as well as two index tables that list the observations and declare which
response should be used with which event data.

For each observation, instrument response functions (namely effective area, point spread function, energy
dispersion, background) are distributed. Some details about the origin of these functions are given
in :ref:`irf-theory`. The functions are stored in the form of multidimensional tables giving the IRF
value as a function of position in the field-of-view and energy of the incident photon.

The formats used are discussed and described in `gadf`_. This format is still a prototype. In the coming
years CTA will develop and define it's release data format, and Gammapy  will adapt to that.

The main classes in Gammapy to access the DL3 data library are the
`~gammapy.data.DataStore` and `~gammapy.data.Observation`.
They are used to store and retrieve dynamically the datasets
relevant to any observation (event list in the form of an `~gammapy.data.EventList`,
IRFs see :ref:`irf` and other relevant informations).

Once some observation selection has been selected, the user can build a list of observations:
a `~gammapy.data.Observations` object, which will be used for the data reduction process.

See :ref:`gammapy.data <data>` and :ref:`gammapy.irf <irf>`

.. _overview_data_reduction:

Data reduction (DL3 -> DL4)
---------------------------

There are many data reduction options, but the main ones are whether to do a 3D
cube analysis or a 1D spectral analysis, and whether to keep individual
observations as separate datasets for a joint likelihood fit or whether to group
and stack them. Partly background modeling choices are also already made at this
data reduction stage. If you have a deep IACT observation, e.g. 100 observation
runs, the data reduction can take a while. So typically you write the output
datasets to file after data reduction, allowing you to read them back at any
time later for modeling and fitting.

See :ref:`gammapy.makers <makers>`.

.. _overview_datasets:

Datasets (DL4)
--------------

The `gammapy.datasets` sub-package contains classes to handle reduced
gamma-ray data for modeling and fitting.

The `Dataset` objects are the result of the data reduction step. They contain the various
products (`counts`, `exposure`, `energy dispersion` etc) with their geometries. They also
serve as the basis for modeling and fitting.

The `Dataset` class bundles reduced data, reduced IRFs and models.
Different sub-classes support different analysis methods and fit statistics
(e.g. Poisson statistics with known background or with OFF background measurements).

The `Datasets` are used to perform joint-likelihood fitting allowing to combine
different measurements, e.g. from different observations but also from different
instruments.

To learn more about datasets, see :ref:`gammapy.datasets <datasets>` and
:ref:`gammapy.maps <maps>`.

Gammapy supports binned simulation, i.e. Poisson fluctuation of predicted
counts maps or spectra, as well as event sampling to simulate DL3 events data.

Note that in Gammapy, 2D image analyses are partly done with actual 2D images
that don't have an energy axis, and partly with 3D cubes with a single energy bin,
e.g. for modeling and fitting,
see the `2D map analysis tutorial <./tutorials/image_analysis.html>`__.


.. _overview_modeling:

Modeling and Fitting (DL4 -> DL5)
---------------------------------
The second step is then typically to model and fit the datasets, either
individually, or in a joint likelihood analysis, using the
`~gammapy.datasets.Dataset`, `~gammapy.datasets.Datasets`,
`~gammapy.modeling.Fit` and model classes (see :ref:`overview_modeling` below).

You can specify your model using a YAML model specification, or write Python
code to specify which spectral and spatial models to use and what their rough
parameters are to start the fit (such as sky position and source extension, or
approximate flux level). Methods to run global model fits are available, as well
as methods to compute flux points or light curves, or run simple source
detection algorithms.


Beyond `Dataset` objects, Gammapy provides numerous functionalities related
to data modeling and fitting, as well as data simulation.
This includes spectral, spatial and temporal model classes to describe the gamma-ray sky.
Gammapy also contains a complete API for model parameter handling and model fitting,
with a large choice of spatial, spectral and temporal models. You may check out the whole list
of built-in models in the :ref:`model-gallery`.

To learn more about modeling and fitting, see  :ref:`gammapy.modeling <modeling>`
and :ref:`gammapy.estimators <estimators>`. To compute light curves, use the
`~gammapy.estimators.LightCurveEstimator`.

For 1D spectral modeling and fitting, `~gammapy.modeling.models.Models` are
used, to provide uniformity within Gammapy, and to allow in future versions of
Gammapy for advanced use cases where a sky region based analysis is used
resulting in 1D counts spectra, but the modeling is done with a spatial model
assumption, allowing for treatment of overlapping emission components, such as
e.g. a gamma-ray binary with underlying emission from a pulsar wind nebula, to
apply proper treatment of containment and contamination corrections. Note that
the spatial model on a `~gammapy.modeling.models.SkyModel` is optional, you can
only pass a `~gammapy.modeling.models.SpectralModel`, as shown in the `First
analysis tutorial notebook <./tutorials/analysis_1.html>`__ and other tutorials.


.. _overview_hli:

High Level Analysis Interface
-----------------------------
A convenient way to do this is to use the high level interface,
see :ref:`gammapy.analysis <analysis>`.

To analyse data from CTA or existing IACTs, the usual workflow is to use the
high-level interface in :ref:`gammapy.analysis <analysis>` as shown in the
example `First analysis tutorial notebook <./tutorials/analysis_1.html>`__, i.e.
to write a YAML config file, and then to use `~gammapy.analysis.AnalysisConfig`
and `~gammapy.analysis.Analysis` to perform the data reduction from event lists
and instrument response functions (IRFs) to a reduced data format called
datasets, using either 3D cube analysis or 1D region-based spectral analysis.
The IACT data distributed by instruments is called "data level 3" (DL3) and is
given as FITS files, as shown in the `CTA with Gammapy <./tutorials/cta.html>`__
and `H.E.S.S. with Gammapy <./tutorials/hess.html>`__ notebooks and explained in
more detail in :ref:`overview_data` below. Then `~gammapy.analysis.Analysis`
class is then used to compute intermediate reduced analysis files like counts
and exposure maps or spectra, and reduced point spread function (PSF) or energy
dispersion (EDISP) information, combined in container objects called datasets
(see below).

The analysis config file and `~gammapy.analysis.Analysis` class currently mostly
scripts the data reduction up to the datasets level for the most common analysis
cases. It might be extended in the future to become the "manager" or "driver"
class for modeling or fitting as well, or that might remain the responsibility
of the datasets, models and fit classes. Advanced users that need to run
specialises analyses such as e.g. complex background modeling, or grouping of
observations, have a second-level API available via dataset makers, that offer
more flexibility. An example of this is shown in the `Second analysis tutorial
notebook <./tutorials/analysis_2.html>`__.

.. _overview_other:

Other topics
------------

Gammapy ships with a ``gammapy`` command line tool, that can be used to check
your installation and show version information via ``gammapy info``, to download
example datasets and tutorials via ``gammapy download`` or to bootstrap an
analysis by creating a default config file via ``gammapy analysis``. To learn
about the Gammapy command line tool, see :ref:`gammapy.scripts <CLI>`.

See :ref:`gammapy.catalog <catalog>`, :ref:`gammapy.astro <astro>`,
:ref:`gammapy.stats <stats>`,
:ref:`gammapy.scripts <CLI>` (``gammapy`` command line tool).


.. _overview_next:

What next?
----------

You now have an overview of Gammapy. We suggest you continue by tring it out,
following the instructions in :ref:`install`, :ref:`getting-started` and then
the first and second analysis tutorials at :ref:`tutorials`.

.. toctree::
    :caption: Overview Subsections
    :maxdepth: 1

    overview/DL3
