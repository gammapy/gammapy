.. include:: references.txt

.. _overview:

Overview
========

This page gives an overview of the main concepts in Gammapy. :ref:`Fig. 1 <data_flow>`
illustrates the general data flow and corresponding sub-package structure of Gammapy.
Gammapy can be typically used with the configuration based high level analysis
API or as a standard Python library by importing the functionality from sub-packages.
The different data levels and data reduction steps and how they map to the Gammapy API
are explained in more detail in the following sections.

.. _data_flow:

.. figure:: _static/data-flow-gammapy.png
    :width: 100%

    Fig. 1 Data flow and sub-package structure of Gammapy. The folder icons
    represent the corresponding sub-packages. The direction of the
    the data flow is illustrated with shaded arrows. The top section
    shows the data levels as defined by `CTA`_.


.. _overview_data:


Data access and selection (DL3)
-------------------------------

The analysis of gamma-ray data with Gammapy starts at the "data level 3" (DL3, ref?).
At this level the data is stored as lists of gamma-like events and the corresponding
instrument response functions (IRFs). The instrument response includes effective
area, point spread function (PSF), energy dispersion and residual hadronic background.
In addition there is associated meta data including information on the observation such
as pointing] direction, observation time and obervation conditions. The main FITS format
supported by Gammapy is documented on the `Gamma astro data formats page <gadf>`_.

The access to the data and instrument response is implemented in
:ref:`gammapy.data <data>` and :ref:`gammapy.irf <irf>`.


.. _overview_data_reduction:

Data reduction (from DL3 to DL4)
--------------------------------

In the next stage of the analysis the user selects a coordinates system, region or
energy binning and events are binned into multidimensional data structures (maps)
with the selected geometry. The instrument response is projected onto the
same geometry as well. At this stage users can select additional background
estimation methods, such as ring background or reflected regions and
exclude parts of the data with high associated IRF systematics by defining
a "safe" data range. The counts data and the reduced IRFs are bundled into
datasets. Those datasets can be optionally grouped and stacked and are
typically writen to disk to allow users to read them back at any time later
for modeling and fitting.

The different data reduction and background estimation options are
implemented in :ref:`gammapy.makers <makers>`.

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


.. _overview_modeling:

Modeling and Fitting (from DL4 to DL5)
--------------------------------------
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

.. _overview_hli:

High Level Analysis Interface
-----------------------------
In addition to the individual

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

