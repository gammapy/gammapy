.. include:: references.txt

.. _overview:

Overview
========

This page gives an overview of the main concepts in Gammapy. It is a theoretical
introduction to Gammapy, explaining what data, packages, classes and methods are
involved in a data analysis with Gammapy, proviging many links to other
documentation pages and tutorials, but not giving code examples here. For a
hands-on introduction how to use Gammapy, see :ref:`install`,
:ref:`getting-started` and the many :ref:`tutorials`.

Gammapy is an open-source Python package for gamma-ray astronomy built on Numpy
and Astropy. It is a prototype for the Cherenkov Telescope Array (CTA) science
tools, and can also be used to analyse data from existing gamma-ray telescopes,
if their data is available in the standard FITS format (gadf_). There is good
support for CTA and existing imaging atmospheric Cherenkov telescopes (IACTs,
e.g. H.E.S.S., MAGIC, VERITAS), and some analysis capabilities for Fermi-LAT and
HAWC and multi-mission joint likelihood analysis.

.. _overview_workflow:

Workflow
--------

To use Gammapy you need a basic knowledge of Python, Numpy, Astropy, as well as
matplotlib for plotting. Many standard gamma-ray analyses can be done with few
lines of configuration and code, so you can get pretty far by copy and pasting
and adapting the working examples from the Gammapy documentation. But
eventually, if you want to script more complex analyses, or inspect analysis
results or intermediate analysis products, you need to acquire a basic to
intermediate Python skill level.

To analyse data from CTA or existing IACTs, the usual workflow is to use the
high-level interface in :ref:`gammapy.analysis <analysis>` as shown in the
example `First analysis tutorial notebook <notebooks/analysis_1.html>`__, i.e.
to write a YAML config file, and then to use `~gammapy.analysis.AnalysisConfig`
and `~gammapy.analysis.Analysis` to perform the data reduction from event lists
and instrument response functions (IRFs) to a reduced data format called
datasets, using either 3D cube analysis or 1D region-based spectral analysis.
The IACT data distributed by instruments is called "data level 3" (DL3) and is
given as FITS files, as shown in the `CTA with Gammapy <notebooks/cta.html>`__
and `H.E.S.S. with Gammapy <notebooks/hess.html>`__ notebooks and explained in
more detail in :ref:`overview_data` below. Then `~gammapy.analysis.Analysis`
class is then used to compute intermediate reduced analysis files like counts
and exposure maps or spectra, and reduced point spread function (PSF) or energy
dispersion (EDISP) information, combined in container objects called datasets
(see below).

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

The analysis config file and `~gammapy.analysis.Analysis` class currently mostly
scripts the data reduction up to the datasets level for the most common analysis
cases. It might be extended in the future to become the "manager" or "driver"
class for modeling or fitting as well, or that might remain the responsibility
of the datasets, models and fit classes. Advanced users that need to run
specialises analyses such as e.g. complex background modeling, or grouping of
observations, have a second-level API available via dataset makers, that offer
more flexibility. An example of this is shown in the `Second analysis tutorial
notebook <notebooks/analysis_2.html>`__.

Gammapy ships with a ``gammapy`` command line tool, that can be used to check
your installation and show version information via ``gammapy info``, to download
example datasets and tutorials via ``gammapy download`` or to bootstrap an
analysis by creating a default config file via ``gammapy analysis``. To learn
about the Gammapy command line tool, see :ref:`gammapy.scripts <CLI>`.

.. _overview_data:

Data reduction
--------------

As already mentioned in the :ref:`overview_workflow` section above, IACT
analysis starts with :ref:`data level 3 <overview_DL3>` (DL3) FITS files
consisting of event lists, instrument response information (effective area,
point spread function, energy dispersion, background) and extra information
concerning the observation (pointing direction, time), as well as two index
tables that list the observations and declare which response should be used
with which event data.



There are many data reduction options, but the main ones are whether to do a 3D
cube analysis or a 1D spectral analysis, and whether to keep individual
observations as separate datasets for a joint likelihood fit or whether to group
and stack them. Partly background modeling choices are also already made at this
data reduction stage. If you have a deep IACT observation, e.g. 100 observation
runs, the data reduction can take a while. So typically you write the output
datasets to file after data reduction, allowing you to read them back at any
time later for modeling and fitting.

.. _overview_datasets:

Datasets
--------

Datasets in Gammapy contain reduced data, models, and the likelihood function
fit statistic for a given set of model parameters. All datasets contain a
`~gammapy.modeling.models.Models` container with one or more
`~gammapy.modeling.models.SkyModel` objects that represent additive emission
components.

To model and fit data in Gammapy, you have to create a
`~gammapy.datasets.Datasets` container object with one or multiple
`~gammapy.datasets.Dataset` objects. Gammapy has built-in support to create and
analyse the following datasets: `~gammapy.datasets.MapDataset`,
`~gammapy.datasets.MapDatasetOnOff`, `~gammapy.datasets.SpectrumDataset`,
`~gammapy.datasets.SpectrumDatasetOnOff` and
`~gammapy.datasets.FluxPointsDataset`.

The map datasets represent 3D cubes (`~gammapy.maps.WcsNDMap` objects) with two
spatial and one energy axis. For 2D images the same map objects and map datasets
are used, an energy axis is present but only has one energy bin. The
`~gammapy.datasets.MapDataset` contains a counzts map, background is modeled with a
`~gammapy.modeling.models.BackgroundModel`, and the fit statistic used is
``cash``. The `~gammapy.datasets.MapDatasetOnOff` contains on and off count maps,
background is implicitly modeled via the off counts map, and the ``wstat`` fit
statistic.

The spectrum datasets represent 1D spectra (`~gammapy.maps.RegionNDMap`
objects) with an energy axis. There are no spatial axes or information, the 1D
spectra are obtained for a given on region. The
`~gammapy.datasets.SpectrumDataset` contains a counts spectrum, background is
modeled with a `~gammapy.maps.RegionNDMap`, and the fit statistic used is
``cash``. The `~gammapy.datasets.SpectrumDatasetOnOff` contains on on and off
count spectra, background is implicitly modeled via the off counts spectrum, and
the ``wstat`` fit statistic. The `~gammapy.datasets.FluxPointsDataset` contains
`~gammapy.estimatorsFluxPoints` and a spectral model, the fit statistic used is
``chi2``.

To learn more about datasets, see :ref:`gammapy.datasets <datasets>` and :ref:`gammapy.modeling <modeling>`.

.. _overview_modeling:

Modeling and Fitting
--------------------

Assuming you have prepared your gamma-ray data as a set of
`~gammapy.datasets.Dataset` objects (see :ref:`overview_datasets` above), and
stored one or more datasets in a `~gammapy.datasets.Datasets` container, you are
all set for modeling and fitting. Either via a YAML config file, or via Python
code, define the `~gammapy.modeling.models.Models` to use, which is a list of
`~gammapy.modeling.models.SkyModel` objects representing additive emission
components, usually sources or diffuse emission, although a single source can
also be modeled by multiple components if you want. The
`~gammapy.modeling.models.SkyModel` is a factorised model with a
`~gammapy.modeling.models.SpectralModel` component and a
`~gammapy.modeling.models.SpatialModel` component. Most commonly used models in
gamma-ray astronomy are built-in, see the `Model gallery
<notebooks/models.html>`__. It is easy to create user-defined models and
datasets, Gammapy is very flexible. 

The `~gammapy.modeling.Fit` class provides methods to fit, i.e. optimise
parameters and estimate parameter errors and correlations. It interfaces with a
`~gammapy.datasets.Datasets` object, which in turn is connected to a
`~gammapy.modeling.models.Models` object, which has a
`~gammapy.modeling.Parameters` object, which contains the model parameters.
Currently ``iminuit`` is used as modeling and fitting backend, in the future we
plan to support other optimiser and error estimation methods, e.g. from
``scipy``. Models can be unique for a given dataset, or contribute to multiple
datasets and thus provide links, allowing e.g. to do a joint fit to multiple
IACT datasets, or to a joint IACT and Fermi-LAT dataset. Many examples are given
in the tutorials.

To learn more about modeling and fitting, see  :ref:`gammapy.modeling
<modeling>`.

.. _overview_time:

Time analysis
-------------

Light curves are represented as `~gammapy.time.LightCurve` objects, a wrapper
class around `~astropy.table.Table`. To compute light curves, use the
`~gammapy.time.LightCurveEstimator`.

To learn more about time, see :ref:`gammapy.time <time>`.

.. _overview_simulation:

Simulation
----------

Gammapy currently supports binned simulation, Poisson fluctuation of predicted
counts maps. The following tutorials illustrate how to use that to predict
observability, significance and sensitivity, using CTA examples: `3D map
simulation <notebooks/simulate_3d.html>`__, `1D spectrum simulation
<notebooks/spectrum_simulation.html>`__, and `Point source sensitivity
<notebooks/cta_sensitivity.html>`__.

Development of event sampling is work in progress, currently Gammapy cannot be
used yet to simulate DL3 events data.

.. _overview_other:

Other topics
------------

Gammapy is organised in sub-packages, containing many classes and functions. In
this overview we only mentioned the most important concepts and parts to get
started. To learn more, see the following sub packages and documentation pages:
:ref:`gammapy.data <data>`, :ref:`gammapy.irf <irf>`, :ref:`gammapy.maps
<maps>`, :ref:`gammapy.catalog <catalog>`, :ref:`gammapy.astro <astro>`,
:ref:`gammapy.stats <stats>`,
:ref:`gammapy.scripts <CLI>` (``gammapy`` command line tool).

Note that in Gammapy, 2D image analyses are partly done with actual 2D images
that don't have an energy axis (e.g. in :ref:`gammapy.detect <detect>`), and
partly with 3D cubes with a single energy bin, e.g. for modeling and fitting,
see the `2D map analysis tutorial <notebooks/image_analysis.html>`__.

For 1D spectral modeling and fitting, `~gammapy.modeling.models.Models` are
used, to provide uniformity within Gammapy, and to allow in future versions of
Gammapy for advanced use cases where a sky region based analysis is used
resulting in 1D counts spetra, but the modeling is done with a spatial model
assumption, allowing for treatment of overlapping emission components, such as
e.g. a gamma-ray binary with underlying emission from a pulsar wind nebula, to
apply proper treatment of containment and contamination corrections. Note that
the spatial model on a `~gammapy.modeling.models.SkyModel` is optional, you can
only pass a `~gammapy.modeling.models.SpectralModel`, as shown in the `First
analysis tutorial notebook <notebooks/analysis_1.html>`__ and other tutorials.

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
