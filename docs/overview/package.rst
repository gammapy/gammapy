.. include:: ../references.txt

.. _overview_package:

Package Structure
=================

This page gives an overview of the main concepts in Gammapy. :ref:`Fig. 1 <data_flow>`
illustrates the general data flow and corresponding sub-package structure of Gammapy.
Gammapy can be typically used with the configuration based high level analysis
API or as a standard Python library by importing the functionality from sub-packages.
The different data levels and data reduction steps and how they map to the Gammapy API
are explained in more detail in the following sections.

.. _data_flow:

.. figure:: ../_static/data-flow-gammapy.png
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
as pointing] direction, observation time and observation conditions. The main FITS format
supported by Gammapy is documented on the `Gamma astro data formats page <gadf>`_.

The access to the data and instrument response is implemented in
:ref:`gammapy.data <data>` and :ref:`gammapy.irf <irf>`.


.. _overview_data_reduction:

Data reduction (DL3 to DL4)
---------------------------

In the next stage of the analysis the user selects a coordinates system, region or
energy binning and events are binned into multidimensional data structures (maps)
with the selected geometry. The instrument response is projected onto the
same geometry as well. At this stage users can select additional background
estimation methods, such as ring background or reflected regions and
exclude parts of the data with high associated IRF systematics by defining
a "safe" data range. The counts data and the reduced IRFs are bundled into
datasets. Those datasets can be optionally grouped and stacked and are
typically written to disk to allow users to read them back at any time later
for modeling and fitting.

The data reduction and background estimation methods are implemented in
:ref:`gammapy.makers <makers>`.

.. _overview_datasets:

Datasets (DL4)
--------------

The datasets classes bundle reduced data in form of maps, reduced IRFs, models and
fit statistics. Different sub-classes support different analysis methods
and fit statistics (e.g. Poisson statistics with known background or
with OFF background measurements). The datasets are used to perform joint-likelihood
fitting allowing to combine different measurements, e.g. from different observations
but also from different instruments or event classes. They can also be used for binned
simulation as well as event sampling to simulate DL3 events data.

To learn more about datasets, see :ref:`gammapy.datasets <datasets>` and
:ref:`gammapy.maps <maps>`.


.. _overview_modeling:

Modeling and Fitting (DL4 to DL5)
---------------------------------

The next step is then typically to model and fit the datasets, either
individually, or in a joint likelihood analysis. For this purpose Gammapy
provides a uniform interface to multiple fitting backends. It also provides
a variety of :ref:`built in models <model-gallery>`. This includes spectral,
spatial and temporal model classes to describe the gamma-ray emission in the sky.
Independently or subsequently to the global modelling, the data can be
re-grouped to compute flux points, light curves and flux as well as significance
maps in energy bands.

To learn more about modeling and fitting, see  :ref:`gammapy.modeling <modeling>`
and :ref:`gammapy.estimators <estimators>`.


.. _overview_hli:

High Level Analysis Interface
-----------------------------
To define and execute a full data analysis process from a YAML configuration file,
Gammapy implements a high level analysis interface. It exposes a subset of
the functionality that is available in the sub-packages to support
standard analysis use case in a convenient way.

The high level analysis interface can be found in :ref:`gammapy.analysis <analysis>`.

.. _overview_other:

Other
-----

Gammapy offers additional functionality in sub-packages not related to the
standard analysis work flow described above. This includes:

* Access to a variety of GeV-TeV gamma-ray catalogs in :ref:`gammapy.catalog <catalog>`
* Support for simulation of TeV source populations and dark matter models in :ref:`gammapy.astro <astro>`
* Statistical utility functions in :ref:`gammapy.stats <stats>`
* Command line tools in :ref:`gammapy.scripts <CLI>`


.. _overview_next:

What next?
----------

After this overview of the Gammapy package, we suggest to continue by trying it out,
following the instructions in :ref:`install`, :ref:`getting-started` and then
the first and second analysis tutorials at :ref:`tutorials`.
