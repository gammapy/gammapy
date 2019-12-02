.. include:: ../references.txt

.. _cube:

************************
cube - Map cube analysis
************************

.. currentmodule:: gammapy.cube

Introduction
============

The `gammapy.cube` sub-package contains functions and classes to prepare
datasets for modeling and fitting.

Getting Started
===============
In the data reduction step the DL3 data is prepared for modeling and fitting,
by binning events into a counts map and interpolating the exposure, background,
psf and energy dispersion on the chosen analysis geometry. The counts, exposure,
background and IRF maps are bundled together in a data structure named a `MapDataset`.
To handle on-off observations Gammapy also features a `MapDatasetOnOff` class, which
stores in addition the `counts_off`, `acceptance` and `acceptance_off` data.

A `MapDataset` can be created from any `WcsGeom` object. This is illustrated
in the following example:

.. code-block:: python

    from gammapy.maps import MapAxis, WcsGeom
    from gammapy.cube import MapDataset

    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=11, name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(axes=[energy_axis])
    dataset_empty = MapDataset.create(geom=geom)
    print(dataset_empty)

At this point the created `dataset` is empty. The ``.create()`` method has
additional options, e.g. it is also possible to specify a true energy axis:

.. code-block:: python

    from gammapy.maps import MapAxis, WcsGeom
    from gammapy.cube import MapDataset

    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=11, name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(axes=[energy_axis])

    energy_axis_true = MapAxis.from_bounds(
        0.3, 10, nbin=31, name="energy", unit="TeV", interp="log"
    )
    dataset_empty = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true)
    print(dataset_empty)

Once this empty "reference" dataset is defined, it can be filled with observational
data using the `MapDatasetMaker`:

.. code-block:: python

    from gammapy.cube import MapDatasetMaker, MapDataset
    from gammapy.data import DataStore
    from gammapy.maps import MapAxis, WcsGeom

    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs = data_store.get_observations([23592])[0]

    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=11, name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(skydir=(83.63, 22.01), axes=[energy_axis], width=5)
    dataset_empty = MapDataset.create(geom=geom)

    maker = MapDatasetMaker()
    dataset = maker.run(dataset_empty, obs)
    print(dataset)

The `MapDatasetMaker` fills the corresponding `counts`, `exposure`, `background`,
`psf` and `edisp` map per observation. The `MapDatasetMaker` has a
`selection` parameter, in case some of the maps should not be computed.


Safe Data Range Handling
========================

To exclude the data range from a `MapDataset`, that is associated with
high systematics on instrument response functions, a `mask_safe`
can be defined. The `mask_safe` is a `Map` object with `bool` data
type, which indicates for each pixel, whether it should be included
in the analysis. The convention is that a value of `True` or `1`
includes the pixel, while a value of `False` or `0` excludes a pixels
from the analysis. To compute safe data range masks according to certain
criteria, Gammapy provides a `SafeMaskMaker` class.

Here is an example how to use it:

.. code-block:: python

    from gammapy.cube import MapDatasetMaker, MapDataset, SafeMaskMaker
    from gammapy.data import DataStore
    from gammapy.maps import MapAxis, WcsGeom

    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs = data_store.get_observations([23592])[0]

    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=11, name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(skydir=(83.63, 22.01), axes=[energy_axis], width=5)
    dataset_empty = MapDataset.create(geom=geom)

    maker = MapDatasetMaker()
    safe_mask_maker = SafeMaskMaker(
        methods=["aeff-default", "offset-max"], offset_max="3 deg"
    )

    dataset = maker.run(dataset_empty, obs)
    dataset = safe_mask_maker.run(dataset, obs)
    print(dataset.mask_safe)

The ``methods`` keyword specifies the method used. Please take a
look at `SafeMaskMaker` to see which methods are available.
The `SafeMaskMaker` does not modify any data, but only defines the
`MapDataset.mask_safe` attribute. This means that the safe data
range can be defined and modified in between the data reduction
and stacking and fitting. For a joint-likelihood analysis of multiple
observations the safe mask is applied to the counts and predicted
number of counts map during fitting. This correctly accounts for
contributions (spill-over) by the PSF from outside the field of view.


Stacking of Datasets
====================

The `MapDataset` as well as `MapDatasetOnOff` both have an in-place ``stack()``
methods, which allows to stack individual `MapDataset`, which are computed
per observation into a larger dataset. During the stacking the safe data
range mask (`MapDataset.mask_safe`) is applied by setting data outside to
zero, then data is added to the larger map dataset. To stack multiple
observations, the larger dataset must be created first:

.. code-block:: python

    from gammapy.cube import MapDatasetMaker, MapDataset
    from gammapy.data import DataStore
    from gammapy.maps import MapAxis, WcsGeom

    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs = data_store.get_observations([23592])[0]

    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=11, name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(skydir=(83.63, 22.01), axes=[energy_axis], width=5, binsz=0.02)
    stacked = MapDataset.create(geom=geom)

    maker = MapDatasetMaker()

    dataset = maker.run(stacked, obs)
    stacked.stack(dataset)

    print(stacked)

Combining Data Reduction Steps
==============================

The data reduction steps can be combined in a single loop to run
a full data reduction chain. For this the `MapDatasetMaker` is run
first and the output dataset is the passed on to the next maker step.
Finally the dataset per observation is stacked into a larger map.

.. code-block:: python

    from gammapy.cube import MapDatasetMaker, MapDataset, SafeMaskMaker
    from gammapy.data import DataStore
    from gammapy.maps import MapAxis, WcsGeom

    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    observations = data_store.get_observations([23523, 23592, 23526, 23559])

    energy_axis = MapAxis.from_bounds(1, 10, nbin=11, name="energy", unit="TeV", interp="log")
    geom = WcsGeom.create(skydir=(83.63, 22.01), axes=[energy_axis], width=5, binsz=0.02)

    maker = MapDatasetMaker()
    safe_mask_maker = SafeMaskMaker(methods=["aeff-default", "offset-max"], offset_max="3 deg")

    stacked = MapDataset.create(geom)

    for obs in observations:
    	cutout = stacked.cutout(obs.pointing_radec, width="6 deg")
        dataset = maker.run(cutout, obs)
        dataset = safe_mask_maker.run(dataset, obs)
        stacked.stack(dataset)

    print(stacked)

To maintain good performance it is always recommended to do a cutout
of the `MapDataset` as shown above. In case you want to increase the
offset-cut later, you can also choose a larger width of the cutout
than `2 * offset_max`.


Ring Background Estimation
==========================

To include the classical ring background estimation into a data reduction
chain, Gammapy provides the `RingBackgroundMaker` and `AdaptiveRingBackgroundMaker`
classed. Theses classes can only be used for image based data.
A given `MapDataset` has to be reduced to a single image by calling
`MapDataset.to_image()`

.. code-block:: python

    from gammapy.cube import MapDatasetMaker, MapDataset, RingBackgroundMaker, SafeMaskMaker
    from gammapy.data import DataStore
    from gammapy.maps import MapAxis, WcsGeom

    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    observations = data_store.get_observations([23592, 23559])

    energy_axis = MapAxis.from_bounds(
        0.3, 30, nbin=11, name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(skydir=(83.63, 22.01), axes=[energy_axis], width=8, binsz=0.02)

    stacked = MapDataset.create(geom)
    stacked_on_off = MapDataset.create(geom.squash(axis="energy"))

    maker = MapDatasetMaker(offset_max="3 deg")
    safe_mask_maker = SafeMaskMaker(
        methods=["aeff-default", "offset-max"], offset_max="3 deg"
    )
    ring_bkg_maker = RingBackgroundMaker(r_in="0.3 deg", width="0.3 deg")

    for obs in observations:
        dataset = maker.run(stacked, obs)
        dataset = safe_mask_maker.run(dataset, obs)
        dataset_on_off = ring_bkg_maker.run(dataset.to_image())
        stacked_on_off.stack(dataset_on_off)

Using `gammapy.cube`
=====================

Gammapy tutorial notebooks that show examples using ``gammapy.cube``:

* `analysis_3d.html <../notebooks/analysis_3d.html>`__
* `simulate_3d.html <../notebooks/simulate_3d.html>`__
* `fermi_lat.html <../notebooks/fermi_lat.html>`__

Reference/API
=============

.. automodapi:: gammapy.cube
    :no-inheritance-diagram:
    :include-all-objects:
