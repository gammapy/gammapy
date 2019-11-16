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
In the data reduction step, the DL3 data is prepared for modeling and fitting.


.. code:: python

	from gammapy.maps import MapAxis, WcsGeom
	from gammapy.cube import MapDataset

	energy_axis = MapAxis.from_bounds(1, 10, nbin=11, name="energy", unit="TeV", interp="log")
	geom = WcsGeom.create(axes=[energy_axis])
	reference = MapDataset.create(geom=geom)

Which creates an empty `MapDataset`, defining the analysis geometry.

.. code:: python

	from gammapy.cube import MapDatasetMaker

	maker = MapDatasetMaker()
	dataset = maker.run(reference, obs)


The `MapDatasetMaker` creates a cutout of the corresponding observation and
fills the correspoding `counts`, `exposure`, `background`, `psf` and `edisp`
map per observation. The size of the cutout is typically the maximal
offset from the observation position.


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

.. code:: python

	from gammapy.cube import SafeMaskMaker

	safe_mask_maker = SafeMaskMaker(methods=["aeff-default"])

	dataset = safe_mask_maker.run(dataset, obs)


The `methods` keyword specifies the method used. Please take a
look at `SafeMaskMaker` for to see which methods are available.
Multiple methods of the `SafeMaskMaker` can be combined:

.. code:: python

	safe_mask_maker = SafeMaskMaker(
		methods=["offset-max", "edisp-bias"],
		offset_max="3 deg",
		bias_percent=10
		)

	dataset = safe_mask_maker.run(dataset, obs)



Stacking of Datasets
====================

The `MapDataset` as well as `MapDatasetOnOff` both have in in-place `.stack()`
methods, which allows to stack individual `MapDataset`, which are computed
per observation into a larger dataset. During the stacking the safe data
range mask (`MapDataset.mask_safe`) is applied by setting data outside to
zero, then data is added to the larger map dataset.

.. code:: python






Combining Data Reductions Steps
===============================

The data reduction steps can be combined in a single loop to run
a full data reduction chain. For this the `MapDatasetMaker` is run
first and the output dataset is the passed on to the next maker step.
Finally the dataset per observation is stacked into a larger map.


.. code:: python

	from gammapy.cube import MapDatasetMaker

	stacked = MapDataset.create(geom)

	for obs in observations:
		dataset = maker.run(stacked, obs)
		dataset = safe_mask_maker.run(dataset, obs)
		stacked.stack(dataset)



Ring Background Analysis
========================

To include the classical ring background estimation into a data reduction
chain, Gammapy provides the `RingBackgroundMaker` and `AdaptiveRingBackgroundMaker`
class. Theses classes can only be used for image based data. So far
it does not handle energy dependent maps.

.. code:: python

	from gammapy.cube import MapDatasetMaker

	stacked = MapDataset.create(geom)

	for obs in observations:
		dataset = maker.run(stacked, obs)
		dataset = safe_mask_maker.run(dataset, obs)
		stacked.stack(dataset)


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
