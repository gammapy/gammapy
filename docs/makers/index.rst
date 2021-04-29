.. include:: ../references.txt

.. _makers:

***********************
makers - Data reduction
***********************

.. currentmodule:: gammapy.makers

Introduction
============

The `gammapy.makers` sub-package contains classes to perform data reduction tasks
from DL3 data to binned datasets.

Getting started
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

    import astropy.units as u
    from gammapy.maps import MapAxis, WcsGeom
    from gammapy.datasets import MapDataset

    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=11, name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(
            skydir=(0.,0),
            frame='galactic',
            binsz=0.05*u.deg,
            width=5*u.deg,
            axes=[energy_axis]
    )
    dataset_empty = MapDataset.create(geom=geom)
    print(dataset_empty)

At this point the created `dataset` is empty. The ``.create()`` method has
additional options, e.g. it is also possible to specify a true energy axis:

.. code-block:: python

    import astropy.units as u
    from gammapy.maps import MapAxis, WcsGeom
    from gammapy.datasets import MapDataset

    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=11, name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(
            skydir=(0.,0),
            frame='galactic',
            binsz=0.05*u.deg,
            width=5*u.deg,
            axes=[energy_axis]
    )

    energy_axis_true = MapAxis.from_bounds(
        0.3, 10, nbin=31, name="energy", unit="TeV", interp="log"
    )
    dataset_empty = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true)
    print(dataset_empty)

Once this empty "reference" dataset is defined, it can be filled with observational
data using the `MapDatasetMaker`:

.. code-block:: python

    import astropy.units as u
    from gammapy.datasets import MapDataset
    from gammapy.makers import MapDatasetMaker
    from gammapy.data import DataStore
    from gammapy.maps import MapAxis, WcsGeom

    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs = data_store.get_observations([23592])[0]

    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=11, name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(
                skydir=(83.63, 22.01),
                axes=[energy_axis],
                width=5*u.deg,
                binsz=0.05*u.deg,
                frame='icrs'
    )
    dataset_empty = MapDataset.create(geom=geom)

    maker = MapDatasetMaker()
    dataset = maker.run(dataset_empty, obs)
    print(dataset)

The `MapDatasetMaker` fills the corresponding `counts`, `exposure`, `background`,
`psf` and `edisp` map per observation. The `MapDatasetMaker` has a
`selection` parameter, in case some of the maps should not be computed.


Safe data range handling
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

    import astropy.units as u
    from gammapy.datasets import  MapDataset
    from gammapy.makers import MapDatasetMaker, SafeMaskMaker
    from gammapy.data import DataStore
    from gammapy.maps import MapAxis, WcsGeom

    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs = data_store.get_observations([23592])[0]

    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=11, name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(
                skydir=(83.63, 22.01),
                axes=[energy_axis],
                width=5*u.deg,
                binsz=0.05*u.deg,
                frame='icrs'
    )
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

Background estimation
=====================

The background computed by the `MapDatasetMaker` gives the number of counts predicted
by the background IRF of the observation. Because its actual normalization, or even its
spectral shape, might be poorly constrained, it is necessary to correct it with the data
themselves. This is the role of background estimation Makers.

FoV background
--------------

If the background energy dependent morphology is well reproduced by the background model
stored in the IRF, it might be that its normalization is incorrect and that some spectral
corrections are necessary. This is made possible thanks to the `~gammapy.makers.FoVBackgroundMaker`.
This technique is recommended in most 3D data reductions.

For more details and usage, see :ref:`fov_background`.

Ring background
---------------

If the background model does not reproduce well the morphology, a classical approach consists
in applying local corrections by smoothing the data with a ring kernel. This allows to build a set
of OFF counts taking into account the inperfect knowledge of the background. This is implemented
in the `~gammapy.makers.RingBackgroundMaker` which transforms the Dataset in a `MapDatasetOnOff`.
This technique is mostly used for imaging, and should not be applied for 3D modeling and fitting.

For more details and usage, see :ref:`ring_background`.

Reflected regions background
----------------------------

In the absence of a solid background model, a classical technique in Cherenkov astronomy for 1D
spectral analysis is to estimate the background in a number of OFF regions. When the background
can be safely estimated as radially symmetric w.r.t. the pointing direction, one can apply the
reflected regions background technique.
This is implemented in the `~gammapy.makers.ReflectedRegionsBackgroundMaker` which transforms a
`SpectrumDataset` in a `SpectrumDatasetOnOff`. This technique is only used for 1D spectral
analysis.

For more details and usage, see :ref:`reflected_background`.


Stacking of datasets
====================

The `MapDataset` as well as `MapDatasetOnOff` both have an in-place ``stack()``
methods, which allows to stack individual `MapDataset`, which are computed
per observation into a larger dataset. During the stacking the safe data
range mask (`MapDataset.mask_safe`) is applied by setting data outside to
zero, then data is added to the larger map dataset. To stack multiple
observations, the larger dataset must be created first:

.. code-block:: python

    from gammapy.datasets import  MapDataset
    from gammapy.makers import MapDatasetMaker
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

Combining data reduction steps
==============================

The data reduction steps can be combined in a single loop to run
a full data reduction chain. For this the `MapDatasetMaker` is run
first and the output dataset is the passed on to the next maker step.
Finally the dataset per observation is stacked into a larger map.

.. code-block:: python

    from gammapy.datasets import  MapDataset
    from gammapy.makers import MapDatasetMaker, SafeMaskMaker
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


Using `gammapy.makers`
======================

Gammapy tutorial notebooks that show examples using ``gammapy.makers``:

.. nbgallery::

   ../tutorials/starting/analysis_2.ipynb
   ../tutorials/analysis/3D/analysis_3d.ipynb
   ../tutorials/analysis/3D/simulate_3d.ipynb
   ../tutorials/analysis/1D/spectral_analysis.ipynb
   ../tutorials/analysis/1D/spectrum_simulation.ipynb

Other examples using background makers:

.. toctree::
    :maxdepth: 1

    fov
    reflected
    ring


Reference/API
=============

.. automodapi:: gammapy.makers
    :no-inheritance-diagram:
    :include-all-objects:
