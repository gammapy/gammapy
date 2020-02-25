.. include:: ../references.txt

.. _ring_background:

***************
Ring background
***************

.. currentmodule:: gammapy.makers

Overview
--------
This technique is used in classical Cherenkov astronomy for the 2D image
computation.

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
