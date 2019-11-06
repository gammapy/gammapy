# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from regions import CircleSkyRegion
from astropy.coordinates import Angle, SkyCoord
from gammapy.cube import AdaptiveRingBackgroundMaker, RingBackgroundMaker
from gammapy.maps import WcsNDMap, WcsGeom, MapAxis
from gammapy.cube.make import MapDatasetMaker
from gammapy.utils.testing import requires_data
from gammapy.data import DataStore


@pytest.fixture(scope="session")
def exclusion_region():
    """Example mask for testing."""
    pos = SkyCoord(83.633, 22.014, unit="deg", frame="icrs")
    return CircleSkyRegion(pos, Angle(0.15, "deg"))


@pytest.fixture(scope="session")
def observations():
    """Example observation list for testing."""
    datastore = DataStore.from_file(
        "$GAMMAPY_DATA/hess-dl3-dr1/hess-dl3-dr3-with-background.fits.gz"
    )
    obs_ids = [23523, 23526]
    return datastore.get_observations(obs_ids)


@pytest.fixture()
def map_dataset_maker():
    energy_axis = MapAxis.from_edges(
        np.logspace(0, 1.0, 5), unit="TeV", name="ENERGY", interp="log"
    )
    geom = WcsGeom.create(
        skydir=SkyCoord(83.633, 22.014, unit="deg"),
        binsz=0.02,
        width=(10, 10),
        coordsys="GAL",
        proj="CAR",
        axes=[energy_axis],
    )
    return MapDatasetMaker(geom=geom, offset_max="2 deg")


def adaptive_ring_bkg_maker(method):
    return AdaptiveRingBackgroundMaker(
        r_in="0.2 deg",
        width="0.3 deg",
        r_out_max="2 deg",
        stepsize="0.2 deg",
        method=method,
    )


@requires_data()
def test_ring_bkg_maker(map_dataset_maker, observations, exclusion_region):
    ring_bkg_maker = RingBackgroundMaker(r_in="0.2 deg", width="0.3 deg")
    datasets = []

    for obs in observations:
        dataset = map_dataset_maker.run(obs)
        dataset = dataset.to_image()

        geom_cutout = dataset.counts.geom
        exclusion = WcsNDMap.from_geom(geom_cutout)
        exclusion.data = exclusion.geom.region_mask([exclusion_region], inside=False)

        dataset_on_off = ring_bkg_maker.run(dataset, exclusion)
        datasets.append(dataset_on_off)

    assert_allclose(datasets[0].counts_off.data.sum(), 2511417.0)
    assert_allclose(datasets[1].counts_off.data.sum(), 2143577.0)
    assert_allclose(datasets[0].acceptance_off.data.sum(), 698869.8)
    assert_allclose(datasets[1].acceptance_off.data.sum(), 697233.6)

    assert_allclose(datasets[0].alpha.data[0][100][100], 0.000668635)
    assert_allclose(datasets[0].exposure.data[0][100][100], 639038346.7895743)


@requires_data()
def test_adaptive_ring_bkg_maker(map_dataset_maker, observations, exclusion_region):
    datasets = {}

    for method in ["fixed_width", "fixed_r_in"]:
        datasets.update({method: []})
        for obs in observations:
            dataset = map_dataset_maker.run(obs)
            dataset = dataset.to_image()

            geom_cutout = dataset.counts.geom
            exclusion = WcsNDMap.from_geom(geom_cutout)
            exclusion.data = exclusion.geom.region_mask(
                [exclusion_region], inside=False
            )

            dataset_on_off = adaptive_ring_bkg_maker(method).run(dataset, exclusion)
            datasets[method].append(dataset_on_off)

    assert_allclose(datasets["fixed_r_in"][0].counts_off.data.sum(), 2511417.0)
    assert_allclose(datasets["fixed_width"][0].counts_off.data.sum(), 2511417.0)
    assert_allclose(datasets["fixed_r_in"][1].counts_off.data.sum(), 2143577.0)
    assert_allclose(datasets["fixed_r_in"][0].acceptance_off.data.sum(), 698869.8)
    assert_allclose(datasets["fixed_width"][1].acceptance_off.data.sum(), 697233.6)

    assert_allclose(datasets["fixed_r_in"][0].alpha.data[0][100][100], 0.000668635)
    assert_allclose(
        datasets["fixed_width"][0].exposure.data[0][100][100], 639038346.7895743
    )
