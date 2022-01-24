# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.datasets import MapDataset
from gammapy.makers import (
    AdaptiveRingBackgroundMaker,
    MapDatasetMaker,
    RingBackgroundMaker,
    SafeMaskMaker,
)
from gammapy.maps import MapAxis, WcsGeom
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def observations():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids = [23523, 23526]
    return datastore.get_observations(obs_ids)


@pytest.fixture(scope="session")
def geom():
    energy_axis = MapAxis.from_edges([1, 10], unit="TeV", name="energy", interp="log")
    return WcsGeom.create(
        skydir=SkyCoord(83.633, 22.014, unit="deg"),
        binsz=0.02,
        width=(10, 10),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis],
    )


@pytest.fixture(scope="session")
def exclusion_mask(geom):
    """Example mask for testing."""
    pos = SkyCoord(83.633, 22.014, unit="deg", frame="icrs")
    region = CircleSkyRegion(pos, Angle(0.15, "deg"))
    return ~geom.region_mask([region])


@requires_data()
def test_ring_bkg_maker(geom, observations, exclusion_mask):
    ring_bkg_maker = RingBackgroundMaker(
        r_in="0.2 deg", width="0.3 deg", exclusion_mask=exclusion_mask
    )
    safe_mask_maker = SafeMaskMaker(methods=["offset-max"], offset_max="2 deg")
    map_dataset_maker = MapDatasetMaker(selection=["counts", "background", "exposure"])

    reference = MapDataset.create(geom)
    datasets = []

    for obs in observations:
        cutout = reference.cutout(obs.pointing_radec, width="4 deg")
        dataset = map_dataset_maker.run(cutout, obs)
        dataset = safe_mask_maker.run(dataset, obs)
        dataset = dataset.to_image()

        dataset_on_off = ring_bkg_maker.run(dataset)
        datasets.append(dataset_on_off)

    mask = dataset.mask_safe
    assert_allclose(datasets[0].counts_off.data[mask].sum(), 2511333)
    assert_allclose(datasets[1].counts_off.data[mask].sum(), 2143577.0)
    assert_allclose(datasets[0].acceptance_off.data[mask].sum(), 2961300, rtol=1e-5)
    assert_allclose(datasets[1].acceptance_off.data[mask].sum(), 2364657.2, rtol=1e-5)
    assert_allclose(datasets[0].alpha.data[0][100][100], 0.00063745599, rtol=1e-5)
    assert_allclose(
        datasets[0].exposure.data[0][100][100], 806254444.8480084, rtol=1e-5
    )


@pytest.mark.parametrize(
    "pars",
    [
        {
            "obs_idx": 0,
            "method": "fixed_r_in",
            "counts_off": 2511417.0,
            "acceptance_off": 2960679.594648,
            "alpha": 0.000637456020,
            "exposure": 806254444.8480084,
        },
        {
            "obs_idx": 0,
            "method": "fixed_width",
            "counts_off": 2511417.0,
            "acceptance_off": 2960679.594648,
            "alpha": 0.000637456020,
            "exposure": 806254444.8480084,
        },
        {
            "obs_idx": 1,
            "method": "fixed_r_in",
            "counts_off": 2143577.0,
            "acceptance_off": 2364657.352647,
            "alpha": 0.00061841976,
            "exposure": 779613265.2688407,
        },
        {
            "obs_idx": 1,
            "method": "fixed_width",
            "counts_off": 2143577.0,
            "acceptance_off": 2364657.352647,
            "alpha": 0.00061841976,
            "exposure": 779613265.2688407,
        },
    ],
)
@requires_data()
def test_adaptive_ring_bkg_maker(pars, geom, observations, exclusion_mask):
    adaptive_ring_bkg_maker = AdaptiveRingBackgroundMaker(
        r_in="0.2 deg",
        width="0.3 deg",
        r_out_max="2 deg",
        stepsize="0.2 deg",
        exclusion_mask=exclusion_mask,
        method=pars["method"],
    )
    safe_mask_maker = SafeMaskMaker(methods=["offset-max"], offset_max="2 deg")
    map_dataset_maker = MapDatasetMaker(selection=["counts", "background", "exposure"])

    obs = observations[pars["obs_idx"]]

    dataset = MapDataset.create(geom).cutout(obs.pointing_radec, width="4 deg")
    dataset = map_dataset_maker.run(dataset, obs)
    dataset = safe_mask_maker.run(dataset, obs)

    dataset = dataset.to_image()
    dataset_on_off = adaptive_ring_bkg_maker.run(dataset)

    mask = dataset.mask_safe
    assert_allclose(dataset_on_off.counts_off.data[mask].sum(), pars["counts_off"])
    assert_allclose(
        dataset_on_off.acceptance_off.data[mask].sum(),
        pars["acceptance_off"],
        rtol=1e-5,
    )
    assert_allclose(dataset_on_off.alpha.data[0][100][100], pars["alpha"], rtol=1e-5)
    assert_allclose(
        dataset_on_off.exposure.data[0][100][100], pars["exposure"], rtol=1e-5
    )
