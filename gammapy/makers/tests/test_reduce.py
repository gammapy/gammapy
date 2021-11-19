# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.datasets import MapDataset, SpectrumDataset
from gammapy.makers import (
    DatasetsMaker,
    FoVBackgroundMaker,
    MapDatasetMaker,
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def observations_cta():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    return data_store.get_observations()[:3]


@pytest.fixture(scope="session")
def observations_hess():
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids = [23523, 23526, 23559, 23592]
    return datastore.get_observations(obs_ids)


def get_mapdataset(name):
    skydir = SkyCoord(0, -1, unit="deg", frame="galactic")
    energy_axis = MapAxis.from_edges(
        [0.1, 1, 10], name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(
        skydir=skydir, binsz=0.5, width=(10, 5), frame="galactic", axes=[energy_axis]
    )
    return MapDataset.create(geom, name=name)


def get_spectrumdataset(name):
    target_position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
    on_region_radius = Angle("0.11 deg")
    on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)

    energy_axis = MapAxis.from_energy_bounds(
        0.1, 40, nbin=15, per_decade=True, unit="TeV", name="energy"
    )
    energy_axis_true = MapAxis.from_energy_bounds(
        0.05, 100, nbin=20, per_decade=True, unit="TeV", name="energy_true"
    )

    geom = RegionGeom.create(region=on_region, axes=[energy_axis])
    return SpectrumDataset.create(
        geom=geom, energy_axis_true=energy_axis_true, name=name
    )


@pytest.fixture(scope="session")
def exclusion_mask():
    exclusion_region = CircleSkyRegion(
        center=SkyCoord(183.604, -8.708, unit="deg", frame="galactic"),
        radius=0.5 * u.deg,
    )

    skydir = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
    geom = WcsGeom.create(
        npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs"
    )

    return ~geom.region_mask([exclusion_region])


@pytest.fixture(scope="session")
def makers_map():
    return [
        MapDatasetMaker(),
        SafeMaskMaker(methods=["offset-max"], offset_max="2 deg"),
        FoVBackgroundMaker(method="scale"),
    ]


@pytest.fixture(scope="session")
def makers_spectrum(exclusion_mask):
    return [
        SpectrumDatasetMaker(
            containment_correction=True, selection=["counts", "exposure", "edisp"]
        ),
        ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask),
        SafeMaskMaker(methods=["aeff-max"], aeff_percent=10),
    ]


@requires_data()
@pytest.mark.parametrize(
    "pars",
    [
        {
            "dataset": get_mapdataset(name="linear_staking"),
            "stack_datasets": True,
            "cutout_width": None,
            "n_jobs": None,
        },
        {
            "dataset": get_mapdataset(name="parallel"),
            "stack_datasets": False,
            "cutout_width": None,
            "n_jobs": 2,
        },
        {
            "dataset": get_mapdataset(name="parallel_staking"),
            "stack_datasets": True,
            "cutout_width": None,
            "n_jobs": 2,
        },
    ],
)
@requires_data()
def test_datasetsmaker_map(pars, observations_cta, makers_map):
    makers = DatasetsMaker(
        makers_map,
        stack_datasets=pars["stack_datasets"],
        cutout_mode="partial",
        cutout_width=pars["cutout_width"],
        n_jobs=pars["n_jobs"],
    )

    datasets = makers.run(pars["dataset"], observations_cta)
    if len(datasets) == 1:
        counts = datasets[0].counts
        assert counts.unit == ""
        assert_allclose(counts.data.sum(), 46716, rtol=1e-5)

        exposure = datasets[0].exposure
        assert exposure.unit == "m2 s"
        assert_allclose(exposure.data.mean(), 1.350841e09, rtol=3e-3)
    else:
        assert len(datasets) == 3
        # get by name because of async
        counts = datasets[0].counts
        assert counts.unit == ""
        assert_allclose(counts.data.sum(), 26318, rtol=1e-5)

        exposure = datasets[0].exposure
        assert exposure.unit == "m2 s"
        assert_allclose(exposure.data.mean(), 2.436063e09, rtol=3e-3)


@requires_data()
def test_datasetsmaker_map_cutout_width(observations_cta, makers_map, tmp_path):
    makers = DatasetsMaker(
        makers_map,
        stack_datasets=True,
        cutout_mode="partial",
        cutout_width="5 deg",
        n_jobs=None,
    )
    datasets = makers.run(get_mapdataset(name="linear_staking_1deg"), observations_cta)

    counts = datasets[0].counts

    assert counts.unit == ""
    assert_allclose(counts.data.sum(), 46716, rtol=1e-5)

    exposure = datasets[0].exposure
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.mean(), 1.350841e09, rtol=3e-3)


@requires_data()
def test_datasetsmaker_map_2steps(observations_cta, makers_map, tmp_path):

    makers = DatasetsMaker(
        [MapDatasetMaker()],
        stack_datasets=False,
        cutout_mode="partial",
        cutout_width="5 deg",
        n_jobs=None,
    )

    dataset = get_mapdataset(name="2steps")
    datasets = makers.run(dataset, observations_cta)

    makers_list = [
        SafeMaskMaker(methods=["offset-max"], offset_max="2 deg"),
        FoVBackgroundMaker(method="scale"),
    ]
    makers = DatasetsMaker(
        makers_list,
        stack_datasets=True,
        cutout_mode="partial",
        cutout_width="5 deg",
        n_jobs=None,
    )
    datasets = makers.run(dataset, observations_cta, datasets)

    counts = datasets[0].counts
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), 46716, rtol=1e-5)

    exposure = datasets[0].exposure
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.mean(), 1.350841e09, rtol=3e-3)


@requires_data()
def test_datasetsmaker_spectrum(observations_hess, makers_spectrum):

    makers = DatasetsMaker(makers_spectrum, stack_datasets=False, n_jobs=2)
    datasets = makers.run(get_spectrumdataset(name="spec"), observations_hess)

    counts = datasets[0].counts
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), 192, rtol=1e-5)
    assert_allclose(datasets[0].background.data.sum(), 18.66666664, rtol=1e-5)

    exposure = datasets[0].exposure
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.mean(), 3.94257338e08, rtol=3e-3)
