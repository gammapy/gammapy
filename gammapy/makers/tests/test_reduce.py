# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from gammapy.makers import (
    DatasetsMaker,
    MapDatasetMaker,
    SafeMaskMaker,
    FoVBackgroundMaker,
)
from gammapy.maps import MapAxis, WcsGeom
from gammapy.datasets import MapDataset
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def observations():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    return data_store.get_observations()[:3]


def get_mapdataset(name):
    skydir = SkyCoord(0, -1, unit="deg", frame="galactic")
    energy_axis = MapAxis.from_edges(
        [0.1, 1, 10], name="energy", unit="TeV", interp="log"
    )
    geom = WcsGeom.create(
        skydir=skydir, binsz=0.5, width=(10, 5), frame="galactic", axes=[energy_axis]
    )
    return MapDataset.create(geom)


@pytest.fixture(scope="session")
def makers_list():
    return [
        MapDatasetMaker(),
        SafeMaskMaker(methods=["offset-max"], offset_max="2 deg"),
        FoVBackgroundMaker(method="scale"),
    ]


@requires_data()
@pytest.mark.parametrize(
    "pars",
    [
        {
            "dataset": get_mapdataset(name="linear_staking"),
            "stacking": True,
            "cutout_width": None,
            "write_all": False,
            "n_jobs": None,
        },
        {
            "dataset": get_mapdataset(name="parallel"),
            "stacking": False,
            "cutout_width": None,
            "write_all": True,
            "n_jobs": 2,
        },
        {
            "dataset": get_mapdataset(name="parallel_staking"),
            "stacking": True,
            "cutout_width": None,
            "write_all": False,
            "n_jobs": 2,
        },
    ],
)
@requires_data()
def test_makers(pars, observations, makers_list, tmp_path):
    makers = DatasetsMaker(
        makers_list,
        pars["dataset"],
        stacking=pars["stacking"],
        cutout_mode="partial",
        cutout_width=pars["cutout_width"],
        n_jobs=pars["n_jobs"],
        path=tmp_path,
        write_all=pars["write_all"],
        overwrite=True,
    )

    datasets = makers.run(observations)

    if len(datasets) == 0:
        counts = datasets[0].counts
        assert counts.unit == ""
        assert_allclose(counts.data.sum(), 46716, rtol=1e-5)

        exposure = datasets[0].exposure
        assert exposure.unit == "m2 s"
        assert_allclose(exposure.data.mean(), 1.350841e09, rtol=3e-3)


@requires_data()
def test_makers_cutout_width(observations, makers_list, tmp_path):

    makers = DatasetsMaker(
        makers_list,
        get_mapdataset(name="linear_staking_1deg"),
        stacking=True,
        cutout_mode="partial",
        cutout_width="5 deg",
        n_jobs=None,
        path=tmp_path,
        write_all=False,
        overwrite=True,
    )
    datasets = makers.run(observations)

    counts = datasets[0].counts
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), 46716, rtol=1e-5)

    exposure = datasets[0].exposure
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.mean(), 1.350841e09, rtol=3e-3)
