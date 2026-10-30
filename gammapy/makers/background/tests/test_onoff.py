# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.datasets import (
    MapDataset,
    MapDatasetOnOff,
    SpectrumDatasetOnOff,
)
from gammapy.makers import (
    MapDatasetMaker,
    OnOffBackgroundMaker,
    check_run_pair_validity,
)
from gammapy.maps import MapAxis, WcsGeom
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def observations():
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    on = datastore.obs(23523)  # Crab
    off = datastore.obs(26850)  # empty-field OFF run
    return on, off


@pytest.fixture
def energy_axis():
    return MapAxis.from_energy_bounds("0.1 TeV", "40 TeV", 8, name="energy")


@pytest.fixture
def map_dataset(energy_axis, observations):
    on, _ = observations
    geom = WcsGeom.create(
        skydir=(83.633, 22.014),
        binsz=0.05,
        width=(4, 4),
        frame="icrs",
        axes=[energy_axis],
    )
    empty = MapDataset.create(geom=geom, name="23523")
    return MapDatasetMaker(selection=["counts", "exposure", "psf", "edisp"]).run(
        empty, on
    )


@pytest.fixture
def spectrum_dataset(energy_axis, map_dataset):
    region = CircleSkyRegion(SkyCoord(83.633, 22.014, unit="deg"), 0.11 * u.deg)
    return map_dataset.to_spectrum_dataset(region)


def test_init_invalid_method():
    with pytest.raises(ValueError):
        OnOffBackgroundMaker(acceptance_method="xx")


@requires_data()
def test_run_map_dataset(map_dataset, observations):
    on, off = observations
    result = OnOffBackgroundMaker().run(map_dataset, on, off)

    assert isinstance(result, MapDatasetOnOff)
    assert np.allclose(result.counts.data, map_dataset.counts.data)
    assert result.counts_off.data.sum() == 7662
    assert result.counts_off.geom == result.counts.geom
    assert_allclose(result.background.data.sum(), 7396.48, atol=1e-1)
    assert_allclose(result.alpha.data.sum(), 49425.7, rtol=1e-4)
    assert_allclose(result.acceptance_off.data[2][2][2], 1638.5, atol=1e-2)

    assert result.psf is not None
    assert result.edisp is not None

    on, off = observations
    result = OnOffBackgroundMaker(acceptance_method="counts-ratio").run(
        map_dataset, on, off
    )

    assert isinstance(result, MapDatasetOnOff)
    assert np.allclose(result.counts.data, map_dataset.counts.data)
    assert result.counts_off.data.sum() == 7662
    assert result.counts_off.geom == result.counts.geom
    assert_allclose(result.background.data.sum(), 5505, atol=1e-1)
    assert result.psf is not None
    assert result.edisp is not None


@requires_data()
def test_run_spectrum_dataset(spectrum_dataset, observations):
    on, off = observations
    result = OnOffBackgroundMaker().run(spectrum_dataset, on, off)

    assert isinstance(result, SpectrumDatasetOnOff)
    assert result.counts_off.data.sum() == 40
    assert result.counts_off.geom == result.counts.geom
    assert_allclose(result.background.data.sum(), 38.61, atol=1e-1)
    assert result.acceptance.geom == result.counts.geom
    assert_allclose(result.alpha.data[0], 0.965, atol=1e-2)
