# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.spectrum import SpectrumDatasetMaker


@pytest.fixture
def observations():
    """Example observation list for testing."""
    datastore = DataStore.from_file(
        "$GAMMAPY_DATA/hess-dl3-dr1/hess-dl3-dr3-with-background.fits.gz"
    )
    obs_ids = [23523, 23526]
    return datastore.get_observations(obs_ids)


@pytest.fixture()
def spectrum_dataset_maker():
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    radius = Angle(0.11, "deg")
    region = CircleSkyRegion(pos, radius)
    e_reco = np.logspace(0, 2, 5) * u.TeV
    e_true = np.logspace(-0.5, 2, 11) * u.TeV
    return SpectrumDatasetMaker(region=region, e_reco=e_reco, e_true=e_true)


def test_spectrum_dataset_maker(spectrum_dataset_maker, observations):
    datasets = []

    for obs in observations:
        dataset = spectrum_dataset_maker.run(obs)
        datasets.append(dataset)

    assert_allclose(datasets[0].counts.data.sum(), 100)
    assert_allclose(datasets[1].counts.data.sum(), 92)

    assert_allclose(datasets[0].livetime.value, 1581.736758)
    assert_allclose(datasets[1].livetime.value, 1572.686724)

    assert_allclose(datasets[0].background.data.sum(), 1.754928, rtol=1e-5)
    assert_allclose(datasets[1].background.data.sum(), 1.759318, rtol=1e-5)
