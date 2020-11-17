# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from gammapy.data import DataStore
from gammapy.datasets import SpectrumDataset
from gammapy.makers import PhaseBackgroundMaker, SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom
from gammapy.utils.regions import SphericalCircleSkyRegion
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def observations():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")
    return datastore.get_observations([111630])


@pytest.fixture(scope="session")
def phase_bkg_maker():
    """Example background estimator for testing."""
    return PhaseBackgroundMaker(on_phase=(0.5, 0.6), off_phase=(0.7, 1))


@requires_data()
def test_basic(phase_bkg_maker):
    assert "PhaseBackgroundMaker" in str(phase_bkg_maker)


@requires_data()
def test_run(observations, phase_bkg_maker):

    maker = SpectrumDatasetMaker()

    e_reco = MapAxis.from_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")
    e_true = MapAxis.from_edges(np.logspace(-0.5, 2, 11) * u.TeV, name="energy_true")

    pos = SkyCoord("08h35m20.65525s", "-45d10m35.1545s", frame="icrs")
    radius = Angle(0.2, "deg")
    region = SphericalCircleSkyRegion(pos, radius)

    geom = RegionGeom.create(region=region, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=e_true)

    obs = observations["111630"]
    dataset = maker.run(dataset_empty, obs)
    dataset_on_off = phase_bkg_maker.run(dataset, obs)

    assert_allclose(dataset_on_off.acceptance, 0.1)
    assert_allclose(dataset_on_off.acceptance_off, 0.3)

    assert_allclose(dataset_on_off.counts.data.sum(), 28)
    assert_allclose(dataset_on_off.counts_off.data.sum(), 57)


@pytest.mark.parametrize(
    "pars",
    [
        {"p_in": [[0.2, 0.3]], "p_out": [[0.2, 0.3]]},
        {"p_in": [[0.9, 0.1]], "p_out": [[0.9, 1], [0, 0.1]]},
    ],
)
def test_check_phase_intervals(pars):
    assert PhaseBackgroundMaker._check_intervals(pars["p_in"]) == pars["p_out"]
