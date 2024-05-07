# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion, PointSkyRegion
from gammapy.data import DataStore, EventList
from gammapy.datasets import MapDataset, SpectrumDataset
from gammapy.makers import MapDatasetMaker, PhaseBackgroundMaker, SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.utils.testing import requires_data

POS_CRAB = SkyCoord(83.6331, +22.0145, frame="icrs", unit="deg")
POS_VELA = SkyCoord("08h35m20.65525s", "-45d10m35.1545s", frame="icrs")


@pytest.fixture(scope="session")
def observations_cta():
    datastore_cta = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")
    return datastore_cta.get_observations([111630])


@pytest.fixture(scope="session")
def observations_magic():
    datastore_magic = DataStore.from_dir("$GAMMAPY_DATA/magic/rad_max/data")
    return datastore_magic.get_observations(required_irf="point-like")[0]


@pytest.fixture(scope="session")
def observations_hess():
    datastore_hess = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    return datastore_hess.obs(23523)


@pytest.fixture(scope="session")
def phase_bkg_maker():
    """Example background estimator for testing."""
    return PhaseBackgroundMaker(on_phase=(0.5, 0.6), off_phase=(0.7, 1))


@requires_data()
def test_basic(phase_bkg_maker):
    assert "PhaseBackgroundMaker" in str(phase_bkg_maker)


@requires_data()
def test_run_spectrum(observations_cta, phase_bkg_maker):

    maker = SpectrumDatasetMaker()

    e_reco = MapAxis.from_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")
    e_true = MapAxis.from_edges(np.logspace(-0.5, 2, 11) * u.TeV, name="energy_true")

    radius = Angle(0.2, "deg")
    region = CircleSkyRegion(POS_VELA, radius)

    geom = RegionGeom.create(region=region, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=e_true)

    obs = observations_cta[0]
    dataset = maker.run(dataset_empty, obs)
    dataset_on_off = phase_bkg_maker.run(dataset, obs)

    assert_allclose(dataset_on_off.acceptance, 0.1)
    assert_allclose(dataset_on_off.acceptance_off, 0.3)

    assert_allclose(dataset_on_off.counts.data.sum(), 28)
    assert_allclose(dataset_on_off.counts_off.data.sum(), 57)


@requires_data()
def test_run_map(observations_cta, phase_bkg_maker):

    maker = MapDatasetMaker()

    e_reco = MapAxis.from_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")
    e_true = MapAxis.from_edges(np.logspace(-0.5, 2, 11) * u.TeV, name="energy_true")

    binsz = Angle(0.02, "deg")
    geom = WcsGeom.create(binsz=binsz, skydir=POS_VELA, width="2 deg", axes=[e_reco])
    dataset_empty = MapDataset.create(geom=geom, energy_axis_true=e_true)

    obs = observations_cta[0]
    dataset = maker.run(dataset_empty, obs)
    dataset_on_off = phase_bkg_maker.run(dataset, obs)

    assert_allclose(dataset_on_off.acceptance, 0.1)
    assert_allclose(dataset_on_off.acceptance_off, 0.3)

    assert_allclose(dataset_on_off.counts.data.sum(), 78)
    assert_allclose(dataset_on_off.counts_off.data.sum(), 263)


@pytest.mark.parametrize(
    "pars",
    [
        {"p_in": [[0.2, 0.3]], "p_out": [[0.2, 0.3]]},
        {"p_in": [[0.9, 0.1]], "p_out": [[0.9, 1], [0, 0.1]]},
        {"p_in": [[-0.2, 0.1]], "p_out": [[0.8, 1], [0, 0.1]]},
        {"p_in": [[0.8, 1.2]], "p_out": [[0.8, 1], [0, 0.2]]},
        {"p_in": [[0.8, 1]], "p_out": [[0.8, 1]]},
        {"p_in": [[0.2, 0.4], [0.8, 0.9]], "p_out": [[0.2, 0.4], [0.8, 0.9]]},
    ],
)
def test_check_phase_intervals(pars):
    assert_allclose(
        PhaseBackgroundMaker._check_intervals(pars["p_in"]), pars["p_out"], rtol=1e-5
    )


def test_copy_interval():
    on = (0.8, 1.2)
    off = [(0.1, 0.3), (0.4, 0.5)]
    PhaseBackgroundMaker(on_phase=on, off_phase=off)
    assert on == (0.8, 1.2)
    assert off == [(0.1, 0.3), (0.4, 0.5)]


@requires_data()
@pytest.mark.parametrize(
    "pars",
    [
        {
            "p_in": ["observations_magic", PointSkyRegion(POS_CRAB)],
            "p_out": [
                np.array([58, 19, 2, 2, 0, 0]),
                np.array([163, 46, 15, 1, 0, 0]),
            ],
        },
        {
            "p_in": ["observations_magic", CircleSkyRegion(POS_CRAB, 0.1 * u.deg)],
            "p_out": [
                np.array([23, 18, 2, 2, 0, 0]),
                np.array([51, 35, 15, 1, 0, 0]),
            ],
        },
        {
            "p_in": ["observations_hess", CircleSkyRegion(POS_CRAB, 0.1 * u.deg)],
            "p_out": [
                np.array([0, 1, 7, 4, 0, 0]),
                np.array([0, 9, 32, 12, 1, 0]),
            ],
        },
    ],
)
def test_make_counts(phase_bkg_maker, pars, request):

    maker = SpectrumDatasetMaker(
        containment_correction=False, selection=["counts", "exposure", "edisp"]
    )

    e_reco = MapAxis.from_energy_bounds(0.05, 100, nbin=6, unit="TeV", name="energy")
    e_true = MapAxis.from_energy_bounds(
        0.01, 300, nbin=15, unit="TeV", name="energy_true"
    )

    obs = request.getfixturevalue(pars["p_in"][0])
    table = obs.events.table
    table["PHASE"] = np.linspace(0, 1, len(table["TIME"]))
    obs._events = EventList(table)

    on_region = pars["p_in"][1]

    geom = RegionGeom.create(region=on_region, axes=[e_reco])
    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=e_true)

    dataset = maker.run(dataset_empty, obs)
    dataset_on_off = phase_bkg_maker.run(dataset, obs)

    assert_allclose(
        [
            np.squeeze(dataset_on_off.counts.data),
            np.squeeze(dataset_on_off.counts_off.data),
        ],
        pars["p_out"],
    )
