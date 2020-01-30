# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.cube import MapDataset, MapDatasetEventSampler, simulate_dataset
from gammapy.cube.tests.test_edisp_map import make_edisp_map_test
from gammapy.cube.tests.test_fit import get_map_dataset
from gammapy.cube.tests.test_psf_map import make_test_psfmap
from gammapy.data import GTI, Observation
from gammapy.irf import load_cta_irfs
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    Models,
)
from gammapy.utils.testing import requires_data


@requires_data()
def test_simulate():
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    # Define sky model to simulate the data
    spatial_model = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    sky_model_simu = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model
    )

    # Define map geometry
    axis = MapAxis.from_edges(
        np.logspace(-1, 1.0, 20), unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.025, width=(1, 1), frame="galactic", axes=[axis]
    )

    # Define some observation parameters
    pointing = SkyCoord(0 * u.deg, 0 * u.deg, frame="galactic")

    dataset = simulate_dataset(
        sky_model_simu, geom, pointing, irfs, livetime=10 * u.h, random_state=42
    )

    assert isinstance(dataset, MapDataset)
    assert isinstance(dataset.models, Models)

    assert dataset.counts.data.dtype is np.dtype("int")
    assert_allclose(dataset.counts.data[5, 20, 20], 2)
    assert_allclose(dataset.exposure.data[5, 20, 20], 16122681486.381285)
    assert_allclose(
        dataset.background_model.map.data[5, 20, 20], 0.9765545345855245, rtol=1e-5
    )
    assert_allclose(dataset.psf.psf_map.data[5, 5, 0, 0], 91987.862)
    assert_allclose(dataset.edisp.data.data[10, 10], 0.864096, rtol=1e-5)


@pytest.fixture(scope="session")
def dataset():
    position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=3, unit="TeV", name="energy", interp="log"
    )

    spatial_model = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic"
    )

    spectral_model = PowerLawSpectralModel(amplitude="1e-11 cm-2 s-1 TeV-1")
    skymodel = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    geom = WcsGeom.create(
        skydir=position, binsz=1, width="5 deg", frame="galactic", axes=[energy_axis]
    )

    t_min = 0 * u.s
    t_max = 30000 * u.s

    gti = GTI.create(start=t_min, stop=t_max)

    dataset = get_map_dataset(
        sky_model=skymodel, geom=geom, geom_etrue=geom, edisp=True
    )
    dataset.gti = gti

    return dataset


@requires_data()
def test_mde_sample_sources(dataset):
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_sources(dataset=dataset)

    assert len(events.table["ENERGY_TRUE"]) == 2407
    assert_allclose(events.table["ENERGY_TRUE"][0], 2.245024, rtol=1e-5)
    assert events.table["ENERGY_TRUE"].unit == "TeV"

    assert_allclose(events.table["RA_TRUE"][0], 266.912888, rtol=1e-5)
    assert events.table["RA_TRUE"].unit == "deg"

    assert_allclose(events.table["DEC_TRUE"][0], -29.034641, rtol=1e-5)
    assert events.table["DEC_TRUE"].unit == "deg"

    assert_allclose(events.table["MC_ID"][0], 1, rtol=1e-5)


@requires_data()
def test_mde_sample_background(dataset):
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_background(dataset=dataset)

    assert len(events.table["ENERGY"]) == 15
    assert_allclose(events.table["ENERGY"][0], 1.894698, rtol=1e-5)
    assert events.table["ENERGY"].unit == "TeV"

    assert_allclose(events.table["RA"][0], 266.454448, rtol=1e-5)
    assert events.table["RA"].unit == "deg"

    assert_allclose(events.table["DEC"][0], -30.870316, rtol=1e-5)
    assert events.table["DEC"].unit == "deg"

    assert_allclose(events.table["MC_ID"][0], 0, rtol=1e-5)


@requires_data()
def test_mde_sample_psf(dataset):
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_sources(dataset=dataset)
    events = sampler.sample_psf(dataset.psf, events)

    assert len(events.table) == 2407
    assert_allclose(events.table["ENERGY_TRUE"][0], 2.245024, rtol=1e-5)
    assert events.table["ENERGY_TRUE"].unit == "TeV"

    assert_allclose(events.table["RA"][0], 266.909362, rtol=1e-5)
    assert events.table["RA"].unit == "deg"

    assert_allclose(events.table["DEC"][0], -29.039877, rtol=1e-5)
    assert events.table["DEC"].unit == "deg"


@requires_data()
def test_mde_sample_edisp(dataset):
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_sources(dataset=dataset)
    events = sampler.sample_edisp(dataset.edisp, events)

    assert len(events.table) == 2407
    assert_allclose(events.table["ENERGY"][0], 2.24502, rtol=1e-5)
    assert events.table["ENERGY"].unit == "TeV"

    assert_allclose(events.table["RA_TRUE"][0], 266.912888, rtol=1e-5)
    assert events.table["RA_TRUE"].unit == "deg"

    assert_allclose(events.table["DEC_TRUE"][0], -29.034641, rtol=1e-5)
    assert events.table["DEC_TRUE"].unit == "deg"

    assert_allclose(events.table["MC_ID"][0], 1, rtol=1e-5)


@requires_data()
def test_mde_run(dataset):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 10.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001, pointing=pointing, livetime=livetime, irfs=irfs
    )

    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    assert len(events.table) == 2422
    assert_allclose(events.table["ENERGY"][0], 1.56446303986587, rtol=1e-5)
    assert_allclose(events.table["RA"][0], 268.8180057255861, rtol=1e-5)
    assert_allclose(events.table["DEC"][0], -28.45051813404372, rtol=1e-5)

    meta = events.table.meta

    assert meta["RA_PNT"] == 266.4049882865447
    assert meta["ONTIME"] == 36000.0
    assert meta["OBS_ID"] == 1001
    assert meta["RADECSYS"] == "icrs"


def test_mde_run_switchoff(dataset):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 10.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001, pointing=pointing, livetime=livetime, irfs=irfs
    )

    dataset.psf = None
    dataset.edisp = None
    dataset.background_model = None

    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    assert len(events.table) == 2407
    assert_allclose(events.table["ENERGY_TRUE"][0], 2.2450239000119323, rtol=1e-5)
    assert_allclose(events.table["RA_TRUE"][0], 266.9128884464542, rtol=1e-5)
    assert_allclose(events.table["DEC_TRUE"][0], -29.034641131874313, rtol=1e-5)

    meta = events.table.meta

    assert meta["RA_PNT"] == 266.4049882865447
    assert meta["ONTIME"] == 36000.0
    assert meta["OBS_ID"] == 1001
    assert meta["RADECSYS"] == "icrs"
