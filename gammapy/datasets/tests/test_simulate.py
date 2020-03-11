# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gammapy.datasets import MapDatasetEventSampler
from gammapy.data import GTI, Observation
from gammapy.datasets.tests.test_map import get_map_dataset
from gammapy.irf import load_cta_irfs
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    LightCurveTemplateTemporalModel,
)
from gammapy.utils.testing import requires_data


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

    t_min = 0 * u.s
    t_max = 1000 * u.s

    time = np.arange(t_max.value) * u.s
    tau = u.Quantity("2e2 s")
    norm = np.exp(-time / tau)

    table = Table()
    table["TIME"] = time
    table["NORM"] = norm / norm.max()
    t_ref = Time("2000-01-01")
    table.meta = dict(MJDREFI=t_ref.mjd, MJDREFF=0, TIMEUNIT="s")
    temporal_model = LightCurveTemplateTemporalModel(table)

    skymodel = SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        temporal_model=temporal_model,
    )

    geom = WcsGeom.create(
        skydir=position, binsz=1, width="5 deg", frame="galactic", axes=[energy_axis]
    )

    gti = GTI.create(start=t_min, stop=t_max, reference_time=t_ref)

    geom_true = geom.copy()
    geom_true.axes[0].name = "energy_true"

    dataset = get_map_dataset(
        sky_model=skymodel, geom=geom, geom_etrue=geom_true, edisp=True
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

    assert_allclose(events.table["TIME"][0], 94.7121239, rtol=1e-5)
    assert events.table["TIME"].unit == "s"

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
    assert_allclose(events.table["ENERGY_TRUE"][0], 2.2450239, rtol=1e-5)
    assert events.table["ENERGY_TRUE"].unit == "TeV"

    assert_allclose(events.table["RA"][0], 266.88654311, rtol=1e-5)
    assert events.table["RA"].unit == "deg"

    assert_allclose(events.table["DEC"][0], -29.01084895, rtol=1e-5)
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
    livetime = 1.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001, pointing=pointing, livetime=livetime, irfs=irfs
    )

    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    dataset_bkg = dataset.copy()
    dataset_bkg.models = dataset_bkg.models[1]
    events_bkg = sampler.run(dataset=dataset_bkg, observation=obs)

    assert len(events.table) == 2423
    assert_allclose(events.table["ENERGY"][0], 3.582666040117894, rtol=1e-5)
    assert_allclose(events.table["RA"][0], 263.876666324552, rtol=1e-5)
    assert_allclose(events.table["DEC"][0], -28.72531131917506, rtol=1e-5)

    assert len(events_bkg.table) == 16
    assert_allclose(events_bkg.table["ENERGY"][0], 2.874495158620, rtol=1e-5)
    assert_allclose(events_bkg.table["RA"][0], 264.56394364251, rtol=1e-5)
    assert_allclose(events_bkg.table["DEC"][0], -28.676648107, rtol=1e-5)
    assert_allclose(events_bkg.table["MC_ID"][0], 0, rtol=1e-5)

    meta = events.table.meta

    assert meta["RA_PNT"] == 266.4049882865447
    assert meta["ONTIME"] == 3600.0
    assert meta["OBS_ID"] == 1001
    assert meta["RADECSYS"] == "icrs"


@requires_data()
def test_mde_run_switchoff(dataset):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 1.0 * u.hr
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
    assert_allclose(events.table["ENERGY"][0], 2.2450239000119323, rtol=1e-5)
    assert_allclose(events.table["RA"][0], 266.9128884464542, rtol=1e-5)
    assert_allclose(events.table["DEC"][0], -29.034641131874313, rtol=1e-5)

    meta = events.table.meta

    assert meta["RA_PNT"] == 266.4049882865447
    assert meta["ONTIME"] == 3600.0
    assert meta["OBS_ID"] == 1001
    assert meta["RADECSYS"] == "icrs"
