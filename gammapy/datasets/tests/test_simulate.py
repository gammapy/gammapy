# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from gammapy.data import GTI, DataStore, Observation
from gammapy.datasets import MapDatasetEventSampler
from gammapy.datasets.tests.test_map import get_map_dataset
from gammapy.irf import load_cta_irfs
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    FoVBackgroundModel,
    GaussianSpatialModel,
    LightCurveTemplateTemporalModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data


@pytest.fixture()
def models():
    spatial_model = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic"
    )

    spectral_model = PowerLawSpectralModel(amplitude="1e-11 cm-2 s-1 TeV-1")

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

    model = SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        temporal_model=temporal_model,
        name="test-source",
    )

    bkg_model = FoVBackgroundModel(dataset_name="test")
    return [model, bkg_model]


@pytest.fixture(scope="session")
def dataset():
    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=3, unit="TeV", name="energy", interp="log"
    )

    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.05, width="5 deg", frame="galactic", axes=[energy_axis]
    )

    geom_true = geom.copy()
    geom_true.axes[0].name = "energy_true"

    dataset = get_map_dataset(
        geom=geom, geom_etrue=geom_true, edisp="edispmap", name="test"
    )
    dataset.background /= 400

    dataset.gti = GTI.create(
        start=0 * u.s, stop=1000 * u.s, reference_time="2000-01-01"
    )

    return dataset


@requires_data()
def test_mde_sample_sources(dataset, models):
    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_sources(dataset=dataset)

    assert len(events.table["ENERGY_TRUE"]) == 88
    assert_allclose(events.table["ENERGY_TRUE"][0], 2.751205, rtol=1e-5)
    assert events.table["ENERGY_TRUE"].unit == "TeV"

    assert_allclose(events.table["RA_TRUE"][0], 266.559566, rtol=1e-5)
    assert events.table["RA_TRUE"].unit == "deg"

    assert_allclose(events.table["DEC_TRUE"][0], -28.742429, rtol=1e-5)
    assert events.table["DEC_TRUE"].unit == "deg"

    assert_allclose(events.table["TIME"][0], 461.424967, rtol=1e-5)
    assert events.table["TIME"].unit == "s"

    assert_allclose(events.table["MC_ID"][0], 1, rtol=1e-5)


@requires_data()
def test_mde_sample_weak_src(dataset, models):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 10.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001, pointing=pointing, livetime=livetime, irfs=irfs
    )

    models[0].parameters["amplitude"].value = 1e-25

    dataset.models = models

    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    assert len(events.table) == 18
    assert_allclose(
        len(np.where(events.table["MC_ID"] == 0)[0]), len(events.table), rtol=1e-5
    )


@requires_data()
def test_mde_sample_background(dataset, models):
    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_background(dataset=dataset)

    assert len(events.table["ENERGY"]) == 15
    assert_allclose(events.table["ENERGY"][0], 1.894698, rtol=1e-5)
    assert events.table["ENERGY"].unit == "TeV"

    assert_allclose(events.table["RA"][0], 266.571824, rtol=1e-5)
    assert events.table["RA"].unit == "deg"

    assert_allclose(events.table["DEC"][0], -27.979152, rtol=1e-5)
    assert events.table["DEC"].unit == "deg"

    assert events.table["DEC_TRUE"][0] == events.table["DEC"][0]

    assert_allclose(events.table["MC_ID"][0], 0, rtol=1e-5)


@requires_data()
def test_mde_sample_psf(dataset, models):
    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_sources(dataset=dataset)
    events = sampler.sample_psf(dataset.psf, events)

    assert len(events.table) == 88
    assert_allclose(events.table["ENERGY_TRUE"][0], 2.751205, rtol=1e-5)
    assert events.table["ENERGY_TRUE"].unit == "TeV"

    assert_allclose(events.table["RA"][0], 266.556053, rtol=1e-5)
    assert events.table["RA"].unit == "deg"

    assert_allclose(events.table["DEC"][0], -28.746459, rtol=1e-5)
    assert events.table["DEC"].unit == "deg"


@requires_data()
def test_mde_sample_edisp(dataset, models):
    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_sources(dataset=dataset)
    events = sampler.sample_edisp(dataset.edisp, events)

    assert len(events.table) == 88
    assert_allclose(events.table["ENERGY"][0], 2.751205, rtol=1e-5)
    assert events.table["ENERGY"].unit == "TeV"

    assert_allclose(events.table["RA_TRUE"][0], 266.559566, rtol=1e-5)
    assert events.table["RA_TRUE"].unit == "deg"

    assert_allclose(events.table["DEC_TRUE"][0], -28.742429, rtol=1e-5)
    assert events.table["DEC_TRUE"].unit == "deg"

    assert_allclose(events.table["MC_ID"][0], 1, rtol=1e-5)


@requires_data()
def test_event_det_coords(dataset, models):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 1.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001, pointing=pointing, livetime=livetime, irfs=irfs
    )

    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    assert len(events.table) == 102
    assert_allclose(events.table["DETX"][0], -2.269308, rtol=1e-5)
    assert events.table["DETX"].unit == "deg"

    assert_allclose(events.table["DETY"][0], -1.391967, rtol=1e-5)
    assert events.table["DETY"].unit == "deg"


@requires_data()
def test_mde_run(dataset, models):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 1.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001, pointing=pointing, livetime=livetime, irfs=irfs
    )

    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    dataset_bkg = dataset.copy(name="new-dataset")
    dataset_bkg.models = [FoVBackgroundModel(dataset_name=dataset_bkg.name)]

    events_bkg = sampler.run(dataset=dataset_bkg, observation=obs)

    assert len(events.table) == 102
    assert_allclose(events.table["ENERGY"][0], 5.792375, rtol=1e-5)
    assert_allclose(events.table["RA"][0], 263.777097, rtol=1e-5)
    assert_allclose(events.table["DEC"][0], -30.302968, rtol=1e-5)

    assert len(events_bkg.table) == 16
    assert_allclose(events_bkg.table["ENERGY"][0], 4.014328, rtol=1e-5)
    assert_allclose(events_bkg.table["RA"][0], 267.488623, rtol=1e-5)
    assert_allclose(events_bkg.table["DEC"][0], -30.924333, rtol=1e-5)
    assert_allclose(events_bkg.table["MC_ID"][0], 0, rtol=1e-5)

    meta = events.table.meta

    assert meta["HDUCLAS1"] == "EVENTS"
    assert meta["EXTNAME"] == "EVENTS"
    assert (
        meta["HDUDOC"]
        == "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
    )
    assert meta["HDUVERS"] == "0.2"
    assert meta["HDUCLASS"] == "GADF"
    assert meta["OBS_ID"] == 1001
    assert_allclose(meta["TSTART"], 0.0)
    assert_allclose(meta["TSTOP"], 3600.0)
    assert_allclose(meta["ONTIME"], 3600.0)
    assert_allclose(meta["LIVETIME"], 3600.0)
    assert_allclose(meta["DEADC"], 1.0)
    assert_allclose(meta["RA_PNT"], 266.4049882865447)
    assert_allclose(meta["DEC_PNT"], -28.936177761791473)
    assert meta["EQUINOX"] == "J2000"
    assert meta["RADECSYS"] == "icrs"
    assert "Gammapy" in meta["CREATOR"]
    assert meta["EUNIT"] == "TeV"
    assert meta["EVTVER"] == ""
    assert meta["OBSERVER"] == "Gammapy user"
    assert meta["DSTYP1"] == "TIME"
    assert meta["DSUNI1"] == "s"
    assert meta["DSVAL1"] == "TABLE"
    assert meta["DSREF1"] == ":GTI"
    assert meta["DSTYP2"] == "ENERGY"
    assert meta["DSUNI2"] == "TeV"
    assert ":" in meta["DSVAL2"]
    assert meta["DSTYP3"] == "POS(RA,DEC)     "
    assert "CIRCLE" in meta["DSVAL3"]
    assert meta["DSUNI3"] == "deg             "
    assert meta["NDSKEYS"] == " 3 "
    assert_allclose(meta["RA_OBJ"], 266.4049882865447)
    assert_allclose(meta["DEC_OBJ"], -28.936177761791473)
    assert_allclose(meta["TELAPSE"], 1000.0)
    assert_allclose(meta["MJDREFI"], 51544)
    assert_allclose(meta["MJDREFF"], 0.0007428703684126958)
    assert meta["TIMEUNIT"] == "s"
    assert meta["TIMESYS"] == "tt"
    assert meta["TIMEREF"] == "LOCAL"
    assert meta["DATE-OBS"] == "2000-01-01"
    assert meta["DATE-END"] == "2000-01-01"
    assert meta["CONV_DEP"] == 0
    assert meta["CONV_RA"] == 0
    assert meta["CONV_DEC"] == 0
    assert meta["MID00001"] == 1
    assert meta["MID00002"] == 2
    assert meta["NMCIDS"] == 2
    assert_allclose(float(meta["ALT_PNT"]), float("-13.5345076464"), rtol=1e-7)
    assert_allclose(float(meta["AZ_PNT"]), float("228.82981620065763"), rtol=1e-7)
    assert meta["ORIGIN"] == "Gammapy"
    assert meta["TELESCOP"] == "CTA"
    assert meta["INSTRUME"] == "1DC"
    assert meta["N_TELS"] == ""
    assert meta["TELLIST"] == ""


@requires_data()
def test_irf_alpha_config(dataset, models):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-caldb/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz"
    )
    livetime = 1.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001, pointing=pointing, livetime=livetime, irfs=irfs
    )

    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)
    assert events is not None


@requires_data()
def test_mde_run_switchoff(dataset, models):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 1.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001, pointing=pointing, livetime=livetime, irfs=irfs
    )

    dataset.models = models

    dataset.psf = None
    dataset.edisp = None
    dataset.background = None

    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    assert len(events.table) == 88
    assert_allclose(events.table["ENERGY"][0], 2.751205, rtol=1e-5)
    assert_allclose(events.table["RA"][0], 266.559566, rtol=1e-5)
    assert_allclose(events.table["DEC"][0], -28.742429, rtol=1e-5)

    meta = events.table.meta

    assert meta["RA_PNT"] == 266.4049882865447
    assert meta["ONTIME"] == 3600.0
    assert meta["OBS_ID"] == 1001
    assert meta["RADECSYS"] == "icrs"


@requires_data()
def test_events_datastore(tmp_path, dataset, models):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 10.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001, pointing=pointing, livetime=livetime, irfs=irfs
    )

    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    primary_hdu = fits.PrimaryHDU()
    hdu_evt = fits.BinTableHDU(events.table)
    hdu_gti = fits.BinTableHDU(dataset.gti.table, name="GTI")
    hdu_all = fits.HDUList([primary_hdu, hdu_evt, hdu_gti])
    hdu_all.writeto(str(tmp_path / "events.fits"))

    DataStore.from_events_files([str(tmp_path / "events.fits")])
