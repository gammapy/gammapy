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


def get_model():
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

    return SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        temporal_model=temporal_model,
        name="test-source",
    )


@pytest.fixture(scope="session")
def dataset():
    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=3, unit="TeV", name="energy", interp="log"
    )

    geom = WcsGeom.create(
        skydir=(0, 0), binsz=1, width="5 deg", frame="galactic", axes=[energy_axis]
    )

    geom_true = geom.copy()
    geom_true.axes[0].name = "energy_true"

    dataset = get_map_dataset(geom=geom, geom_etrue=geom_true, edisp="edispmap")

    dataset.gti = GTI.create(
        start=0 * u.s, stop=1000 * u.s, reference_time="2000-01-01"
    )

    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = [get_model(), bkg_model]
    return dataset


@requires_data()
def test_mde_sample_sources(dataset):
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_sources(dataset=dataset)

    assert len(events.table["ENERGY_TRUE"]) == 348
    assert_allclose(events.table["ENERGY_TRUE"][0], 3.536090852832, rtol=1e-5)
    assert events.table["ENERGY_TRUE"].unit == "TeV"

    assert_allclose(events.table["RA_TRUE"][0], 266.296592804, rtol=1e-5)
    assert events.table["RA_TRUE"].unit == "deg"

    assert_allclose(events.table["DEC_TRUE"][0], -29.056521954, rtol=1e-5)
    assert events.table["DEC_TRUE"].unit == "deg"

    assert_allclose(events.table["TIME"][0], 32.73797244764, rtol=1e-5)
    assert events.table["TIME"].unit == "s"

    assert_allclose(events.table["MC_ID"][0], 1, rtol=1e-5)


@requires_data()
def test_mde_sample_weak_src(dataset):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 10.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001, pointing=pointing, livetime=livetime, irfs=irfs
    )

    dataset_weak_src = dataset.copy()
    dataset_weak_src.models[0].parameters["amplitude"].value = 1e-25

    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset_weak_src, observation=obs)

    assert len(events.table) == 18
    assert_allclose(
        len(np.where(events.table["MC_ID"] == 0)[0]), len(events.table), rtol=1e-5
    )


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

    assert events.table["DEC_TRUE"][0] == events.table["DEC"][0]

    assert_allclose(events.table["MC_ID"][0], 0, rtol=1e-5)


@requires_data()
def test_mde_sample_psf(dataset):
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_sources(dataset=dataset)
    events = sampler.sample_psf(dataset.psf, events)

    assert len(events.table) == 348
    assert_allclose(events.table["ENERGY_TRUE"][0], 3.536090852832, rtol=1e-5)
    assert events.table["ENERGY_TRUE"].unit == "TeV"

    assert_allclose(events.table["RA"][0], 266.2757792220, rtol=1e-5)
    assert events.table["RA"].unit == "deg"

    assert_allclose(events.table["DEC"][0], -29.030939, rtol=1e-5)
    assert events.table["DEC"].unit == "deg"


@requires_data()
def test_mde_sample_edisp(dataset):
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_sources(dataset=dataset)
    events = sampler.sample_edisp(dataset.edisp, events)

    assert len(events.table) == 348
    assert_allclose(events.table["ENERGY"][0], 3.53609085283, rtol=1e-5)
    assert events.table["ENERGY"].unit == "TeV"

    assert_allclose(events.table["RA_TRUE"][0], 266.296592804, rtol=1e-5)
    assert events.table["RA_TRUE"].unit == "deg"

    assert_allclose(events.table["DEC_TRUE"][0], -29.05652195, rtol=1e-5)
    assert events.table["DEC_TRUE"].unit == "deg"

    assert_allclose(events.table["MC_ID"][0], 1, rtol=1e-5)


@requires_data()
def test_event_det_coords(dataset):
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

    assert len(events.table) == 374
    assert_allclose(events.table["DETX"][0], -2.44563584, rtol=1e-5)
    assert events.table["DETX"].unit == "deg"

    assert_allclose(events.table["DETY"][0], 0.01414569, rtol=1e-5)
    assert events.table["DETY"].unit == "deg"


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

    dataset_bkg = dataset.copy(name="new-dataset")
    dataset_bkg.models = [FoVBackgroundModel(dataset_name=dataset_bkg.name)]

    events_bkg = sampler.run(dataset=dataset_bkg, observation=obs)

    assert len(events.table) == 374
    assert_allclose(events.table["ENERGY"][0], 4.09979515940, rtol=1e-5)
    assert_allclose(events.table["RA"][0], 263.611383742, rtol=1e-5)
    assert_allclose(events.table["DEC"][0], -28.89318805, rtol=1e-5)

    assert len(events_bkg.table) == 10
    assert_allclose(events_bkg.table["ENERGY"][0], 2.84808850102, rtol=1e-5)
    assert_allclose(events_bkg.table["RA"][0], 266.6138405848, rtol=1e-5)
    assert_allclose(events_bkg.table["DEC"][0], -29.0489180785, rtol=1e-5)
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
    assert_allclose(meta["DEADC"], 0.0)
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
    assert meta["TIME-OBS"] == "00:01:04.184"
    assert meta["TIME-END"] == "00:17:44.184"
    assert_allclose(meta["TIMEDEL"], 1e-09)
    assert meta["CONV_DEP"] == 0
    assert meta["CONV_RA"] == 0
    assert meta["CONV_DEC"] == 0
    assert meta["MID00001"] == 1
    assert meta["MID00002"] == 2
    assert meta["NMCIDS"] == 2
    assert meta["ALTITUDE"] == "20.000"
    assert meta["ALT_PNT"] == "20.000"
    assert meta["AZ_PNT"] == "0.000"
    assert meta["ORIGIN"] == "Gammapy"
    assert meta["CALDB"] == "1dc"
    assert meta["IRF"] == "South_z20_50"
    assert meta["TELESCOP"] == "CTA"
    assert meta["INSTRUME"] == "1DC"
    assert meta["N_TELS"] == ""
    assert meta["TELLIST"] == ""
    assert meta["GEOLON"] == ""
    assert meta["GEOLAT"] == ""


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
    dataset.background = None

    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    assert len(events.table) == 348
    assert_allclose(events.table["ENERGY"][0], 3.5360908528324453, rtol=1e-5)
    assert_allclose(events.table["RA"][0], 266.2965928042457, rtol=1e-5)
    assert_allclose(events.table["DEC"][0], -29.056521954411902, rtol=1e-5)

    meta = events.table.meta

    assert meta["RA_PNT"] == 266.4049882865447
    assert meta["ONTIME"] == 3600.0
    assert meta["OBS_ID"] == 1001
    assert meta["RADECSYS"] == "icrs"


@requires_data()
def test_events_datastore(tmp_path, dataset):
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

    primary_hdu = fits.PrimaryHDU()
    hdu_evt = fits.BinTableHDU(events.table)
    hdu_gti = fits.BinTableHDU(dataset.gti.table, name="GTI")
    hdu_all = fits.HDUList([primary_hdu, hdu_evt, hdu_gti])
    hdu_all.writeto(str(tmp_path / "events.fits"))

    DataStore.from_events_files([str(tmp_path / "events.fits")])
