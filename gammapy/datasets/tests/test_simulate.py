# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from gammapy.data import GTI, DataStore, Observation
from gammapy.datasets import MapDataset, MapDatasetEventSampler
from gammapy.datasets.tests.test_map import get_map_dataset
from gammapy.irf import load_cta_irfs
from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    FoVBackgroundModel,
    GaussianSpatialModel,
    LightCurveTemplateTemporalModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data

LOCATION = EarthLocation(lon="-70d18m58.84s", lat="-24d41m0.34s", height="2000m")


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
    temporal_model = LightCurveTemplateTemporalModel.from_table(table)

    model = SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        temporal_model=temporal_model,
        name="test-source",
    )

    bkg_model = FoVBackgroundModel(dataset_name="test")
    return [model, bkg_model]


@pytest.fixture()
def model_alternative():
    spatial_model1 = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic"
    )

    spectral_model = PowerLawSpectralModel(amplitude="1e-11 cm-2 s-1 TeV-1")

    mod1 = SkyModel(
        spatial_model=spatial_model1,
        spectral_model=spectral_model,
        name="test-source",
    )

    spatial_model2 = GaussianSpatialModel(
        lon_0="0.5 deg", lat_0="0.5 deg", sigma="0.2 deg", frame="galactic"
    )

    mod2 = SkyModel(
        spatial_model=spatial_model2,
        spectral_model=spectral_model,
        name="test-source2",
    )

    spatial_model3 = GaussianSpatialModel(
        lon_0="0.5 deg", lat_0="0.0 deg", sigma="0.2 deg", frame="galactic"
    )

    mod3 = SkyModel(
        spatial_model=spatial_model3,
        spectral_model=spectral_model,
        name="test-source3",
    )

    bkg_model = FoVBackgroundModel(dataset_name="test")

    model2 = Models([mod1, mod2, bkg_model, mod3])
    return model2


@pytest.fixture(scope="session")
def dataset():
    energy_axis = MapAxis.from_bounds(
        1, 10, nbin=3, unit="TeV", name="energy", interp="log"
    )

    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.05, width="5 deg", frame="galactic", axes=[energy_axis]
    )

    etrue_axis = energy_axis.copy(name="energy_true")
    geom_true = geom.to_image().to_cube(axes=[etrue_axis])

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

    assert len(events.table["ENERGY_TRUE"]) == 90
    assert_allclose(events.table["ENERGY_TRUE"][0], 2.383778805, rtol=1e-5)
    assert events.table["ENERGY_TRUE"].unit == "TeV"

    assert_allclose(events.table["RA_TRUE"][0], 266.56408893, rtol=1e-5)
    assert events.table["RA_TRUE"].unit == "deg"

    assert_allclose(events.table["DEC_TRUE"][0], -28.748145, rtol=1e-5)
    assert events.table["DEC_TRUE"].unit == "deg"

    assert_allclose(events.table["TIME"][0], 119.7494479, rtol=1e-5)
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
        obs_id=1001,
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=LOCATION,
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

    assert len(events.table) == 90
    assert_allclose(events.table["ENERGY_TRUE"][0], 2.38377880, rtol=1e-5)
    assert events.table["ENERGY_TRUE"].unit == "TeV"

    assert_allclose(events.table["RA"][0], 266.542912, rtol=1e-5)
    assert events.table["RA"].unit == "deg"

    assert_allclose(events.table["DEC"][0], -28.78829, rtol=1e-5)
    assert events.table["DEC"].unit == "deg"


@requires_data()
def test_mde_sample_edisp(dataset, models):
    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.sample_sources(dataset=dataset)
    events = sampler.sample_edisp(dataset.edisp, events)

    assert len(events.table) == 90
    assert_allclose(events.table["ENERGY"][0], 2.383778805, rtol=1e-5)
    assert events.table["ENERGY"].unit == "TeV"

    assert_allclose(events.table["RA_TRUE"][0], 266.564088, rtol=1e-5)
    assert events.table["RA_TRUE"].unit == "deg"

    assert_allclose(events.table["DEC_TRUE"][0], -28.7481450, rtol=1e-5)
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
        obs_id=1001,
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=LOCATION,
    )

    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    assert len(events.table) == 99
    assert_allclose(events.table["DETX"][0], -1.15531813, rtol=1e-5)
    assert events.table["DETX"].unit == "deg"

    assert_allclose(events.table["DETY"][0], -1.3343611, rtol=1e-5)
    assert events.table["DETY"].unit == "deg"


@requires_data()
def test_mde_run(dataset, models):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 1.0 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001,
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=LOCATION,
    )

    dataset.models = models
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    dataset_bkg = dataset.copy(name="new-dataset")
    dataset_bkg.models = [FoVBackgroundModel(dataset_name=dataset_bkg.name)]

    events_bkg = sampler.run(dataset=dataset_bkg, observation=obs)

    assert len(events.table) == 99
    assert_allclose(events.table["ENERGY"][0], 4.406880, rtol=1e-5)
    assert_allclose(events.table["RA"][0], 265.0677009, rtol=1e-5)
    assert_allclose(events.table["DEC"][0], -30.2640157, rtol=1e-5)

    assert len(events_bkg.table) == 21
    assert_allclose(events_bkg.table["ENERGY"][0], 1.5462581456, rtol=1e-5)
    assert_allclose(events_bkg.table["RA"][0], 265.77338329, rtol=1e-5)
    assert_allclose(events_bkg.table["DEC"][0], -30.701417442, rtol=1e-5)
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
    assert meta["MID00000"] == 0
    assert meta["MMN00000"] == "test-bkg"
    assert meta["MID00001"] == 1
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
        obs_id=1001,
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=LOCATION,
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
        obs_id=1001,
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=LOCATION,
    )

    dataset.models = models

    dataset.psf = None
    dataset.edisp = None
    dataset.background = None

    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    assert len(events.table) == 90
    assert_allclose(events.table["ENERGY"][0], 2.3837788, rtol=1e-5)
    assert_allclose(events.table["RA"][0], 266.56408893, rtol=1e-5)
    assert_allclose(events.table["DEC"][0], -28.748145, rtol=1e-5)

    meta = events.table.meta

    assert meta["RA_PNT"] == 266.4049882865447
    assert_allclose(meta["ONTIME"], 3600.0)
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
        obs_id=1001,
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=LOCATION,
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


@requires_data()
def test_MC_ID(model_alternative):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 0.1 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001,
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=LOCATION,
    )

    energy_axis = MapAxis.from_energy_bounds(
        "1.0 TeV", "10 TeV", nbin=10, per_decade=True
    )
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.5 TeV", "20 TeV", nbin=20, per_decade=True, name="energy_true"
    )
    migra_axis = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")

    geom = WcsGeom.create(
        skydir=pointing,
        width=(2, 2),
        binsz=0.06,
        frame="icrs",
        axes=[energy_axis],
    )

    empty = MapDataset.create(
        geom,
        energy_axis_true=energy_axis_true,
        migra_axis=migra_axis,
        name="test",
    )
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    dataset = maker.run(empty, obs)

    dataset.models = model_alternative
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    assert len(events.table) == 215
    assert len(np.where(events.table["MC_ID"] == 0)[0]) == 40

    meta = events.table.meta
    assert meta["MID00000"] == 0
    assert meta["MMN00000"] == "test-bkg"
    assert meta["MID00001"] == 1
    assert meta["MID00002"] == 2
    assert meta["MID00003"] == 3
    assert meta["NMCIDS"] == 4


@requires_data()
def test_MC_ID_NMCID(model_alternative):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 0.1 * u.hr
    pointing = SkyCoord(0, 0, unit="deg", frame="galactic")
    obs = Observation.create(
        obs_id=1001,
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=LOCATION,
    )

    energy_axis = MapAxis.from_energy_bounds(
        "1.0 TeV", "10 TeV", nbin=10, per_decade=True
    )
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.5 TeV", "20 TeV", nbin=20, per_decade=True, name="energy_true"
    )
    migra_axis = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")

    geom = WcsGeom.create(
        skydir=pointing,
        width=(2, 2),
        binsz=0.06,
        frame="icrs",
        axes=[energy_axis],
    )

    empty = MapDataset.create(
        geom,
        energy_axis_true=energy_axis_true,
        migra_axis=migra_axis,
        name="test",
    )
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    dataset = maker.run(empty, obs)

    model_alternative[0].spectral_model.parameters["amplitude"].value = 1e-16
    dataset.models = model_alternative
    sampler = MapDatasetEventSampler(random_state=0)
    events = sampler.run(dataset=dataset, observation=obs)

    assert len(events.table) == 47
    assert len(np.where(events.table["MC_ID"] == 0)[0]) == 47

    meta = events.table.meta
    assert meta["MID00000"] == 0
    assert meta["MMN00000"] == "test-bkg"
    assert meta["MID00001"] == 1
    assert meta["MID00002"] == 2
    assert meta["MID00003"] == 3
    assert meta["NMCIDS"] == 4
