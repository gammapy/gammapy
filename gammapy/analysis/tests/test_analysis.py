# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.maps import Map
from gammapy.modeling.models import SkyModels
from gammapy.utils.testing import requires_data, requires_dependency

CONFIG_PATH = Path(__file__).resolve().parent / ".." / "config"
MODEL_FILE = CONFIG_PATH / "model.yaml"


def test_init():
    cfg = {"general": {"outdir": "test"}}
    analysis = Analysis(cfg)
    assert analysis.config.general.outdir == "test"
    with pytest.raises(TypeError):
        Analysis("spam")


def test_update_config():
    analysis = Analysis(AnalysisConfig())
    data = {"general": {"outdir": "test"}}
    cfg2 = AnalysisConfig(**data)
    analysis.update_config(cfg2)
    assert analysis.config.general.outdir == "test"
    analysis = Analysis(AnalysisConfig())
    data = """
    general:
        outdir: test
    """
    analysis.update_config(data)
    assert analysis.config.general.outdir == "test"
    analysis = Analysis(AnalysisConfig())
    data = 0
    with pytest.raises(TypeError):
        analysis.update_config(data)


@requires_data()
def test_get_observations():
    config = AnalysisConfig()
    analysis = Analysis(config)
    analysis.config.observations.datastore = "other"
    with pytest.raises(FileNotFoundError):
        analysis.get_observations()
    analysis.config.observations.datastore = "$GAMMAPY_DATA/cta-1dc/index/gps/"
    analysis.get_observations()
    assert len(analysis.observations) == 4
    analysis.config.observations.obs_ids = ["110380"]
    analysis.get_observations()
    assert len(analysis.observations) == 1
    config = AnalysisConfig.from_template("1d")
    analysis = Analysis(config)
    analysis.get_observations()
    assert len(analysis.observations) == 4
    # TODO
    # obs_file
    # obs_time


@requires_data()
def test_set_model():
    config = AnalysisConfig.from_template("1d")
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    model_str = Path(MODEL_FILE).read_text()
    analysis.set_model(model=model_str)
    assert isinstance(analysis.model, SkyModels) is True
    assert analysis.set_model() is False


@requires_dependency("iminuit")
@requires_data()
def test_analysis_1d():
    cfg = """
    observations:
        datastore: $GAMMAPY_DATA/hess-dl3-dr1
        obs_ids: [23523, 23526]
    datasets:
        type: 1d
        background:
            method: reflected
        on_region: {frame: icrs, lon: 83.633 deg, lat: 22.014 deg, radius: 0.11 deg}
        containment_correction: false
    flux_points:
        energy: {min: 1 TeV, max: 50 TeV, nbins: 4}
    """
    config = AnalysisConfig.from_template("1d")
    analysis = Analysis(config)
    analysis.update_config(cfg)
    analysis.get_observations()
    analysis.get_datasets()
    analysis.set_model(filename=MODEL_FILE)
    analysis.run_fit()
    analysis.get_flux_points()

    assert len(analysis.datasets) == 2
    assert len(analysis.flux_points.data.table) == 4
    dnde = analysis.flux_points.data.table["dnde"].quantity
    assert dnde.unit == "cm-2 s-1 TeV-1"

    assert_allclose(dnde[0].value, 8.03604e-12, rtol=1e-2)
    assert_allclose(dnde[-1].value, 5.382879e-21, rtol=1e-2)


@requires_data()
def test_exclusion_region(tmp_path):
    config = AnalysisConfig.from_template("1d")
    analysis = Analysis(config)

    exclusion_region = CircleSkyRegion(center=SkyCoord("85d 23d"), radius=1 * u.deg)
    exclusion_mask = Map.create(npix=(150, 150), binsz=0.05, skydir=SkyCoord("83d 22d"))
    mask = exclusion_mask.geom.region_mask([exclusion_region], inside=False)
    exclusion_mask.data = mask.astype(int)
    filename = tmp_path / "exclusion.fits"
    exclusion_mask.write(filename)
    config.datasets.background.exclusion = filename

    analysis.get_observations()
    analysis.get_datasets()
    assert len(analysis.datasets) == 4


@requires_dependency("iminuit")
@requires_data()
def test_analysis_1d_stacked():
    config = AnalysisConfig.from_template("1d")
    analysis = Analysis(config)
    analysis.config.datasets.stack = True
    analysis.get_observations()
    analysis.get_datasets()
    analysis.set_model(filename=MODEL_FILE)
    analysis.run_fit()

    assert len(analysis.datasets) == 1
    assert_allclose(analysis.datasets["stacked"].counts.data.sum(), 404)
    pars = analysis.fit_result.parameters

    assert_allclose(pars["index"].value, 2.676283, rtol=1e-3)
    assert_allclose(pars["amplitude"].value, 4.554215e-11, rtol=1e-3)


@requires_dependency("iminuit")
@requires_data()
def test_analysis_3d():
    config = AnalysisConfig.from_template("3d")
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    analysis.set_model(filename=MODEL_FILE)
    analysis.datasets["stacked"].background_model.tilt.frozen = False
    analysis.run_fit()
    analysis.get_flux_points()

    assert len(analysis.datasets) == 1
    assert len(analysis.fit_result.parameters) == 8
    res = analysis.fit_result.parameters
    assert res["amplitude"].unit == "cm-2 s-1 TeV-1"
    assert len(analysis.flux_points.data.table) == 2
    dnde = analysis.flux_points.data.table["dnde"].quantity

    assert_allclose(dnde[0].value, 1.182768e-11, rtol=1e-1)
    assert_allclose(dnde[-1].value, 4.051367e-13, rtol=1e-1)
    assert_allclose(res["index"].value, 2.76607, rtol=1e-1)
    assert_allclose(res["tilt"].value, -0.143204, rtol=1e-1)


@requires_data()
def test_analysis_3d_joint_datasets():
    config = AnalysisConfig.from_template("3d")
    config.datasets.stack = False
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    assert len(analysis.datasets) == 4


@requires_dependency("iminuit")
@requires_data()
def test_usage_errors():
    config = AnalysisConfig.from_template("1d")
    analysis = Analysis(config)
    with pytest.raises(RuntimeError):
        analysis.get_datasets()
    with pytest.raises(RuntimeError):
        analysis.set_model()
    with pytest.raises(RuntimeError):
        analysis.run_fit()
    with pytest.raises(RuntimeError):
        analysis.get_flux_points()
