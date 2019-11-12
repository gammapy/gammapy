# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import pytest
from numpy.testing import assert_allclose
import yaml
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import SkyModels
from gammapy.utils.testing import requires_data, requires_dependency

CONFIG_PATH = Path(__file__).resolve().parent / ".." / "config"
MODEL_FILE = CONFIG_PATH / "model.yaml"
DOC_FILE = CONFIG_PATH / "docs.yaml"


def test_config():
    config = AnalysisConfig()
    assert config.settings["general"]["logging"]["level"] == "INFO"
    cfg = {"general": {"outdir": "test"}}
    config.update_settings(cfg)
    assert config.settings["general"]["logging"]["level"] == "INFO"
    assert config.settings["general"]["outdir"] == "test"

    with pytest.raises(ValueError):
        Analysis()

    assert "AnalysisConfig" in str(config)


def test_config_to_yaml(tmp_path):
    config = AnalysisConfig()
    config.settings["general"]["outdir"] = tmp_path
    config.to_yaml(overwrite=True)
    text = (tmp_path / config.filename).read_text()
    assert "stack-datasets" in text


def config_observations():
    cfg = """
      - observations:
            datastore: $GAMMAPY_DATA/cta-1dc/index/gps/
            filters:
            - filter_type: all
        result: 4
      - observations:
            datastore: $GAMMAPY_DATA/cta-1dc/index/gps/
            filters:
            - filter_type: ids
              obs_ids:
              - 110380
        result: 1
      - observations:
            datastore: $GAMMAPY_DATA/cta-1dc/index/gps/
            filters:
            - filter_type: all
            - filter_type: ids
              exclude: true
              obs_ids:
              - 110380
        result: 3
      - observations:
            datastore: $GAMMAPY_DATA/cta-1dc/index/gps/
            filters:
            - filter_type: sky_circle
              frame: galactic
              lat: 0 deg
              lon: 0 deg
              border: 0.5 deg
              radius: 1 deg
        result: 1
      - observations:
            datastore: $GAMMAPY_DATA/cta-1dc/index/gps/
            filters:
            - filter_type: angle_box
              value_range:
              - 265 deg
              - 268 deg
              variable: RA_PNT
        result: 2
      - observations:
            datastore: $GAMMAPY_DATA/cta-1dc/index/gps/
            filters:
            - filter_type: par_box
              value_range:
              - 106000
              - 107000
              variable: EVENT_COUNT
        result: 2
      - observations:
            datastore: $GAMMAPY_DATA/hess-dl3-dr1
            filters:
            - filter_type: par_value
              value_param: Off data
              variable: TARGET_NAME
        result: 45
    """
    config_obs = yaml.safe_load(cfg)
    return config_obs


@requires_data()
@pytest.mark.parametrize("config_obs", config_observations())
def test_get_observations(config_obs):
    config = AnalysisConfig()
    analysis = Analysis(config)
    analysis.config.update_settings(config_obs)
    analysis.get_observations()
    assert len(analysis.observations) == config_obs["result"]


@pytest.fixture(scope="session")
def config_analysis_data():
    """Get test config, extend to several scenarios"""
    cfg = """
    observations:
        datastore: $GAMMAPY_DATA/hess-dl3-dr1
        filters:
            - filter_type: ids
              obs_ids: [23523, 23526]
    datasets:
        background:
            background_estimator: reflected
        containment_correction: false
        dataset-type: SpectrumDatasetOnOff
        geom:
            region:
                center:
                - 83.633 deg
                - 22.014 deg
                frame: icrs
                radius: 0.11 deg
    flux-points:
        fp_binning:
            lo_bnd: 1
            hi_bnd: 50
            nbin: 4
            unit: TeV
            interp: log
    """
    return cfg


@requires_dependency("iminuit")
@requires_data()
def test_analysis_1d(config_analysis_data):
    config = AnalysisConfig.from_template("1d")
    analysis = Analysis(config)
    analysis.config.update_settings(config_analysis_data)
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
    assert_allclose(dnde[-1].value, 4.780021e-21, rtol=1e-2)


@requires_dependency("iminuit")
@requires_data()
def test_analysis_1d_stacked():
    config = AnalysisConfig.from_template("1d")
    analysis = Analysis(config)
    analysis.settings["datasets"]["stack-datasets"] = True
    analysis.get_observations()
    analysis.get_datasets()
    analysis.set_model(filename=MODEL_FILE)
    analysis.run_fit()

    assert len(analysis.datasets) == 1
    assert_allclose(analysis.datasets["stacked"].counts.data.sum(), 404)
    pars = analysis.fit_result.parameters

    assert_allclose(pars["index"].value, 2.689559, rtol=1e-3)
    assert_allclose(pars["amplitude"].value, 2.81629e-11, rtol=1e-3)


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
    assert res[3].unit == "cm-2 s-1 TeV-1"
    assert len(analysis.flux_points.data.table) == 2
    dnde = analysis.flux_points.data.table["dnde"].quantity

    assert_allclose(dnde[0].value, 1.182768e-11, rtol=1e-1)
    assert_allclose(dnde[-1].value, 4.051367e-13, rtol=1e-1)
    assert_allclose(res["index"].value, 2.76607, rtol=1e-1)
    assert_allclose(res["tilt"].value, -0.143204, rtol=1e-1)


@requires_data()
def test_analysis_3d_joint_datasets():
    config = AnalysisConfig.from_template("3d")
    config.settings["datasets"]["stack-datasets"] = False
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    assert len(analysis.datasets) == 4


def test_validate_astropy_quantities():
    config = AnalysisConfig()
    cfg = {"observations": {"filters": [{"filter_type": "all", "lon": "1 deg"}]}}
    config.update_settings(cfg)
    assert config.validate() is None


def test_validate_config():
    config = AnalysisConfig()
    assert config.validate() is None


def test_docs_file():
    config = AnalysisConfig.from_yaml(DOC_FILE)
    assert config.validate() is None


def test_help():
    config = AnalysisConfig()
    assert config.help() is None


@requires_data()
def test_analysis_3d_no_geom_irf():
    config = AnalysisConfig.from_template("3d")
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()

    assert len(analysis.datasets) == 1


@requires_dependency("iminuit")
@requires_data()
def test_validation_checks():
    config = AnalysisConfig()
    analysis = Analysis(config)
    analysis.settings["observations"]["datastore"] = "other"
    with pytest.raises(FileNotFoundError):
        analysis.get_observations()

    config = AnalysisConfig.from_template("1d")
    analysis = Analysis(config)
    assert analysis.get_flux_points() is False
    assert analysis.run_fit() is False
    assert analysis.set_model() is False
    assert analysis.get_datasets() is False

    analysis.get_observations()
    analysis.settings["datasets"]["dataset-type"] = "not assigned"
    assert analysis.get_datasets() is False

    analysis.settings["datasets"]["dataset-type"] = "SpectrumDatasetOnOff"
    analysis.get_observations()
    analysis.get_datasets()
    model_str = Path(MODEL_FILE).read_text()
    analysis.set_model(model=model_str)
    assert isinstance(analysis.model, SkyModels) is True
    assert analysis.set_model() is False

    analysis.run_fit()
    del analysis.settings["flux-points"]
    assert analysis.get_flux_points() is False
