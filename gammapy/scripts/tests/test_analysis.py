# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import yaml
from gammapy.scripts import Analysis
from gammapy.utils.testing import requires_data, requires_dependency


def test_config():
    analysis = Analysis()
    assert analysis.settings["general"]["logging"]["level"] == "INFO"

    config = {"general": {"outdir": "test"}}
    analysis = Analysis(config)
    assert analysis.settings["general"]["logging"]["level"] == "INFO"
    assert analysis.settings["general"]["outdir"] == "test"


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
          - exclude: true
            filter_type: ids
            obs_ids:
            - 110380
        result: 3
      - observations:
          datastore: $GAMMAPY_DATA/cta-1dc/index/gps/
          filters:
          - border: 0.5 deg
            filter_type: sky_circle
            frame: galactic
            lat: 0 deg
            lon: 0 deg
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
@pytest.mark.parametrize("config", config_observations())
def test_get_observations(config):
    analysis = Analysis(config)
    analysis.get_observations()
    assert len(analysis.observations) == config["result"]


@pytest.fixture(scope="session")
def config_analysis_data():
    """Get test config, extend to several scenarios"""
    cfg = """
    general:
      logging:
        level: INFO
      outdir: .
    model:
        components:
        - name: source
          spectral:
                parameters:
                - factor: 2.0
                  frozen: false
                  name: index
                  unit: ''
                  value: 2.0
                - factor: 1.0e-12
                  frozen: false
                  name: amplitude
                  unit: cm-2 s-1 TeV-1
                  value: 1.0e-11
                - factor: 1.0
                  frozen: true
                  name: reference
                  unit: TeV
                  value: 1.0
                type: PowerLaw
    observations:
      datastore: $GAMMAPY_DATA/hess-dl3-dr1
      filters:
      - filter_type: ids
        obs_ids: [23523, 23526]
    reduction:
        background:
            background_estimator: reflected
            on_region:
                center:
                - 83.633 deg
                - 22.014 deg
                frame: icrs
                radius: 0.11 deg
        containment_correction: false
        data_reducer: 1d 
    flux:
        fp_binning:
            lo_bnd: 1
            hi_bnd: 50
            nbin: 4
            unit: TeV
            interp: log      
    """
    return yaml.safe_load(cfg)


@requires_dependency("iminuit")
@requires_data()
def test_analysis(config_analysis_data):
    analysis = Analysis(config_analysis_data)
    analysis.get_observations()
    analysis.reduce()
    analysis.fit()
    analysis.get_flux_points()
    assert len(analysis.extraction.spectrum_observations) == 2
    assert_allclose(analysis.fit_result.total_stat, 79.8849, rtol=1e-2)
    assert len(analysis.flux_points_dataset.data.table) == 4
    dnde = analysis.flux_points_dataset.data.table["dnde"].quantity
    assert dnde.unit == "cm-2 s-1 TeV-1"
    assert_allclose(dnde[0].value, 6.601518e-12, rtol=1e-2)
    assert_allclose(dnde[-1].value, 1.295918e-15, rtol=1e-2)


def test_validate_astropy_quantities():
    config = {"observations": {"filters": [{"lon": "1 deg"}]}}
    analysis = Analysis(config)
    assert analysis.config.validate() is None


def test_validate_config():
    analysis = Analysis()
    assert analysis.config.validate() is None
