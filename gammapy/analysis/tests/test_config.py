# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import pytest
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
from pydantic import ValidationError
from gammapy.analysis.config import AnalysisConfig, FrameEnum, GeneralConfig
from gammapy.utils.testing import assert_allclose

CONFIG_PATH = Path(__file__).resolve().parent / ".." / "config"
DOC_FILE = CONFIG_PATH / "docs.yaml"


def test_config_default_types():
    config = AnalysisConfig()
    assert config.observations.obs_cone.frame is None
    assert config.observations.obs_cone.lon is None
    assert config.observations.obs_cone.lat is None
    assert config.observations.obs_cone.radius is None
    assert config.observations.obs_time.start is None
    assert config.observations.obs_time.stop is None
    assert config.datasets.geom.wcs.skydir.frame is None
    assert config.datasets.geom.wcs.skydir.lon is None
    assert config.datasets.geom.wcs.skydir.lat is None
    assert isinstance(config.datasets.geom.wcs.binsize, Angle)
    assert isinstance(config.datasets.geom.wcs.binsize_irf, Angle)
    assert isinstance(config.datasets.geom.axes.energy.min, Quantity)
    assert isinstance(config.datasets.geom.axes.energy.max, Quantity)
    assert isinstance(config.datasets.geom.axes.energy_true.min, Quantity)
    assert isinstance(config.datasets.geom.axes.energy_true.max, Quantity)
    assert isinstance(config.datasets.geom.selection.offset_max, Angle)
    assert config.fit.fit_range.min is None
    assert config.fit.fit_range.max is None
    assert isinstance(config.excess_map.correlation_radius, Angle)
    assert config.excess_map.energy_edges.min is None
    assert config.excess_map.energy_edges.max is None
    assert config.excess_map.energy_edges.nbins is None


def test_config_not_default_types():
    config = AnalysisConfig()
    config.observations.obs_cone = {
        "frame": "galactic",
        "lon": "83.633 deg",
        "lat": "22.014 deg",
        "radius": "1 deg",
    }
    config.fit.fit_range = {"min": "0.1 TeV", "max": "10 TeV"}
    assert isinstance(config.observations.obs_cone.frame, FrameEnum)
    assert isinstance(config.observations.obs_cone.lon, Angle)
    assert isinstance(config.observations.obs_cone.lat, Angle)
    assert isinstance(config.observations.obs_cone.radius, Angle)
    config.observations.obs_time.start = "2019-12-01"
    assert isinstance(config.observations.obs_time.start, Time)
    with pytest.raises(ValueError):
        config.flux_points.energy.min = "1 deg"
    assert isinstance(config.fit.fit_range.min, Quantity)
    assert isinstance(config.fit.fit_range.max, Quantity)


def test_config_basics():
    config = AnalysisConfig()
    assert "AnalysisConfig" in str(config)
    config = AnalysisConfig.read(DOC_FILE)
    assert config.general.outdir == "."


def test_config_create_from_dict():
    data = {"general": {"log": {"level": "warning"}}}
    config = AnalysisConfig(**data)
    assert config.general.log.level == "warning"


def test_config_create_from_yaml():
    config = AnalysisConfig.read(DOC_FILE)
    assert isinstance(config.general, GeneralConfig)
    config_str = Path(DOC_FILE).read_text()
    config = AnalysisConfig.from_yaml(config_str)
    assert isinstance(config.general, GeneralConfig)


def test_config_to_yaml(tmp_path):
    config = AnalysisConfig()
    assert "level: info" in config.to_yaml()
    config = AnalysisConfig()
    fpath = Path(tmp_path) / "temp.yaml"
    config.write(fpath)
    text = Path(fpath).read_text()
    assert "stack" in text
    with pytest.raises(IOError):
        config.write(fpath)


def test_get_doc_sections():
    config = AnalysisConfig()
    doc = config._get_doc_sections()
    assert "general" in doc.keys()


def test_safe_mask_config_validation():
    config = AnalysisConfig()
    # Check empty list is accepted
    config.datasets.safe_mask.methods = []

    with pytest.raises(ValidationError):
        config.datasets.safe_mask.methods = ["bad"]


def test_time_range_iso():
    cfg = """
    observations:
        datastore: $GAMMAPY_DATA/hess-dl3-dr1
        obs_ids: [23523, 23526]
        obs_time: {
            start: [2004-12-04 22:04:48.000, 2004-12-04 22:26:24.000, 2004-12-04 22:53:45.600],
            stop: [2004-12-04 22:26:24.000, 2004-12-04 22:53:45.600, 2004-12-04 23:31:12.000]
            }
    """
    config = AnalysisConfig.from_yaml(cfg)

    assert_allclose(
        config.observations.obs_time.start.mjd, [53343.92, 53343.935, 53343.954]
    )


def test_time_range_jyear():
    cfg = """
    observations:
        datastore: $GAMMAPY_DATA/hess-dl3-dr1
        obs_ids: [23523, 23526]
        obs_time: {
            start: [J2004.92654346, J2004.92658453, J2004.92663655],
            stop: [J2004.92658453, J2004.92663655, J2004.92670773]
            }
    """
    config = AnalysisConfig.from_yaml(cfg)

    assert_allclose(
        config.observations.obs_time.start.mjd, [53343.92, 53343.935, 53343.954]
    )
