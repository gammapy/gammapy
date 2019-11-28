# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import pytest
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
from gammapy.analysis.config import AnalysisConfig, GeneralConfig, FrameEnum

CONFIG_PATH = Path(__file__).resolve().parent / ".." / "config"
CONFIG_FILE = CONFIG_PATH / "config.yaml"
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
    assert isinstance(config.datasets.geom.wcs.margin_irf, Angle)
    assert isinstance(config.datasets.geom.selection.offset_max, Angle)
    assert isinstance(config.datasets.geom.axes.energy.min, Quantity)
    assert isinstance(config.datasets.geom.axes.energy.max, Quantity)
    assert isinstance(config.datasets.geom.axes.energy_true.min, Quantity)
    assert isinstance(config.datasets.geom.axes.energy_true.max, Quantity)
    assert isinstance(config.fit.fit_range.min, Quantity)
    assert isinstance(config.fit.fit_range.max, Quantity)


def test_config_not_default_types():
    config = AnalysisConfig()
    config.observations.obs_cone = {
        "frame": "galactic",
        "lon": "83.633 deg",
        "lat": "22.014 deg",
        "radius": "1 deg"
    }
    assert isinstance(config.observations.obs_cone.frame, FrameEnum)
    assert isinstance(config.observations.obs_cone.lon, Angle)
    assert isinstance(config.observations.obs_cone.lat, Angle)
    assert isinstance(config.observations.obs_cone.radius, Angle)
    config.observations.obs_time.start = "2019-12-01"
    assert isinstance(config.observations.obs_time.start, Time)
    with pytest.raises(ValueError):
        config.flux_points.energy.min = "1 deg"


def test_config_basics():
    config = AnalysisConfig()
    assert "AnalysisConfig" in str(config)
    assert config.help() is None
    config = AnalysisConfig.read(DOC_FILE)
    assert config.general.outdir == "."


def test_config_create_from_dict():
    data = {"general": {"log": {"level": "warning"}}}
    config = AnalysisConfig(**data)
    assert config.general.log.level == "warning"


def test_config_create_from_yaml():
    config = AnalysisConfig.read(CONFIG_FILE)
    assert isinstance(config.general, GeneralConfig)
    config_str = Path(CONFIG_FILE).read_text()
    config = AnalysisConfig.from_yaml(config_str)
    assert isinstance(config.general, GeneralConfig)


def test_config_to_yaml(tmp_path):
    config = AnalysisConfig()
    assert "level: info" in config.to_yaml()

    filename = "temp.yaml"
    config = AnalysisConfig()
    config.general.outdir = str(tmp_path)
    config.write(filename)
    text = (tmp_path / filename).read_text()
    assert "stack" in text
    with pytest.raises(IOError):
        config.write(filename)
