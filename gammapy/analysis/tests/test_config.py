# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import pytest
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
from gammapy.analysis.config import AnalysisConfig, General, FrameEnum

CONFIG_PATH = Path(__file__).resolve().parent / ".." / "config"
CONFIG_FILE = CONFIG_PATH / "config.yaml"
DOC_FILE = CONFIG_PATH / "docs.yaml"


def test_config_default_types():
    config = AnalysisConfig()
    assert config.data.obs_cone.frame is None
    assert config.data.obs_cone.lon is None
    assert config.data.obs_cone.lat is None
    assert config.data.obs_cone.radius is None
    assert config.data.obs_time.start is None
    assert config.data.obs_time.stop is None
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
    assert isinstance(config.datasets.geom.wcs.binsize, Angle)


def test_config_not_default_types():
    config = AnalysisConfig()
    config.data.obs_cone = {
        "frame": "galactic",
        "lon": "83.633 deg",
        "lat": "22.014 deg",
        "radius": "1 deg"
    }
    assert isinstance(config.data.obs_cone.frame, FrameEnum)
    assert isinstance(config.data.obs_cone.lon, Angle)
    assert isinstance(config.data.obs_cone.lat, Angle)
    assert isinstance(config.data.obs_cone.radius, Angle)
    config.data.obs_time.start = "2019-12-01"
    assert isinstance(config.data.obs_time.start, Time)
    with pytest.raises(ValueError):
        config.flux_points.energy.min = "1 deg"


def test_config_basics():
    config = AnalysisConfig()
    assert "AnalysisConfig" in str(config)
    assert config.help() is None
    config = AnalysisConfig.from_yaml(filename=DOC_FILE)
    assert config.general.outdir == "."


def test_config_create_from_dict():
    data = {"general": {"log": {"level": "warning"}}}
    config = AnalysisConfig(**data)
    assert config.general.log.level == "warning"


def test_config_create_from_yaml():
    config = AnalysisConfig.from_yaml(filename=CONFIG_FILE)
    assert isinstance(config.general, General)


def test_config_to_yaml(tmp_path):
    config = AnalysisConfig()
    assert "level: info" in config.to_yaml()

    filename = "temp.yaml"
    config = AnalysisConfig()
    config.general.outdir = str(tmp_path)
    config.to_yaml(filename=filename)
    text = (tmp_path / filename).read_text()
    assert "stack" in text
    with pytest.raises(IOError):
        config.to_yaml(filename=filename)


def test_config_update():
    config1 = AnalysisConfig()
    data = {"fit": {"fit_range": {"min": "1 TeV", "max": "100 TeV"}}}
    config2 = AnalysisConfig(**data)
    config = config1._update(config2)
    assert config.fit.fit_range.min == Quantity("1 TeV")
    assert config.general.log.level == "info"
