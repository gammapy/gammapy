import pytest
from astropy.units import Quantity
from gammapy.analysis.config import Config, General
from pathlib import Path


def test_config_basics():
    config = Config()
    assert isinstance(config.general, General)


def test_config_create_from_dict():
    data = {"general": {"log": {"level": "warning"}}}
    config = Config(**data)
    assert config.general.log.level == "warning"


def test_config_create_from_yaml():
    cfg = Path(__file__).resolve().parent / ".." / "config" / "config.yaml"
    config = Config.from_yaml(cfg)
    assert isinstance(config.general, General)


def test_config_to_yaml():
    config = Config()
    assert "level: info" in config.to_yaml()


def test_config_update_from_dict():
    config1 = Config()
    data = {"fit": {"fit_range": {"min": "1 TeV",  "max": "100 TeV"}}}
    config2 = Config(**data)
    config = config1.update_from_dict(config2)
    assert config.fit.fit_range.min == Quantity("1 TeV")
    assert config.general.log.level == "info"
