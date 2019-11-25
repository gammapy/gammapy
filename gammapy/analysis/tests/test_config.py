import pytest
from astropy.units import Quantity
from gammapy.analysis.config import AnalysisConfig, General
from pathlib import Path


def test_config_basics():
    config = AnalysisConfig()
    assert isinstance(config.general, General)


def test_config_create_from_dict():
    data = {"general": {"log": {"level": "warning"}}}
    config = AnalysisConfig(**data)
    assert config.general.log.level == "warning"


def test_config_create_from_yaml():
    cfg = Path(__file__).resolve().parent / ".." / "config" / "config.yaml"
    config = AnalysisConfig.from_yaml(cfg)
    assert isinstance(config.general, General)


def test_config_to_yaml():
    config = AnalysisConfig()
    assert "level: info" in config.to_yaml()


def test_config_update_from_dict():
    config1 = AnalysisConfig()
    data = {"fit": {"fit_range": {"min": "1 TeV",  "max": "100 TeV"}}}
    config2 = AnalysisConfig(**data)
    config = config1.update_from_dict(config2)
    assert config.fit.fit_range.min == Quantity("1 TeV")
    assert config.general.log.level == "info"
