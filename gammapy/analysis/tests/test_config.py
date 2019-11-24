import pytest
from gammapy.analysis.config import Config, General


def test_config_basics():
    config = Config()
    assert isinstance(config.general, General)


def test_config_create_from_dict():
    data = {"general": {"log": {"level": "warning"}}}
    config = Config(**data)
    assert config.general.log.level == "warning"


@pytest.mark.xfail(reason="TODO")
def test_config_create_from_yaml():
    config = Config.from_yaml("../config/config.yaml")
    assert isinstance(config.general, General)


@pytest.mark.xfail(reason="TODO")
def test_config_to_yaml():
    config = Config()
    assert "level: info" in config.to_yaml()


@pytest.mark.xfail(reason="TODO")
def test_config_update_from_dict():
    config = Config()
    config.update_from_dict()
