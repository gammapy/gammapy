from ..analysis import Analysis


def test_config():
    assert "global" in Analysis().config.keys()
    assert "global" not in Analysis(configfile="").config.keys()
    assert Analysis(test=True).config["test"]


def test_validate_config():
    cfg = {"global": {"logging": {"level": "INFO"}}}
    assert Analysis().validate_config() is None
    assert Analysis(configfile="", **cfg).validate_config() is None
