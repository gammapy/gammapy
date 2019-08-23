from ..analysis import Analysis
import copy


def test_config():
    assert Analysis()._validate_schema() is None


def test_list_config():
    assert "global" in Analysis().list_config().keys()


def test_set_config():
    analysis = Analysis()
    analysis.set_config("global.logging.level", "info")
    cfgset = copy.deepcopy(analysis.list_config())
    analysis.set_config("global.test", "test")
    cfgadd = copy.deepcopy(analysis.list_config())
    analysis.set_config("global.test", None)
    cfgdel = copy.deepcopy(analysis.list_config())

    assert cfgset["global"]["logging"]["level"] == "info"
    assert cfgadd["global"]["test"] == "test"
    assert "test" not in cfgdel["global"].keys()
