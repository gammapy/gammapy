from ..analysis import Analysis


def test_config():
    cfg = {"general": {"out_folder": "test"}}
    analysis = Analysis(config=cfg)
    assert analysis.configuration.settings["general"]["logging"]["level"] == "INFO"
    assert analysis.configuration.settings["general"]["out_folder"] == "test"
