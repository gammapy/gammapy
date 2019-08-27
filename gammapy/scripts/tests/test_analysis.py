from ..analysis import Analysis


def test_config():
    analysis = Analysis()
    assert analysis.configuration.settings["general"]["logging"]["level"] == "INFO"

    cfg = {"general": {"out_folder": "test"}}
    analysis = Analysis(config=cfg)
    assert analysis.configuration.settings["general"]["logging"]["level"] == "INFO"
    assert analysis.configuration.settings["general"]["out_folder"] == "test"


def test_validate_config():
    analysis = Analysis()
    assert analysis.configuration.validate() is None


def test_validate_astropy_quantities():
    cfg = {"observations":
            {
                "filter": [
                            {"lon": "1 deg"}
                          ]
                }
            }
    analysis = Analysis(config=cfg)
    assert analysis.configuration.validate() is None
