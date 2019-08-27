# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ..analysis import Analysis


def test_config():
    analysis = Analysis()
    assert analysis.settings["general"]["logging"]["level"] == "INFO"

    cfg = {"general": {"out_folder": "test"}}
    analysis = Analysis(cfg)
    assert analysis.settings["general"]["logging"]["level"] == "INFO"
    assert analysis.settings["general"]["out_folder"] == "test"


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
    analysis = Analysis(cfg)
    assert analysis.configuration.validate() is None
