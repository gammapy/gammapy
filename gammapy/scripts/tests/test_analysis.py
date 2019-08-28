# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.scripts import Analysis


def test_config():
    analysis = Analysis()
    assert analysis.settings["general"]["logging"]["level"] == "INFO"

    config = {"general": {"out_folder": "test"}}
    analysis = Analysis(config)
    assert analysis.settings["general"]["logging"]["level"] == "INFO"
    assert analysis.settings["general"]["out_folder"] == "test"


def test_validate_config():
    analysis = Analysis()
    assert analysis.config.validate() is None


def test_validate_astropy_quantities():
    config = {"observations": {"filter": [{"lon": "1 deg"}]}}
    analysis = Analysis(config)
    assert analysis.config.validate() is None
