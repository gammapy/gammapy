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
    cfg = {"observations": {"filter": [{"lon": "1 deg"}]}}
    analysis = Analysis(cfg)
    assert analysis.configuration.validate() is None


@pytest.mark.parametrize(
    "config",
    [
        {"result": 4},
        {
            "observations": {"filter": [{"filter_type": "ids", "obs_ids": [110380]}]},
            "result": 1,
        },
        {
            "observations": {
                "filter": [
                    {"filter_type": "all"},
                    {"filter_type": "ids", "obs_ids": [110380], "exclude": True},
                ]
            },
            "result": 3,
        },
        {
            "observations": {
                "filter": [
                    {
                        "filter_type": "sky_circle",
                        "frame": "galactic",
                        "lon": "0 deg",
                        "lat": "0 deg",
                        "radius": "1 deg",
                        "border": "0.5 deg",
                    }
                ]
            },
            "result": 1,
        },
        {
            "observations": {
                "filter": [
                    {
                        "filter_type": "angle_box",
                        "variable": "RA_PNT",
                        "value_range": ["265 deg", "268 deg"],
                    }
                ]
            },
            "result": 2,
        },
        {
            "observations": {
                "filter": [
                    {
                        "filter_type": "par_box",
                        "variable": "EVENT_COUNT",
                        "value_range": [106000, 107000],
                    }
                ]
            },
            "result": 2,
        },
    ],
)
def test_get_observations(config):
    analysis = Analysis(config)
    analysis.get_observations()
    assert len(analysis.observations) == config["result"]
