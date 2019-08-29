# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.scripts import Analysis
from gammapy.utils.testing import requires_data


def test_config():
    analysis = Analysis()
    assert analysis.settings["general"]["logging"]["level"] == "INFO"

    config = {"general": {"outdir": "test"}}
    analysis = Analysis(config)
    assert analysis.settings["general"]["logging"]["level"] == "INFO"
    assert analysis.settings["general"]["outdir"] == "test"


def test_validate_config():
    analysis = Analysis()
    assert analysis.config.validate() is None


def test_validate_astropy_quantities():
    config = {"observations": {"filters": [{"lon": "1 deg"}]}}
    analysis = Analysis(config)
    assert analysis.config.validate() is None


@requires_data()
@pytest.mark.parametrize(
    "config",
    [
        {
            "observations": {
                "datastore": "$GAMMAPY_DATA/cta-1dc/index/gps/",
                "filters": [{"filter_type": "all"}],
            },
            "result": 4,
        },
        {
            "observations": {
                "datastore": "$GAMMAPY_DATA/cta-1dc/index/gps/",
                "filters": [{"filter_type": "ids", "obs_ids": [110380]}],
            },
            "result": 1,
        },
        {
            "observations": {
                "datastore": "$GAMMAPY_DATA/cta-1dc/index/gps/",
                "filters": [
                    {"filter_type": "all"},
                    {"filter_type": "ids", "obs_ids": [110380], "exclude": True},
                ],
            },
            "result": 3,
        },
        {
            "observations": {
                "datastore": "$GAMMAPY_DATA/cta-1dc/index/gps/",
                "filters": [
                    {
                        "filter_type": "sky_circle",
                        "frame": "galactic",
                        "lon": "0 deg",
                        "lat": "0 deg",
                        "radius": "1 deg",
                        "border": "0.5 deg",
                    }
                ],
            },
            "result": 1,
        },
        {
            "observations": {
                "datastore": "$GAMMAPY_DATA/cta-1dc/index/gps/",
                "filters": [
                    {
                        "filter_type": "angle_box",
                        "variable": "RA_PNT",
                        "value_range": ["265 deg", "268 deg"],
                    }
                ],
            },
            "result": 2,
        },
        {
            "observations": {
                "datastore": "$GAMMAPY_DATA/cta-1dc/index/gps/",
                "filters": [
                    {
                        "filter_type": "par_box",
                        "variable": "EVENT_COUNT",
                        "value_range": [106000, 107000],
                    }
                ],
            },
            "result": 2,
        },
        {
            "observations": {
                "datastore": "$GAMMAPY_DATA/hess-dl3-dr1/hess-dl3-dr3-with-background.fits.gz",
                "filters": [
                    {
                        "filter_type": "par_value",
                        "variable": "TARGET_NAME",
                        "par_value": "Crab",
                    }
                ],
            },
            "result": 4,
        },
    ],
)
def test_get_observations(config):
    analysis = Analysis(config)
    analysis.get_observations()
    assert len(analysis.observations) == config["result"]
