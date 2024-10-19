# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data.utils import get_irfs_features
from gammapy.utils.testing import (
    assert_allclose,
    requires_data,
)
from gammapy.data import DataStore


@requires_data()
def test_irfs_features():
    selection = dict(
        type="sky_circle",
        frame="icrs",
        lon="329.716 deg",
        lat="-30.225 deg",
        radius="2 deg",
    )

    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_table = data_store.obs_table.select_observations(selection)
    obs = data_store.get_observations(obs_table["OBS_ID"][:1])

    position = SkyCoord(329.716 * u.deg, -30.225 * u.deg, frame="icrs")
    names = ["edisp-bias", "edisp-res", "psf-radius"]
    features_irfs = get_irfs_features(
        obs,
        energy_true="1 TeV",
        position=position,
        names=names,
    )
    assert_allclose(features_irfs[0]["edisp-bias"], 0.11587, rtol=1.0e-4)
    assert_allclose(features_irfs[0]["edisp-res"], 0.36834, rtol=1.0e-4)
    assert_allclose(features_irfs[0]["psf-radius"], 0.14149, rtol=1.0e-4)
