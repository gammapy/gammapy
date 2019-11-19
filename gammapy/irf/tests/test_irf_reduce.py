# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import Angle, SkyCoord
from gammapy.data import DataStore
from gammapy.irf import make_mean_psf, make_psf
from gammapy.maps import MapAxis
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def data_store():
    return DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")


@requires_data()
@pytest.mark.parametrize(
    "pars",
    [
        {
            "energy": None,
            "rad": None,
            "energy_shape": (32,),
            "psf_energy": 865.9643,
            "rad_shape": (144,),
            "psf_rad": 0.0015362848,
            "psf_exposure": 3.14711e12,
            "psf_value_shape": (32, 144),
            "psf_value": 4369.96391,
        },
        {
            "energy": MapAxis.from_energy_bounds(1, 10, 100, "TeV").edges,
            "rad": None,
            "energy_shape": (101,),
            "psf_energy": 1412.537545,
            "rad_shape": (144,),
            "psf_rad": 0.0015362848,
            "psf_exposure": 4.688142e12,
            "psf_value_shape": (101, 144),
            "psf_value": 3726.58798,
        },
        {
            "energy": None,
            "rad": Angle(np.arange(0, 2, 0.002), "deg"),
            "energy_shape": (32,),
            "psf_energy": 865.9643,
            "rad_shape": (1000,),
            "psf_rad": 0.000524,
            "psf_exposure": 3.14711e12,
            "psf_value_shape": (32, 1000),
            "psf_value": 25888.5047,
        },
        {
            "energy": MapAxis.from_energy_bounds(1, 10, 100, "TeV").edges,
            "rad": Angle(np.arange(0, 2, 0.002), "deg"),
            "energy_shape": (101,),
            "psf_energy": 1412.537545,
            "rad_shape": (1000,),
            "psf_rad": 0.000524,
            "psf_exposure": 4.688142e12,
            "psf_value_shape": (101, 1000),
            "psf_value": 22723.879272,
        },
    ],
)
def test_make_psf(pars, data_store):
    psf = make_psf(
        data_store.obs(23523),
        position=SkyCoord(83.63, 22.01, unit="deg"),
        energy=pars["energy"],
        rad=pars["rad"],
    )

    assert psf.energy.unit == "GeV"
    assert psf.energy.shape == pars["energy_shape"]
    assert_allclose(psf.energy.value[15], pars["psf_energy"], rtol=1e-3)

    assert psf.rad.unit == "rad"
    assert psf.rad.shape == pars["rad_shape"]
    assert_allclose(psf.rad.value[15], pars["psf_rad"], rtol=1e-3)

    assert psf.exposure.unit == "cm2 s"
    assert psf.exposure.shape == pars["energy_shape"]
    assert_allclose(psf.exposure.value[15], pars["psf_exposure"], rtol=1e-3)

    assert psf.psf_value.unit == "sr-1"
    assert psf.psf_value.shape == pars["psf_value_shape"]
    assert_allclose(psf.psf_value.value[15, 50], pars["psf_value"], rtol=1e-3)


@requires_data()
def test_make_mean_psf(data_store):
    position = SkyCoord(83.63, 22.01, unit="deg")

    observations = data_store.get_observations([23523, 23526])
    psf = make_mean_psf(observations, position=position)

    assert not np.isnan(psf.psf_value.value).any()
    assert_allclose(psf.psf_value.value[22, 22], 12206.1665)
