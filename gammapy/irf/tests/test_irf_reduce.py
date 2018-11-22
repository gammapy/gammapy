# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.coordinates import Angle, SkyCoord
from ..irf_reduce import make_psf, make_mean_psf, make_mean_edisp
from ...data import DataStore, Observations
from ...utils.testing import requires_data, assert_quantity_allclose
from ...utils.energy import Energy, EnergyBounds


@pytest.fixture(scope="session")
def data_store():
    return DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/")


@requires_data("gammapy-extra")
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
            "energy": EnergyBounds.equal_log_spacing(1, 10, 100, "TeV"),
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
            # TODO: should this be psf_value_shape == (32, 1000) ?
            "psf_value_shape": (32, 144),
            "psf_value": 4369.96391,
        },
        {
            "energy": EnergyBounds.equal_log_spacing(1, 10, 100, "TeV"),
            "rad": Angle(np.arange(0, 2, 0.002), "deg"),
            "energy_shape": (101,),
            "psf_energy": 1412.537545,
            "rad_shape": (1000,),
            "psf_rad": 0.000524,
            "psf_exposure": 4.688142e12,
            "psf_value_shape": (101, 144),
            "psf_value": 3726.58798,
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


@requires_data("gammapy-extra")
def test_make_mean_psf(data_store):
    position = SkyCoord(83.63, 22.01, unit="deg")
    obs1 = data_store.obs(23523)
    obs2 = data_store.obs(23526)
    energy = EnergyBounds.equal_log_spacing(1, 10, 100, "TeV")
    energy_band = Energy([energy[0].value, energy[-1].value], energy.unit)

    psf1 = make_psf(obs1, position=position, energy=energy, rad=None)
    psf2 = make_psf(obs2, position=position, energy=energy, rad=None)
    psf1_int = psf1.table_psf_in_energy_band(energy_band, spectral_index=2.3)
    psf2_int = psf2.table_psf_in_energy_band(energy_band, spectral_index=2.3)
    observations = Observations([obs1, obs2])
    psf_tot = make_mean_psf(observations, position=position, energy=energy)
    psf_tot_int = psf_tot.table_psf_in_energy_band(energy_band, spectral_index=2.3)

    # Check that the mean PSF is consistent with the individual PSFs
    # (in this case the R68 of the mean PSF is in between the R68 of the individual PSFs)
    assert_allclose(psf1_int.containment_radius(0.68).deg, 0.12307, rtol=1e-3)
    assert_allclose(psf2_int.containment_radius(0.68).deg, 0.11231, rtol=1e-3)
    assert_allclose(psf_tot_int.containment_radius(0.68).deg, 0.117803, rtol=1e-3)


@requires_data("gammapy-extra")
def test_make_mean_edisp(data_store):
    position = SkyCoord(83.63, 22.01, unit="deg")

    obs1 = data_store.obs(23523)
    obs2 = data_store.obs(23592)
    observations = Observations([obs1, obs2])

    e_true = EnergyBounds.equal_log_spacing(0.01, 150, 80, "TeV")
    e_reco = EnergyBounds.equal_log_spacing(0.5, 100, 15, "TeV")
    rmf = make_mean_edisp(observations, position=position, e_true=e_true, e_reco=e_reco)

    assert len(rmf.e_true.nodes) == 80
    assert len(rmf.e_reco.nodes) == 15
    assert_quantity_allclose(rmf.data.data[53, 8], 0.056, atol=2e-2)

    rmf2 = make_mean_edisp(
        observations,
        position=position,
        e_true=e_true,
        e_reco=e_reco,
        low_reco_threshold=Energy(1, "TeV"),
        high_reco_threshold=Energy(60, "TeV"),
    )
    i2 = np.where(rmf2.data.evaluate(e_reco=Energy(0.8, "TeV")) != 0)[0]
    assert len(i2) == 0
    i2 = np.where(rmf2.data.evaluate(e_reco=Energy(61, "TeV")) != 0)[0]
    assert len(i2) == 0
    i = np.where(rmf.data.evaluate(e_reco=Energy(1.5, "TeV")) != 0)[0]
    i2 = np.where(rmf2.data.evaluate(e_reco=Energy(1.5, "TeV")) != 0)[0]
    assert_equal(i, i2)
    i = np.where(rmf.data.evaluate(e_reco=Energy(40, "TeV")) != 0)[0]
    i2 = np.where(rmf2.data.evaluate(e_reco=Energy(40, "TeV")) != 0)[0]
    assert_equal(i, i2)
