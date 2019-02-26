# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from ..irf_reduce import (
    make_psf,
    make_mean_psf,
    make_mean_edisp,
    apply_containment_fraction,
)
from ..effective_area import EffectiveAreaTable
from ..psf_table import EnergyDependentTablePSF, TablePSF
from ...data import DataStore, Observations
from ...utils.testing import requires_data, assert_quantity_allclose
from ...utils.energy import Energy, EnergyBounds
from ...spectrum.models import PowerLaw


@pytest.fixture(scope="session")
def data_store():
    return DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")


@requires_data("gammapy-data")
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
            "psf_value_shape": (32, 1000),
            "psf_value": 25888.5047,
        },
        {
            "energy": EnergyBounds.equal_log_spacing(1, 10, 100, "TeV"),
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


@requires_data("gammapy-data")
def test_make_mean_psf(data_store):
    position = SkyCoord(83.63, 22.01, unit="deg")

    observations = data_store.get_observations([23523, 23526])
    psf = make_mean_psf(observations, position=position)

    assert not np.isnan(psf.psf_value.value).any()
    assert_allclose(psf.psf_value.value[22, 22], 12206.1665)


@requires_data("gammapy-data")
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


def test_apply_containment_fraction():
    n_edges_energy = 5
    energy = EnergyBounds.equal_log_spacing(0.1, 10.0, nbins=n_edges_energy, unit="TeV")
    area = np.ones(n_edges_energy) * 4 * u.m ** 2
    aeff = EffectiveAreaTable(energy.lower_bounds, energy.upper_bounds, data=area)

    nrad = 100
    rad = Angle(np.linspace(0, 0.5, nrad), "deg")
    psf_table = TablePSF.from_shape(shape="disk", width="0.2 deg", rad=rad)
    psf_values = (
        np.resize(psf_table.psf_value.value, (n_edges_energy, nrad))
        * psf_table.psf_value.unit
    )
    edep_psf_table = EnergyDependentTablePSF(
        aeff.energy.nodes, rad, psf_value=psf_values
    )

    new_aeff = apply_containment_fraction(aeff, edep_psf_table, Angle("0.1 deg"))

    assert_allclose(new_aeff.data.data.value, 1.0, rtol=5e-4)
    assert new_aeff.data.data.unit == "m2"
