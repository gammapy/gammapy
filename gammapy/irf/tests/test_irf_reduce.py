# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from gammapy.data import DataStore, Observations
from gammapy.irf import (
    EffectiveAreaTable,
    EnergyDependentTablePSF,
    EnergyDispersion,
    TablePSF,
    apply_containment_fraction,
    compute_energy_thresholds,
    make_mean_edisp,
    make_mean_psf,
    make_psf,
)
from gammapy.utils.energy import energy_logspace
from gammapy.utils.testing import assert_quantity_allclose, requires_data


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
            "energy": energy_logspace(1, 10, 101, "TeV"),
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
            "energy": energy_logspace(1, 10, 101, "TeV"),
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


@requires_data("gammapy-data")
def test_compute_thresholds_from_crab_data():
    """Obs read from file"""
    arffile = "$GAMMAPY_DATA/joint-crab/spectra/hess/arf_obs23523.fits"
    rmffile = "$GAMMAPY_DATA/joint-crab/spectra/hess/rmf_obs23523.fits"

    aeff = EffectiveAreaTable.read(arffile)
    edisp = EnergyDispersion.read(rmffile)

    thresh_lo, thresh_hi = compute_energy_thresholds(
        aeff=aeff,
        edisp=edisp,
        method_lo="energy_bias",
        method_hi="none",
        bias_percent_lo=10,
        bias_percent_hi=10,
    )

    assert_allclose(thresh_lo.to("TeV").value, 0.9174, rtol=1e-4)
    assert_allclose(thresh_hi.to("TeV").value, 100.0, rtol=1e-4)


def test_compute_thresholds_from_parametrization():
    energy = np.logspace(-2, 2.0, 100) * u.TeV
    aeff = EffectiveAreaTable.from_parametrization(energy=energy)
    edisp = EnergyDispersion.from_gauss(e_true=energy, e_reco=energy, sigma=0.2, bias=0)

    thresh_lo, thresh_hi = compute_energy_thresholds(
        aeff=aeff,
        edisp=edisp,
        method_lo="area_max",
        method_hi="area_max",
        area_percent_lo=10,
        area_percent_hi=90,
    )

    assert_allclose(thresh_lo.to("TeV").value, 0.18557, rtol=1e-4)
    assert_allclose(thresh_hi.to("TeV").value, 43.818, rtol=1e-4)

    thresh_lo, thresh_hi = compute_energy_thresholds(
        aeff=aeff, edisp=edisp, method_hi="area_max", area_percent_hi=70
    )

    assert_allclose(thresh_hi.to("TeV").value, 100.0, rtol=1e-4)
