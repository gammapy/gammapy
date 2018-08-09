# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from astropy.coordinates import Angle, SkyCoord
from astropy.units import Quantity
import astropy.units as u
from astropy.time import Time
from ...data import DataStore, ObservationList
from ...utils.testing import requires_data, requires_dependency
from ...utils.testing import assert_quantity_allclose, assert_time_allclose, assert_skycoord_allclose
from ...utils.energy import Energy
from ...utils.energy import EnergyBounds


@pytest.fixture(scope='session')
def data_store():
    return DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')


@requires_data('gammapy-extra')
def test_data_store_observation(data_store):
    """Test DataStoreObservation class"""
    obs = data_store.obs(23523)

    assert_time_allclose(obs.tstart, Time(51545.11740650318, scale='tt', format='mjd'))
    assert_time_allclose(obs.tstop, Time(51545.11740672924, scale='tt', format='mjd'))

    c = SkyCoord(83.63333129882812, 21.51444435119629, unit='deg')
    assert_skycoord_allclose(obs.pointing_radec, c)

    c = SkyCoord(26.533863067626953, 40.60616683959961, unit='deg')
    assert_skycoord_allclose(obs.pointing_altaz, c)

    c = SkyCoord(83.63333129882812, 22.01444435119629, unit='deg')
    assert_skycoord_allclose(obs.target_radec, c)


@requires_data('gammapy-extra')
def test_data_store_observation_to_observation_cta(data_store):
    from gammapy.data import EventList, GTI, ObservationCTA
    from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D, EnergyDependentMultiGaussPSF

    obs = data_store.obs(23523).to_observation_cta()

    assert type(obs) == ObservationCTA
    assert type(obs.obs_id) == int
    assert type(obs.gti) == GTI
    assert type(obs.events) == EventList
    assert type(obs.aeff) == EffectiveAreaTable2D
    assert type(obs.edisp) == EnergyDispersion2D
    assert type(obs.psf) == EnergyDependentMultiGaussPSF
    assert type(obs.pointing_radec) == SkyCoord
    assert type(obs.observation_live_time_duration) == Quantity
    assert type(obs.observation_dead_time_fraction) == np.float64


@requires_dependency('scipy')
@requires_data('gammapy-extra')
@pytest.mark.parametrize("pars,result", [
    (dict(energy=None, rad=None),
     dict(energy_shape=18, rad_shape=300, psf_energy=2.5178505859375 * u.TeV,
          psf_rad=0.05 * u.deg,
          psf_exposure=Quantity(6878545291473.34, "cm2 s"),
          psf_value=Quantity(1837.4367332530592, "1/sr"))),
    (dict(energy=EnergyBounds.equal_log_spacing(1, 10, 100, "TeV"), rad=None),
     dict(energy_shape=101, rad_shape=300,
          psf_energy=1.2589254117941673 * u.TeV, psf_rad=0.05 * u.deg,
          psf_exposure=Quantity(4622187644084.735, "cm2 s"),
          psf_value=Quantity(1682.8135627097995, "1/sr"))),
    (dict(energy=None, rad=Angle(np.arange(0, 2, 0.002), 'deg')),
     dict(energy_shape=18, rad_shape=1000,
          psf_energy=2.5178505859375 * u.TeV, psf_rad=0.02 * u.deg,
          psf_exposure=Quantity(6878545291473.34, "cm2 s"),
          psf_value=Quantity(20455.914082287516, "1/sr"))),
    (dict(energy=EnergyBounds.equal_log_spacing(1, 10, 100, "TeV"),
          rad=Angle(np.arange(0, 2, 0.002), 'deg')),
     dict(energy_shape=101, rad_shape=1000,
          psf_energy=1.2589254117941673 * u.TeV, psf_rad=0.02 * u.deg,
          psf_exposure=Quantity(4622187644084.735, "cm2 s"),
          psf_value=Quantity(25016.103907407552, "1/sr"))),
])
def test_make_psf(pars, result, data_store):
    position = SkyCoord(83.63, 22.01, unit='deg')

    obs1 = data_store.obs(23523)
    psf = obs1.make_psf(position=position, energy=pars["energy"], rad=pars["rad"])

    assert_allclose(psf.rad.shape, result["rad_shape"])
    assert_allclose(psf.energy.shape, result["energy_shape"])
    assert_allclose(psf.exposure.shape, result["energy_shape"])
    assert_allclose(psf.psf_value.shape, (result["energy_shape"],
                                          result["rad_shape"]))

    assert_quantity_allclose(psf.rad[10], result["psf_rad"])
    assert_quantity_allclose(psf.energy[10], result["psf_energy"])
    assert_quantity_allclose(psf.exposure[10], result["psf_exposure"])
    assert_quantity_allclose(psf.psf_value[10, 50], result["psf_value"])


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_make_psftable():
    position = SkyCoord(83.63, 22.01, unit='deg')
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')
    obs1 = data_store.obs(23523)
    obs2 = data_store.obs(23526)
    energy = EnergyBounds.equal_log_spacing(1, 10, 100, "TeV")
    energy_band = Energy([energy[0].value, energy[-1].value], energy.unit)

    psf1 = obs1.make_psf(position=position, energy=energy, rad=None)
    psf2 = obs2.make_psf(position=position, energy=energy, rad=None)
    psf1_int = psf1.table_psf_in_energy_band(energy_band, spectral_index=2.3)
    psf2_int = psf2.table_psf_in_energy_band(energy_band, spectral_index=2.3)
    obslist = ObservationList([obs1, obs2])
    psf_tot = obslist.make_mean_psf(position=position, energy=energy)
    psf_tot_int = psf_tot.table_psf_in_energy_band(energy_band, spectral_index=2.3)

    # Check that the mean PSF is consistent with the individual PSFs
    # (in this case the R68 of the mean PSF is in between the R68 of the individual PSFs)
    assert_quantity_allclose(psf1_int.containment_radius(0.68), Angle(0.1050259592154517, 'deg'))
    assert_quantity_allclose(psf2_int.containment_radius(0.68), Angle(0.09173224724288895, 'deg'))
    assert_quantity_allclose(psf_tot_int.containment_radius(0.68), Angle(0.09838901174312292, 'deg'))


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_make_mean_edisp(data_store):
    position = SkyCoord(83.63, 22.01, unit='deg')

    obs1 = data_store.obs(23523)
    obs2 = data_store.obs(23592)
    obslist = ObservationList([obs1, obs2])

    e_true = EnergyBounds.equal_log_spacing(0.01, 150, 80, "TeV")
    e_reco = EnergyBounds.equal_log_spacing(0.5, 100, 15, "TeV")
    rmf = obslist.make_mean_edisp(position=position, e_true=e_true,
                                  e_reco=e_reco)

    assert len(rmf.e_true.nodes) == 80
    assert len(rmf.e_reco.nodes) == 15
    assert_quantity_allclose(rmf.data.data[53, 8], 0.056, atol=2e-2)

    rmf2 = obslist.make_mean_edisp(position=position, e_true=e_true,
                                   e_reco=e_reco,
                                   low_reco_threshold=Energy(1, "TeV"),
                                   high_reco_threshold=Energy(60, "TeV"))
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
