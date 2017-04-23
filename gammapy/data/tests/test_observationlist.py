# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_data, requires_dependency
from ...utils.energy import EnergyBounds, Energy
from ...data import DataStore, ObservationList


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
