# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest
from ...utils.testing import requires_dependency, requires_data, data_manager

productions = [
    dict(
        prod='hap-hd-prod2',
        datastore='hess-crab4-hd-hap-prod2',
        test_obs=23523,
        aeff_ref=267252.7018649852 * u.m ** 2,
        psf_type='psf_3gauss',
        psf_ref=106.31031643387723 / u.sr,
        edisp_ref=2.059754779846907,
    ),
    dict(
        prod='pa-release1',
        datastore='hess-crab4-pa',
        test_obs=23523,
        aeff_ref=207835.97395833334 * u.m ** 2,
        psf_type='psf_king',
        psf_ref=51.0044923171571 / u.sr,
        edisp_ref=2.783763964427291,
    ),
]


class FitsProductionTester:
    def __init__(self, prod):
        self.ref_dict = prod
        dm = data_manager()
        self.ds = dm[prod['datastore']]
        self.ref_energy = 1 * u.TeV
        self.ref_offset = 0.25 * u.deg
        self.ref_rad = np.arange(0, 2, 0.1) * u.deg
        self.ref_migra = 0.95
        self.obs = self.ds.obs(prod['test_obs'])

    def test_all(self):
        self.test_aeff()
        self.test_psf()
        self.test_edisp()

    def test_aeff(self):
        aeff = self.obs.load(hdu_type='aeff', hdu_class='aeff_2d')
        actual = aeff.data.evaluate(energy=self.ref_energy, offset=self.ref_offset)
        desired = self.ref_dict['aeff_ref']
        assert_quantity_allclose(actual, desired)

    def test_edisp(self):
        edisp = self.obs.load(hdu_type='edisp', hdu_class='edisp_2d')
        actual = edisp.data.evaluate(e_true=self.ref_energy, offset=self.ref_offset, migra=self.ref_migra)
        desired = self.ref_dict['edisp_ref']
        assert_quantity_allclose(actual, desired)

    def test_psf(self):
        psf = self.obs.load(hdu_type='psf', hdu_class=self.ref_dict['psf_type'])
        table_psf = psf.to_energy_dependent_table_psf(rad=self.ref_rad, theta=self.ref_offset)
        actual = table_psf.evaluate(energy=self.ref_energy)
        desired = self.ref_dict['psf_ref']
        assert_quantity_allclose(actual[0][4], desired)


@pytest.mark.parametrize('prod', productions)
@requires_dependency('yaml')
@requires_data('gammapy-extra')
def test_fits_prods(prod):
    tester = FitsProductionTester(prod)
    tester.test_all()
