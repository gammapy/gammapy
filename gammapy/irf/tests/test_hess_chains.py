# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.tests.helper import pytest, assert_quantity_allclose
from ...datasets.core import GammapyExtraNotFoundError
from ...utils.scripts import make_path
from ...utils.testing import requires_dependency, requires_data, data_manager
from ...irf import EffectiveAreaTable2D, EnergyDispersion2D

@requires_data('gammapy-extra')
class TestHAPHDExporter:
    def setup(self):
        dm = data_manager()
        ref_store = 'hess-crab4-hd-hap-prod2'
        self.ds = dm[ref_store]
        self.obs_id = 23523
        self.ref_energy = 1 * u.TeV
        self.ref_offset = 0.25 * u.deg
        self.obs = self.ds.obs(self.obs_id)

    def test_aeff(self):
        aeff = self.obs.load(hdu_type='aeff', hdu_class='aeff_2d')
        actual = aeff.evaluate(energy=self.ref_energy, offset=self.ref_offset)
        desired = 267252.7018649852 * u.m ** 2
        assert_quantity_allclose(actual, desired)


