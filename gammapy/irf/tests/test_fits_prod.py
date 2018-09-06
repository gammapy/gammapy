# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from ...data import DataStore
from ...utils.testing import requires_dependency, requires_data

# TODO: clean up these FITS production tests
# There is some duplication with `gammapy/data/tests/test_data_store.py`
# This should either be moved there, or to a separate test file
# called "test legacy" or "test high level" or something like that
# The goal would be to not accidentally break support for old files
productions = [
    dict(
        prod="pa-release1",
        datastore="$GAMMAPY_EXTRA/datasets/hess-crab4-pa",
        test_obs=23523,
        aeff_ref=207835.9,
        psf_type="psf_king",
        psf_ref=51.004,
        edisp_ref=2.783,
    ),
    dict(
        prod="hap-hd-prod2",
        datastore="$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2",
        test_obs=23523,
        aeff_ref=267252.7,
        psf_type="psf_3gauss",
        psf_ref=106.310,
        edisp_ref=2.059,
    ),
    dict(
        prod="hess-dl3-dr1",
        datastore="$GAMMAPY_EXTRA/datasets/hess-dl3-dr1",
        test_obs=23523,
        aeff_ref=229490.7,
        psf_type="psf_table",
        psf_ref=237.759,
        edisp_ref=2.247,
    ),
]


class FitsProductionTester:
    def __init__(self, prod):
        self.ref_dict = prod
        self.ds = DataStore.from_dir(prod["datastore"])
        self.ref_energy = 1 * u.TeV
        self.ref_offset = 0.25 * u.deg
        self.ref_rad = np.arange(0, 2, 0.1) * u.deg
        self.ref_migra = 0.95
        self.obs = self.ds.obs(prod["test_obs"])

    def test_all(self):
        self.test_aeff()
        self.test_psf()
        self.test_edisp()

    def test_aeff(self):
        aeff = self.obs.load(hdu_type="aeff", hdu_class="aeff_2d")
        actual = aeff.data.evaluate(energy=self.ref_energy, offset=self.ref_offset)
        desired = self.ref_dict["aeff_ref"]
        assert actual.unit == "m2"
        assert_allclose(actual.value, desired, rtol=1e-3)

    def test_edisp(self):
        edisp = self.obs.load(hdu_type="edisp", hdu_class="edisp_2d")
        actual = edisp.data.evaluate(
            e_true=self.ref_energy, offset=self.ref_offset, migra=self.ref_migra
        )
        desired = self.ref_dict["edisp_ref"]
        assert actual.unit == ""
        assert_allclose(actual.value, desired, rtol=1e-3)

    def test_psf(self):
        psf = self.obs.load(hdu_type="psf", hdu_class=self.ref_dict["psf_type"])
        table_psf = psf.to_energy_dependent_table_psf(
            rad=self.ref_rad, theta=self.ref_offset
        )
        actual = table_psf.evaluate(energy=self.ref_energy)
        desired = self.ref_dict["psf_ref"]
        assert actual.unit == "sr-1"
        assert_allclose(actual.value[0][4], desired, rtol=1e-3)


@pytest.mark.parametrize("prod", productions)
@requires_data("gammapy-extra")
@requires_dependency("scipy")
def test_fits_prods(prod):
    tester = FitsProductionTester(prod)
    tester.test_all()
