# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest
from ..hawc import SourceCatalog2HWC
from ...utils.testing import requires_data, requires_dependency


@pytest.fixture(scope='session')
def hwc2_fhl():
    return SourceCatalog2HWC()


@requires_data('gammapy-extra')
class TestSourceCatalog2HWC:
    def test_source_table(self, hwc2_fhl):
        assert hwc2_fhl.name == '2hwc'
        assert len(hwc2_fhl.table) == 40


@requires_data('gammapy-extra')
class TestSourceCatalogObject2HWC:
    def test_data(self, hwc2_fhl):
        source = hwc2_fhl[0]

        assert source.data['source_name'] == '2HWC J0534+220'

    @requires_dependency('uncertainties')
    def test_spectra(self, hwc2_fhl):
        source0 = hwc2_fhl[0]
        source1 = hwc2_fhl[1]

        src0_mod0 = source0.spectral_model()[0]
        src1_mod0 = source1.spectral_model()[0]
        src1_mod1 = source1.spectral_model()[1]

        dnde_7TeV_src0, dnde_7TeV_src0_err = src0_mod0.evaluate_error(7 * u.TeV)
        dnde_7TeV_src1_mod0, dnde_7TeV_src1_mod0_err = src1_mod0.evaluate_error(7 * u.TeV)
        dnde_7TeV_src1_mod1, dnde_7TeV_src1_mod1_err = src1_mod1.evaluate_error(7 * u.TeV)

        assert_quantity_allclose(dnde_7TeV_src0, source0['dnde_spec0'], rtol=1e-3)
        assert_quantity_allclose(dnde_7TeV_src1_mod0, source1['dnde_spec0'], rtol=1e-3)
        assert_quantity_allclose(dnde_7TeV_src1_mod1, source1['dnde_spec1'], rtol=1e-3)

        assert_quantity_allclose(dnde_7TeV_src0_err, source0['dnde_spec0_err'], rtol=1e-3)
        assert_quantity_allclose(dnde_7TeV_src1_mod0_err, source1['dnde_spec0_err'], rtol=1e-3)
        assert_quantity_allclose(dnde_7TeV_src1_mod1_err, source1['dnde_spec1_err'], rtol=1e-3)
