# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_data, requires_dependency
from ..hawc import SourceCatalog2HWC


@pytest.fixture(scope='session')
def hawc_2hwc():
    return SourceCatalog2HWC()


@requires_data('gammapy-extra')
class TestSourceCatalog2HWC:
    def test_source_table(self, hawc_2hwc):
        assert hawc_2hwc.name == '2hwc'
        assert len(hawc_2hwc.table) == 40


@requires_data('gammapy-extra')
class TestSourceCatalogObject2HWC:
    def test_data(self, hawc_2hwc):
        source = hawc_2hwc[0]

        assert source.data['source_name'] == '2HWC J0534+220'

    @requires_dependency('uncertainties')
    def test_spectra(self, hawc_2hwc):
        source0 = hawc_2hwc[0]
        source1 = hawc_2hwc[1]

        src0_mod0 = source0.spectral_models[0]
        src1_mod0 = source1.spectral_models[0]
        src1_mod1 = source1.spectral_models[1]

        dnde_7TeV_src0, dnde_7TeV_src0_err = src0_mod0.evaluate_error(7 * u.TeV)
        dnde_7TeV_src1_mod0, dnde_7TeV_src1_mod0_err = src1_mod0.evaluate_error(7 * u.TeV)
        dnde_7TeV_src1_mod1, dnde_7TeV_src1_mod1_err = src1_mod1.evaluate_error(7 * u.TeV)

        assert_quantity_allclose(dnde_7TeV_src0, source0.data['spec0_dnde'], rtol=1e-3)
        assert_quantity_allclose(dnde_7TeV_src1_mod0, source1.data['spec0_dnde'], rtol=1e-3)
        assert_quantity_allclose(dnde_7TeV_src1_mod1, source1.data['spec1_dnde'], rtol=1e-3)

        assert_quantity_allclose(dnde_7TeV_src0_err, source0.data['spec0_dnde_err'], rtol=1e-3)
        assert_quantity_allclose(dnde_7TeV_src1_mod0_err, source1.data['spec0_dnde_err'], rtol=1e-3)
        assert_quantity_allclose(dnde_7TeV_src1_mod1_err, source1.data['spec1_dnde_err'], rtol=1e-3)
