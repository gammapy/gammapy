# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing.utils import assert_allclose
from astropy import units as u
from ...utils.testing import requires_data, requires_dependency
from ..hess import SourceCatalogHGPS


@requires_data('hgps')
class TestSourceCatalogHGPS:
    def setup(self):
        self.cat = SourceCatalogHGPS()

    def test_source_table(self):
        assert self.cat.name == 'hgps'
        assert len(self.cat.table) == 78

    def test_component_table(self):
        assert len(self.cat.components) == 98

    def test_associations_table(self):
        assert len(self.cat.associations) == 223


@requires_data('hgps')
class TestSourceCatalogObjectHGPS:
    def setup(self):
        self.cat = SourceCatalogHGPS()
        self.source_name = 'HESS J1843-033'
        self.source = self.cat[self.source_name]

    def test_single_gauss(self):
        source = self.cat['HESS J1930+188']
        assert source.data['Spatial_Model'] == 'Gaussian'
        assert 'Spatial components   : HGPSC 097' in str(source)

    def test_multi_gauss(self):
        source = self.cat['HESS J1825-137']
        assert source.data['Spatial_Model'] == '3-Gaussian'
        assert 'Spatial components   : HGPSC 065, HGPSC 066, HGPSC 067' in str(source)

    def test_snr(self):
        source = self.cat['HESS J1713-397']
        assert source.data['Spatial_Model'] == 'Shell'
        assert 'Source name          : HESS J1713-397' in str(source)

    def test_name(self):
        assert self.source.name == self.source_name

    def test_index(self):
        assert self.source.index == 64

    def test_data(self):
        data = self.source.data
        assert data['Source_Class'] == 'Unid'

    def test_pprint(self):
        self.source.pprint()

    def test_str(self):
        ss = self.source.__str__()
        assert 'Source name          : HESS J1843-033' in ss
        assert 'Component HGPSC 083:' in ss

    def test_model(self):
        source = self.source
        model = source.spectral_model
        pars = model.parameters
        assert_allclose(pars['amplitude'].value, 9.140179932365378e-13)
        assert_allclose(pars['index'].value, 2.1513476371765137)
        assert_allclose(pars['reference'].value, 1.867810606956482)

        emin, emax = u.Quantity([1, 1e5], 'TeV')
        actual = model.integral(emin, emax).value
        desired = source.data['Flux_Spec_Int_1TeV'].value
        assert_allclose(actual, desired, rtol=0.01)

    def test_ecpl_model(self):
        source = self.cat['HESS J0835-455']
        model = source.spectral_model
        pars = model.parameters
        assert_allclose(pars['amplitude'].value, 6.408420542586617e-12)
        assert_allclose(pars['index'].value, 1.3543991614920847)
        assert_allclose(pars['reference'].value, 1.696938754239)
        assert_allclose(pars['lambda_'].value, 0.081517637)

        emin, emax = u.Quantity([1, 1e5], 'TeV')
        actual = model.integral(emin, emax).value
        desired = source.data['Flux_Spec_Int_1TeV'].value
        assert_allclose(actual, desired, rtol=0.01)

    @requires_dependency('matplotlib')
    def test_model_plot(self):
        model = self.source.spectral_model
        erange = [1, 10] * u.TeV
        model.plot(erange)

    def test_spatial_model_gaussian(self):
        source = self.cat['HESS J1119-614']
        model = source.spatial_model(emin=1 * u.TeV, emax=1e3 * u.TeV)
        actual = model.amplitude
        desired = 1.52453e-11
        assert_allclose(actual, desired, rtol=1e-3)

    def test_spatial_model_shell(self):
        source = self.cat['Vela Junior']
        model = source.spatial_model(emin=1 * u.TeV, emax=1e3 * u.TeV)
        actual = model.amplitude
        desired = 2.33949e-11
        assert_allclose(actual, desired, rtol=1e-3)

    def test_spatial_model_point(self):
        source = self.cat['HESS J1826-148']
        model = source.spatial_model(emin=1 * u.TeV, emax=1e3 * u.TeV)
        actual = model.amplitude
        desired = 8.353370e-13
        assert_allclose(actual, desired, rtol=1e-3)