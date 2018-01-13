# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing.utils import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as u
from ...utils.testing import requires_data, requires_dependency
from ...image import SkyImage
from ..hess import SourceCatalogHGPS


@requires_data('hgps')
class TestSourceCatalogHGPS:
    def setup_class(self):
        self.cat = SourceCatalogHGPS()

    def test_source_table(self):
        assert self.cat.name == 'hgps'
        assert len(self.cat.table) == 78

    def test_component_table(self):
        assert len(self.cat.components) == 98

    def test_associations_table(self):
        assert len(self.cat.associations) == 223

    @requires_dependency('scipy')
    @pytest.mark.parametrize('source_name', ['HESS J1837-069', 'HESS J1809-193', 'HESS J1841-055'])
    def test_large_scale_component(self, source_name):
        # This test compares the flux values from the LS model within source
        # regions with ones listed in the catalog, agreement is <1%
        ls_model = self.cat.large_scale_component

        source = self.cat[source_name]
        rspec = source.data['RSpec']
        npix = int(2.5 * rspec.value / 0.02)

        image = SkyImage.empty(
            xref=source.position.galactic.l.deg,
            yref=source.position.galactic.b.deg,
            nxpix=npix,
            nypix=npix,
        )
        coordinates = image.coordinates()
        image.data = ls_model.evaluate(coordinates)
        image.data *= image.solid_angle()

        mask = coordinates.separation(source.position) < rspec
        flux_ls = image.data[mask].sum()

        assert_quantity_allclose(flux_ls, source.data['Flux_Map_RSpec_LS'], rtol=1E-2)


@requires_data('hgps')
class TestSourceCatalogObjectHGPS:
    def setup_class(self):
        self.cat = SourceCatalogHGPS()
        self.source = self.cat['HESS J1843-033']

    def test_name(self):
        assert self.source.name == 'HESS J1843-033'

    def test_index(self):
        assert self.source.index == 64

    def test_data(self):
        data = self.source.data
        assert data['Source_Class'] == 'Unid'

    def test_str(self):
        ss = str(self.source)
        assert 'Source name          : HESS J1843-033' in ss
        assert 'Component HGPSC 083:' in ss

    def test_str_single_gauss(self):
        source = self.cat['HESS J1930+188']
        assert source.data['Spatial_Model'] == 'Gaussian'
        assert 'Spatial components   : HGPSC 097' in str(source)

    def test_str_multi_gauss(self):
        source = self.cat['HESS J1825-137']
        assert source.data['Spatial_Model'] == '3-Gaussian'
        assert 'Spatial components   : HGPSC 065, HGPSC 066, HGPSC 067' in str(source)

    def test_str_snr(self):
        source = self.cat['HESS J1713-397']
        assert source.data['Spatial_Model'] == 'Shell'
        assert 'Source name          : HESS J1713-397' in str(source)

    def test_pprint(self):
        self.source.pprint()

    def test_energy_range(self):
        energy_range = self.source.energy_range
        assert energy_range.unit == 'TeV'
        assert_allclose(energy_range.value, [0.21544346, 61.89658356])

    def test_spectral_model_pl(self):
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

    def test_spectral_model_ecpl(self):
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

    def test_spatial_model_point(self):
        source = self.cat['HESS J1826-148']
        model = source.spatial_model()
        assert_allclose(model.amplitude, 8.354304806121845e-13)

    def test_spatial_model_gaussian(self):
        source = self.cat['HESS J1119-614']
        model = source.spatial_model()
        assert_allclose(model.amplitude, 1.524557226374496e-11)

    def test_spatial_model_gaussian3(self):
        source = self.cat['HESS J1825-137']
        model = source.spatial_model()
        assert_allclose(model[0].amplitude, 3.662450902166903e-12)
        assert_allclose(model[1].amplitude, 1.2805462035898928e-11)
        assert_allclose(model[2].amplitude, 2.1553481912856457e-11)

    def test_spatial_model_gaussian_extern(self):
        # special test for the only extern source with a gaussian morphology
        source = self.cat['HESS J1801-233']
        model = source.spatial_model()
        assert_allclose(model.amplitude, 2.4881435269261268e-12)

    def test_spatial_model_shell(self):
        source = self.cat['Vela Junior']
        model = source.spatial_model()
        assert_allclose(model.amplitude, 2.3394972370498355e-11)
