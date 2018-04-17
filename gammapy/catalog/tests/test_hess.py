# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import Counter
import pytest
from numpy.testing.utils import assert_allclose
from astropy import units as u
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_data, requires_dependency
from ...image import SkyImage
from ...spectrum.models import PowerLaw, ExponentialCutoffPowerLaw
from ..hess import SourceCatalogHGPS


@pytest.fixture(scope='session')
def cat():
    return SourceCatalogHGPS('$GAMMAPY_EXTRA/datasets/catalogs/hgps_catalog_v1.fits.gz')


@requires_data('gammapy-extra')
class TestSourceCatalogHGPS:

    @staticmethod
    def test_source_table(cat):
        assert cat.name == 'hgps'
        assert len(cat.table) == 78

    @staticmethod
    def test_table_components(cat):
        assert len(cat.table_components) == 98

    @staticmethod
    def test_table_associations(cat):
        assert len(cat.table_associations) == 223

    @staticmethod
    def test_table_identifications(cat):
        assert len(cat.table_identifications) == 31

    @staticmethod
    @requires_dependency('scipy')
    @pytest.mark.parametrize('source_name', ['HESS J1837-069', 'HESS J1809-193', 'HESS J1841-055'])
    def test_large_scale_component(cat, source_name):
        # This test compares the flux values from the LS model within source
        # regions with ones listed in the catalog, agreement is <1%
        ls_model = cat.large_scale_component

        source = cat[source_name]
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


@requires_data('gammapy-extra')
class TestSourceCatalogObjectHGPS:

    @pytest.fixture(scope='class')
    def source(self, cat):
        return cat['HESS J1843-033']

    @staticmethod
    @pytest.mark.slow
    def test_all_sources(cat):
        """Check that properties and methods work for all sources,
        i.e. don't raise an error."""
        for source in cat:
            str(source)
            source.energy_range
            source.spectral_model_type
            source.spectral_model()
            source.spatial_model_type
            source.is_pointlike
            source.spatial_model()
            source.flux_points

    @staticmethod
    def test_basics(source):
        assert source.name == 'HESS J1843-033'
        assert source.index == 64
        data = source.data
        assert data['Source_Class'] == 'Unid'
        assert 'SourceCatalogObjectHGPS' in repr(source)

        ss = str(source)
        assert 'Source name          : HESS J1843-033' in ss
        assert 'Component HGPSC 083:' in ss

    @staticmethod
    def test_str(cat):
        source = cat['HESS J1930+188']
        assert source.data['Spatial_Model'] == 'Gaussian'
        assert 'Spatial components   : HGPSC 097' in str(source)

        source = cat['HESS J1825-137']
        assert source.data['Spatial_Model'] == '3-Gaussian'
        assert 'Spatial components   : HGPSC 065, HGPSC 066, HGPSC 067' in str(source)

        source = cat['HESS J1713-397']
        assert source.data['Spatial_Model'] == 'Shell'
        assert 'Source name          : HESS J1713-397' in str(source)

    @staticmethod
    def test_energy_range(source):
        energy_range = source.energy_range
        assert energy_range.unit == 'TeV'
        assert_allclose(energy_range.value, [0.21544346, 61.89658356])

    @staticmethod
    def test_spectral_model_type(cat):
        spec_types = Counter([_.spectral_model_type for _ in cat])
        assert spec_types == {'pl': 66, 'ecpl': 12}

    @staticmethod
    @requires_dependency('uncertainties')
    def test_spectral_model_pl(cat):
        source = cat['HESS J1843-033']

        model = source.spectral_model()

        assert isinstance(model, PowerLaw)
        pars = model.parameters
        assert_allclose(pars['amplitude'].value, 9.140179932365378e-13)
        assert_allclose(pars['index'].value, 2.1513476371765137)
        assert_allclose(pars['reference'].value, 1.867810606956482)

        val, err = model.integral_error(1 * u.TeV, 1e5 * u.TeV).value
        assert_allclose(val, source.data['Flux_Spec_Int_1TeV'].value, rtol=0.01)
        assert_allclose(err, source.data['Flux_Spec_Int_1TeV_Err'].value, rtol=0.01)

    @staticmethod
    @requires_dependency('uncertainties')
    def test_spectral_model_ecpl(cat):
        source = cat['HESS J0835-455']

        model = source.spectral_model()
        assert isinstance(model, ExponentialCutoffPowerLaw)

        pars = model.parameters
        assert_allclose(pars['amplitude'].value, 6.408420542586617e-12)
        assert_allclose(pars['index'].value, 1.3543991614920847)
        assert_allclose(pars['reference'].value, 1.696938754239)
        assert_allclose(pars['lambda_'].value, 0.081517637)

        val, err = model.integral_error(1 * u.TeV, 1e5 * u.TeV).value
        assert_allclose(val, source.data['Flux_Spec_Int_1TeV'].value, rtol=0.01)
        assert_allclose(err, source.data['Flux_Spec_Int_1TeV_Err'].value, rtol=0.01)

        model = source.spectral_model('pl')
        assert isinstance(model, PowerLaw)

        pars = model.parameters
        assert_allclose(pars['amplitude'].value, 1.833056926733856e-12)
        assert_allclose(pars['index'].value, 1.8913707)
        assert_allclose(pars['reference'].value, 3.0176312923431396)

        val, err = model.integral_error(1 * u.TeV, 1e5 * u.TeV).value
        assert_allclose(val, source.data['Flux_Spec_PL_Int_1TeV'].value, rtol=0.01)
        assert_allclose(err, source.data['Flux_Spec_PL_Int_1TeV_Err'].value, rtol=0.01)

    @staticmethod
    def test_spatial_model_type(cat):
        morph_types = Counter([_.spatial_model_type for _ in cat])
        assert morph_types == {'gaussian': 52, '2-gaussian': 8, 'shell': 7, 'point-like': 6, '3-gaussian': 5}

    @staticmethod
    def test_spatial_model_point(cat):
        source = cat['HESS J1826-148']
        model = source.spatial_model()
        assert_allclose(model.amplitude, 8.354304806121845e-13)
        assert_allclose(model.x_0, 16.882482528686523)
        assert_allclose(model.y_0, -1.2889292240142822)

    @staticmethod
    def test_spatial_model_gaussian(cat):
        source = cat['HESS J1119-614']
        model = source.spatial_model()
        assert_allclose(model.amplitude, 1.524557226374496e-11)
        assert_allclose(model.x_mean, -6.78719177e+01)
        assert_allclose(model.y_mean, -5.33235371e-01)
        assert_allclose(model.x_stddev, 9.78596658e-02)
        assert_allclose(model.y_stddev, 9.78596658e-02)
        assert_allclose(model.theta, 0)

        bbox = model.bounding_box
        assert_allclose(bbox, [[-1.07146, 0.00499], [-68.41014, -67.33368]], atol=0.001)

    @staticmethod
    def test_spatial_model_gaussian2(cat):
        source = cat['HESS J1843-033']
        models = source.spatial_model()

        model = models[0]
        assert_allclose(model.amplitude, 1.44536430e-11)
        assert_allclose(model.x_mean, 2.90472164e+01)
        assert_allclose(model.y_mean, 2.43896767e-01)
        assert_allclose(model.x_stddev, 1.24991007e-01)
        assert_allclose(model.y_stddev, 1.24991007e-01)
        assert_allclose(model.theta, 0)

        model = models[1]
        assert_allclose(model.amplitude, 4.91294805e-12)
        assert_allclose(model.x_mean, 2.87703781e+01)
        assert_allclose(model.y_mean, -7.27819949e-02)
        assert_allclose(model.x_stddev, 2.29470655e-01)
        assert_allclose(model.y_stddev, 2.29470655e-01)
        assert_allclose(model.theta, 0)

        bbox = model.bounding_box
        assert_allclose(bbox, [[-1.33487, 1.18930], [27.50829, 30.03246]], atol=0.001)

    @staticmethod
    def test_spatial_model_gaussian3(cat):
        source = cat['HESS J1825-137']
        model = source.spatial_model()
        assert_allclose(model[0].amplitude, 3.662450902166903e-12)
        assert_allclose(model[1].amplitude, 1.2805462035898928e-11)
        assert_allclose(model[2].amplitude, 2.1553481912856457e-11)

    @staticmethod
    def test_spatial_model_gaussian_extern(cat):
        # special test for the only extern source with a gaussian morphology
        source = cat['HESS J1801-233']
        model = source.spatial_model()
        assert_allclose(model.amplitude, 2.4881435269261268e-12)
        assert_allclose(model.x_mean, 6.65688896e+00)
        assert_allclose(model.y_mean, -2.67688125e-01)
        assert_allclose(model.x_stddev, 1.70000002e-01)
        assert_allclose(model.y_stddev, 1.70000002e-01)
        assert_allclose(model.theta, 0)

    @staticmethod
    def test_spatial_model_shell(cat):
        source = cat['Vela Junior']
        model = source.spatial_model()
        assert_allclose(model.amplitude, 2.33949724e-11)
        assert_allclose(model.x_0, -9.37126160e+01)
        assert_allclose(model.y_0, -1.24326038e+00)
        assert_allclose(model.r_in, 9.50000000e-01)
        assert_allclose(model.width, 5.00000000e-02)


@requires_data('gammapy-extra')
class TestSourceCatalogObjectHGPSComponent:

    @staticmethod
    def test_get_by_row_idx(cat):
        # Row index starts at 0, component numbers at 1
        # Thus we expect `HGPSC 084` at row 83
        c = cat.gaussian_component(83)
        assert c.name == 'HGPSC 084'

    @staticmethod
    def test_it(cat):
        c = cat['HESS J1843-033'].components[1]
        assert c.name == 'HGPSC 084'
        assert c.index == 83
        assert 'SourceCatalogObjectHGPSComponent' in repr(c)
