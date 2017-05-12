# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose, pytest
from astropy import units as u
from ...utils.testing import requires_data, requires_dependency
from ..gammacat import SourceCatalogGammaCat

SOURCES = [
    {
        'name': 'Vela X',

        'dnde_1TeV': 1.36e-11 * u.Unit('cm-2 s-1 TeV-1'),
        'dnde_1TeV_err': 7.531e-13 * u.Unit('cm-2 s-1 TeV-1'),
        'flux_1TeV': 2.104e-11 * u.Unit('cm-2 s-1'),
        'flux_1TeV_err': 1.973e-12 * u.Unit('cm-2 s-1'),
        'eflux_1_10TeV': 9.265778680255336e-11 * u.Unit('erg cm-2 s-1'),
        'eflux_1_10TeV_err': 9.590978299538194e-12 * u.Unit('erg cm-2 s-1'),

        'n_flux_points': 24,
    },
    {
        'name': 'HESS J1848-018',

        'dnde_1TeV': 3.7e-12 * u.Unit('cm-2 s-1 TeV-1'),
        'dnde_1TeV_err': 4e-13 * u.Unit('cm-2 s-1 TeV-1'),
        'flux_1TeV': 2.056e-12 * u.Unit('cm-2 s-1'),
        'flux_1TeV_err': 3.187e-13 * u.Unit('cm-2 s-1'),
        'eflux_1_10TeV': 6.235650344765057e-12 * u.Unit('erg cm-2 s-1'),
        'eflux_1_10TeV_err': 1.2210315515569183e-12 * u.Unit('erg cm-2 s-1'),

        'n_flux_points': 11,
    },
    {
        'name': 'HESS J1813-178',

        'dnde_1TeV': 2.678e-12 * u.Unit('cm-2 s-1 TeV-1'),
        'dnde_1TeV_err': 2.55e-13 * u.Unit('cm-2 s-1 TeV-1'),
        'flux_1TeV': 2.457e-12 * u.Unit('cm-2 s-1'),
        'flux_1TeV_err': 3.692e-13 * u.Unit('cm-2 s-1'),
        'eflux_1_10TeV': 8.923614018939419e-12 * u.Unit('erg cm-2 s-1'),
        'eflux_1_10TeV_err': 1.4613807070890267e-12 * u.Unit('erg cm-2 s-1'),

        'n_flux_points': 13,
    },
]


@pytest.fixture(scope='session')
def gammacat():
    filename = '$GAMMAPY_EXTRA/datasets/catalogs/gammacat.fits.gz'
    return SourceCatalogGammaCat(filename=filename)


@requires_data('gammapy-extra')
@requires_data('gamma-cat')
class TestSourceCatalogGammaCat:
    def test_source_table(self, gammacat):
        assert gammacat.name == 'gamma-cat'
        assert len(gammacat.table) == 162

    def test_w28_alias_names(self, gammacat):
        names = ['W28', 'HESS J1801-233', 'W 28', 'SNR G6.4-0.1',
                 'SNR G006.4-00.1', 'GRO J1801-2320']
        for name in names:
            assert str(gammacat[name]) == str(gammacat['W28'])

    def test_sort_table(self):
        name = 'HESS J1848-018'
        sort_keys = ['ra', 'dec', 'reference_id']
        for sort_key in sort_keys:
            # this test modifies the catalog, so we make a copy
            cat = gammacat()
            before = str(cat[name])
            cat.table.sort(sort_key)
            after = str(cat[name])
            assert before == after

    def test_to_source_library(self, gammacat):
        sources = gammacat.to_source_library()
        source = sources.source_list[0]

        assert len(sources.source_list) == 72
        assert source.source_name == 'CTA 1'
        assert_allclose(source.spectral_model.parameters['Index'].value, -2.2)


@requires_data('gammapy-extra')
@requires_data('gamma-cat')
class TestSourceCatalogObjectGammaCat:
    def test_data(self, gammacat):
        source = gammacat[0]

        assert isinstance(source.data, OrderedDict)
        assert source.data['common_name'] == 'CTA 1'
        assert_quantity_allclose(source.data['dec'], 72.782997 * u.deg)

    @pytest.mark.parametrize('ref', SOURCES, ids=lambda _: _['name'])
    def test_spectral_model(self, gammacat, ref):
        source = gammacat[ref['name']]
        spectral_model = source.spectral_model

        e_min, e_max, e_inf = [1, 10, 1e10] * u.TeV

        dnde_1TeV = spectral_model(e_min)
        flux_1TeV = spectral_model.integral(emin=e_min, emax=e_inf)
        eflux_1_10TeV = spectral_model.energy_flux(emin=e_min, emax=e_max).to('erg cm-2 s-1')

        assert_quantity_allclose(dnde_1TeV, ref['dnde_1TeV'], rtol=1e-3)
        assert_quantity_allclose(flux_1TeV, ref['flux_1TeV'], rtol=1e-3)
        assert_quantity_allclose(eflux_1_10TeV, ref['eflux_1_10TeV'], rtol=1e-3)

    @requires_dependency('uncertainties')
    @pytest.mark.parametrize('ref', SOURCES, ids=lambda _: _['name'])
    def test_spectral_model_err(self, gammacat, ref):
        source = gammacat[ref['name']]
        spectral_model = source.spectral_model

        e_min, e_max, e_inf = [1, 10, 1e10] * u.TeV

        dnde_1TeV = spectral_model.evaluate_error(e_min)
        flux_1TeV = spectral_model.integral_error(emin=e_min, emax=e_inf)
        eflux_1_10TeV = spectral_model.energy_flux_error(emin=e_min, emax=e_max).to('erg cm-2 s-1')

        assert_quantity_allclose(dnde_1TeV[0], ref['dnde_1TeV'], rtol=1e-3)
        assert_quantity_allclose(flux_1TeV[0], ref['flux_1TeV'], rtol=1e-3)
        assert_quantity_allclose(eflux_1_10TeV[0], ref['eflux_1_10TeV'], rtol=1e-3)

        assert_quantity_allclose(dnde_1TeV[1], ref['dnde_1TeV_err'], rtol=1e-3)
        assert_quantity_allclose(flux_1TeV[1], ref['flux_1TeV_err'], rtol=1e-3)
        assert_quantity_allclose(eflux_1_10TeV[1], ref['eflux_1_10TeV_err'], rtol=1e-3)

    @pytest.mark.parametrize('ref', SOURCES, ids=lambda _: _['name'])
    def test_flux_points(self, gammacat, ref):
        source = gammacat[ref['name']]

        flux_points = source.flux_points

        assert len(flux_points.table) == ref['n_flux_points']
