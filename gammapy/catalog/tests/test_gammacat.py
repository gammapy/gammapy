# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose, pytest
from astropy import units as u
from ...utils.testing import requires_data, requires_dependency
from ...utils.energy import Energy
from ..gammacat import SourceCatalogGammaCat

SOURCES = ['Vela X', 'HESS J1848-018', 'HESS J1813-178']

DESIRED_SM = [
    {
        'flux_at_1TeV': 1.36e-11 * u.Unit('1 / (cm2 TeV s)'),
        'flux_at_1TeV_err': 7.531e-13 * u.Unit('1 / (cm2 TeV s)'),
        'flux_above_1TeV': 2.104e-11 * u.Unit('1 / (cm2 s)'),
        'flux_above_1TeV_err': 1.973e-12 * u.Unit('1 / (cm2 s)'),
        'eflux_1_10TeV': 5.783e-11 * u.Unit('TeV / (cm2 s)'),
        'eflux_1_10TeV_err': 5.986e-12 * u.Unit('TeV / (cm2 s)'),
    },
    {
        'flux_at_1TeV': 3.7e-12 * u.Unit('1 / (cm2 TeV s)'),
        'flux_at_1TeV_err': 4e-13 * u.Unit('1 / (cm2 TeV s)'),
        'flux_above_1TeV': 2.056e-12 * u.Unit('1 / (cm2 s)'),
        'flux_above_1TeV_err': 3.187e-13 * u.Unit('1 / (cm2 s)'),
        'eflux_1_10TeV': 3.892e-12 * u.Unit('TeV / (cm2 s)'),
        'eflux_1_10TeV_err': 7.621e-13 * u.Unit('TeV / (cm2 s)'),
    },
    {
        'flux_at_1TeV': 2.678e-12 * u.Unit('1 / (cm2 TeV s)'),
        'flux_at_1TeV_err': 2.55e-13 * u.Unit('1 / (cm2 TeV s)'),
        'flux_above_1TeV': 2.457e-12 * u.Unit('1 / (cm2 s)'),
        'flux_above_1TeV_err': 3.692e-13 * u.Unit('1 / (cm2 s)'),
        'eflux_1_10TeV': 5.5697e-12 * u.Unit('TeV / (cm2 s)'),
        'eflux_1_10TeV_err': 9.121e-13 * u.Unit('TeV / (cm2 s)'),
    },
]

DESIRED_FP = [{'N': 24},
              {'N': 11},
              {'N': 13}]

DESIRED_BF = [
    {
        'energy_sum': 40.8695 * u.TeV,
        'flux_lo_sum': 3.965e-11 * u.Unit('1 / (cm2 s TeV)'),
        'flux_hi_sum': 4.555e-11 * u.Unit('1 / (cm2 s TeV)')
    },
    {
        'energy_sum': 40.8695 * u.TeV,
        'flux_lo_sum': 6.2880e-12 * u.Unit('1 / (cm2 s TeV)'),
        'flux_hi_sum': 8.168e-12 * u.Unit('1 / (cm2 s TeV)'),
    },
    {
        'energy_sum': 40.8695 * u.TeV,
        'flux_lo_sum': 5.691e-12 * u.Unit('1 / (cm2 s TeV)'),
        'flux_hi_sum': 7.181e-12 * u.Unit('1 / (cm2 s TeV)'),
    },
]

W28_NAMES = ['W28', 'HESS J1801-233', 'W 28', 'SNR G6.4-0.1', 'SNR G006.4-00.1',
             'GRO J1801-2320']

SORT_KEYS = ['ra', 'dec', 'reference_id']


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

    @pytest.mark.parametrize('name', W28_NAMES)
    def test_w28_alias_names(self, gammacat, name):
        assert str(gammacat[name]) == str(gammacat['W28'])

    @pytest.mark.parametrize(['name', 'key'], zip(SOURCES, SORT_KEYS))
    def test_sort_table(self, name, key):
        # this test modifies the catalog, so we make a copy
        cat = gammacat()
        before = str(cat[name])
        cat.table.sort(key)
        after = str(cat[name])
        assert before == after

    def test_to_source_library(self, gammacat):
        sources = gammacat.to_source_library()
        source = sources.source_list[0]
        assert len(sources.source_list) == 69
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

    @pytest.mark.parametrize(['name', 'desired'], zip(SOURCES, DESIRED_SM))
    def test_spectral_model(self, gammacat, name, desired):
        source = gammacat[name]
        spectral_model = source.spectral_model

        emin, emax = [1, 10] * u.TeV
        einf = 1e10 * u.TeV
        flux_at_1TeV = spectral_model(emin)
        flux_above_1TeV = spectral_model.integral(emin=emin, emax=einf)
        eflux_1_10TeV = spectral_model.energy_flux(emin=emin, emax=emax)

        assert_quantity_allclose(flux_at_1TeV, desired['flux_at_1TeV'], rtol=1e-3)
        assert_quantity_allclose(flux_above_1TeV, desired['flux_above_1TeV'], rtol=1e-3)
        assert_quantity_allclose(eflux_1_10TeV, desired['eflux_1_10TeV'], rtol=1e-3)

    @requires_dependency('uncertainties')
    @pytest.mark.parametrize(['name', 'desired'], zip(SOURCES, DESIRED_SM))
    def test_spectral_model_err(self, gammacat, name, desired):
        source = gammacat[name]
        spectral_model = source.spectral_model

        emin, emax = [1, 10] * u.TeV
        einf = 1e10 * u.TeV
        flux_at_1TeV = spectral_model.evaluate_error(emin)
        flux_above_1TeV = spectral_model.integral_error(emin=emin, emax=einf)
        eflux_1_10TeV = spectral_model.energy_flux_error(emin=emin, emax=emax)

        assert_quantity_allclose(flux_at_1TeV[0], desired['flux_at_1TeV'], rtol=1e-3)
        assert_quantity_allclose(flux_above_1TeV[0], desired['flux_above_1TeV'], rtol=1e-3)
        assert_quantity_allclose(eflux_1_10TeV[0], desired['eflux_1_10TeV'], rtol=1e-3)

        assert_quantity_allclose(flux_at_1TeV[1], desired['flux_at_1TeV_err'], rtol=1e-3)
        assert_quantity_allclose(flux_above_1TeV[1], desired['flux_above_1TeV_err'], rtol=1e-3)
        assert_quantity_allclose(eflux_1_10TeV[1], desired['eflux_1_10TeV_err'], rtol=1e-3)

    @pytest.mark.parametrize(['name', 'desired'], zip(SOURCES, DESIRED_FP))
    def test_flux_points(self, gammacat, name, desired):
        source = gammacat[name]

        assert name == source.name
        flux_points = source.flux_points
        assert len(flux_points.table) == desired['N']

    @requires_dependency('uncertainties')
    @pytest.mark.parametrize(['name', 'desired'], zip(SOURCES, DESIRED_BF))
    def test_butterfly(self, gammacat, name, desired):
        source = gammacat[name]
        emin, emax = [1, 10] * u.TeV
        energies = Energy.equal_log_spacing(emin, emax, 10)

        flux, flux_err = source.spectral_model.evaluate_error(energies)
        flux_lo = flux - flux_err
        flux_hi = flux + flux_err
        assert_quantity_allclose(energies.sum(), desired['energy_sum'], rtol=1e-3)
        assert_quantity_allclose(flux_lo.sum(), desired['flux_lo_sum'], rtol=1e-3)
        assert_quantity_allclose(flux_hi.sum(), desired['flux_hi_sum'], rtol=1e-3)
