# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
import pytest
from astropy import units as u
from ...utils.testing import requires_data, requires_dependency
from ..gammacat import SourceCatalogGammaCat
from ..gammacat import GammaCatResource, GammaCatResourceIndex

SOURCES = [
    {
        'name': 'Vela X',

        'spec_type': 'ecpl',
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

        'spec_type': 'pl',
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

        'spec_type': 'pl2',
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
    filename = '$GAMMAPY_EXTRA/datasets/catalogs/gammacat/gammacat.fits.gz'
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
            cat.table.sort(sort_key)
            assert cat[name].name == name

    def test_to_source_library(self, gammacat):
        sources = gammacat.to_source_library()
        source = sources.source_list[0]

        assert len(sources.source_list) == 74
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
    def test_str(self, gammacat, ref):
        ss = gammacat[ref['name']]._info_spectral_fit()
        print(ss)
        assert ss == STRING_REPRESENTATION[ref['name']]

    def test_data_python_dict(self, gammacat):
        source = gammacat[0]
        data = source._data_python_dict
        assert type(data['ra']) == float
        assert data['ra'] == 1.649999976158142
        assert type(data['sed_e_min']) == list
        assert type(data['sed_e_min'][0]) == float
        assert_allclose(data['sed_e_min'][0], 0.5600000023841858)

    @pytest.mark.parametrize('ref', SOURCES, ids=lambda _: _['name'])
    def test_spectral_model(self, gammacat, ref):
        source = gammacat[ref['name']]
        spectral_model = source.spectral_model

        assert source.data['spec_type'] == ref['spec_type']

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

        dnde_1TeV, dnde_1TeV_err = spectral_model.evaluate_error(e_min)
        flux_1TeV, flux_1TeV_err = spectral_model.integral_error(emin=e_min, emax=e_inf)
        eflux_1_10TeV, eflux_1_10TeV_err = spectral_model.energy_flux_error(emin=e_min, emax=e_max).to('erg cm-2 s-1')

        assert_quantity_allclose(dnde_1TeV, ref['dnde_1TeV'], rtol=1e-3)
        assert_quantity_allclose(flux_1TeV, ref['flux_1TeV'], rtol=1e-3)
        assert_quantity_allclose(eflux_1_10TeV, ref['eflux_1_10TeV'], rtol=1e-3)

        assert_quantity_allclose(dnde_1TeV_err, ref['dnde_1TeV_err'], rtol=1e-3)
        assert_quantity_allclose(flux_1TeV_err, ref['flux_1TeV_err'], rtol=1e-3)
        assert_quantity_allclose(eflux_1_10TeV_err, ref['eflux_1_10TeV_err'], rtol=1e-3)

    @pytest.mark.parametrize('ref', SOURCES, ids=lambda _: _['name'])
    def test_flux_points(self, gammacat, ref):
        source = gammacat[ref['name']]

        flux_points = source.flux_points

        assert len(flux_points.table) == ref['n_flux_points']


class TestGammaCatResource:
    def setup(self):
        self.resource = GammaCatResource(source_id=42, reference_id='2010A&A...516A..62A', file_id=2)
        self.global_id = '42|2010A&A...516A..62A|2|none'

    def test_global_id(self):
        assert self.resource.global_id == self.global_id

    def test_eq(self):
        resource1 = self.resource
        resource2 = GammaCatResource(source_id=42, reference_id='2010A&A...516A..62A')

        assert resource1 == resource1
        assert resource1 != resource2

    def test_lt(self):
        resource = GammaCatResource(source_id=42, reference_id='2010A&A...516A..62A', file_id=2)

        assert resource < GammaCatResource(source_id=43, reference_id='2010A&A...516A..62A', file_id=2)
        assert resource < GammaCatResource(source_id=42, reference_id='2010A&A...516A..62B', file_id=2)
        assert resource < GammaCatResource(source_id=42, reference_id='2010A&A...516A..62A', file_id=3)

        assert resource > GammaCatResource(source_id=41, reference_id='2010A&A...516A..62A', file_id=2)

    def test_repr(self):
        expected = ("GammaCatResource(source_id=42, reference_id='2010A&A...516A..62A', "
                    "file_id=2, type='none', location='none')")
        assert repr(self.resource) == expected

    def test_to_dict(self):
        expected = OrderedDict([
            ('source_id', 42), ('reference_id', '2010A&A...516A..62A'),
            ('file_id', 2), ('type', 'none'), ('location', 'none'),
        ])
        assert self.resource.to_dict() == expected

    def test_dict_roundtrip(self):
        actual = GammaCatResource.from_dict(self.resource.to_dict())
        assert actual == self.resource


class TestGammaCatResourceIndex:
    def setup(self):
        self.resource_index = GammaCatResourceIndex([
            GammaCatResource(source_id=99, reference_id='2014ApJ...780..168A'),
            GammaCatResource(source_id=42, reference_id='2010A&A...516A..62A', file_id=2, type='sed'),
            GammaCatResource(source_id=42, reference_id='2010A&A...516A..62A', file_id=1),
        ])

    def test_repr(self):
        assert repr(self.resource_index) == 'GammaCatResourceIndex(n_resources=3)'

    def test_eq(self):
        resource_index1 = self.resource_index
        resource_index2 = GammaCatResourceIndex(resource_index1.resources[:-1])

        assert resource_index1 == resource_index1
        assert resource_index1 != resource_index2

    def test_unique_source_ids(self):
        expected = [42, 99]
        assert self.resource_index.unique_source_ids == expected

    def test_unique_reference_ids(self):
        expected = ['2010A&A...516A..62A', '2014ApJ...780..168A']
        assert self.resource_index.unique_reference_ids == expected

    def test_global_ids(self):
        expected = [
            '99|2014ApJ...780..168A|-1|none',
            '42|2010A&A...516A..62A|2|sed',
            '42|2010A&A...516A..62A|1|none',
        ]
        assert self.resource_index.global_ids == expected

    def test_sort(self):
        expected = [
            '42|2010A&A...516A..62A|1|none',
            '42|2010A&A...516A..62A|2|sed',
            '99|2014ApJ...780..168A|-1|none',
        ]
        assert self.resource_index.sort().global_ids == expected

    def test_to_list(self):
        result = self.resource_index.to_list()
        assert isinstance(result, list)
        assert len(result) == 3

    def test_list_roundtrip(self):
        data = self.resource_index.to_list()
        actual = GammaCatResourceIndex.from_list(data)
        assert actual == self.resource_index

    def test_to_table(self):
        table = self.resource_index.to_table()
        assert len(table) == 3
        assert table.colnames == ['source_id', 'reference_id', 'file_id', 'type', 'location']

    def test_table_roundtrip(self):
        table = self.resource_index.to_table()
        actual = GammaCatResourceIndex.from_table(table)
        assert actual == self.resource_index

    @requires_dependency('pandas')
    def test_to_pandas(self):
        df = self.resource_index.to_pandas()
        df2 = df.query('source_id == 42')
        assert len(df2) == 2

    @requires_dependency('pandas')
    def test_pandas_roundtrip(self):
        df = self.resource_index.to_pandas()
        actual = GammaCatResourceIndex.from_pandas(df)
        assert actual == self.resource_index

    @requires_dependency('pandas')
    def test_query(self):
        resource_index = self.resource_index.query('type == "sed" and source_id == 42')
        assert len(resource_index.resources) == 1
        assert resource_index.resources[0].global_id == '42|2010A&A...516A..62A|2|sed'


STRING_REPRESENTATION = {
    'Vela X': """
*** Spectral info ***

Significance    : 27.900
Livetime        : 53.100 h

Spectrum type   : ecpl
norm            : 1.46e-11 +- 8e-13 (stat) +- 3e-12 (sys) cm-2 s-1 TeV-1
index           : 1.32 +- 0.06 (stat) +- 0.12 (sys)
e_cut           : 14.0 +- 1.6 (stat) +- 2.6 (stat) TeV
reference       : 1.0 TeV

Energy range         : (0.75, nan) TeV
theta                : 1.2 deg


Derived fluxes:
Spectral model norm (1 TeV)    : 1.36e-11 +- 7.53e-13 (stat) cm-2 s-1 TeV-1
Integrated flux (>1 TeV)       : 2.1e-11 +- 1.97e-12 (stat) cm-2 s-1
Integrated flux (>1 TeV)       : 101.425 +- 9.511 (% Crab)
Integrated flux (1-10 TeV)     : 9.27e-11 +- 9.59e-12 (stat) erg cm-2 s-1
""",
    'HESS J1813-178': """
*** Spectral info ***

Significance    : 13.500
Livetime        : 9.700 h

Spectrum type   : pl2
flux            : 1.42e-11 +- 1.1e-12 (stat) +- 3e-13 (sys) cm-2 s-1
index           : 2.09 +- 0.08 (stat)
e_min           : 0.2 TeV
e_max           : nan TeV

Energy range         : (nan, nan) TeV
theta                : 0.15 deg


Derived fluxes:
Spectral model norm (1 TeV)    : 2.68e-12 +- 2.55e-13 (stat) cm-2 s-1 TeV-1
Integrated flux (>1 TeV)       : 2.46e-12 +- 3.69e-13 (stat) cm-2 s-1
Integrated flux (>1 TeV)       : 11.844 +- 1.780 (% Crab)
Integrated flux (1-10 TeV)     : 8.92e-12 +- 1.46e-12 (stat) erg cm-2 s-1
""",
    'HESS J1848-018': """
*** Spectral info ***

Significance    : 9.000
Livetime        : 50.000 h

Spectrum type   : pl
norm            : 3.7e-12 +- 4e-13 (stat) +- 7e-13 (syst) cm-2 s-1 TeV-1
index           : 2.8 +- 0.2 (stat) +- 0.2 (sys)
reference       : 1.0 TeV

Energy range         : (0.9, 12.0) TeV
theta                : 0.2 deg


Derived fluxes:
Spectral model norm (1 TeV)    : 3.7e-12 +- 4e-13 (stat) cm-2 s-1 TeV-1
Integrated flux (>1 TeV)       : 2.06e-12 +- 3.19e-13 (stat) cm-2 s-1
Integrated flux (>1 TeV)       : 9.909 +- 1.536 (% Crab)
Integrated flux (1-10 TeV)     : 6.24e-12 +- 1.22e-12 (stat) erg cm-2 s-1
"""
}
