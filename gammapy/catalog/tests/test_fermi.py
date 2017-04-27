# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.tests.helper import pytest, assert_quantity_allclose
from ..fermi import SourceCatalog3FGL, SourceCatalog2FHL, SourceCatalog1FHL, SourceCatalog3FHL
from ...spectrum.models import (PowerLaw, LogParabola, ExponentialCutoffPowerLaw3FGL,
                                PLSuperExpCutoff3FGL)
from ...utils.testing import requires_data, requires_dependency

MODEL_TEST_DATA_3FGL = [
    (0, PowerLaw, u.Quantity(1.4351261e-9, 'cm-2 s-1 GeV-1'),
                  u.Quantity(2.1356270e-10, 'cm-2 s-1 GeV-1')),
    (4, LogParabola, u.Quantity(8.3828599e-10, 'cm-2 s-1 GeV-1'),
                     u.Quantity(2.6713238e-10, 'cm-2 s-1 GeV-1')),
    (55, ExponentialCutoffPowerLaw3FGL, u.Quantity(1.8666925e-09, 'cm-2 s-1 GeV-1'),
                                        u.Quantity(2.2068837e-10, 'cm-2 s-1 GeV-1'),),
    (960, PLSuperExpCutoff3FGL, u.Quantity(1.6547128794756733e-06, 'cm-2 s-1 GeV-1'),
                                u.Quantity(1.6621504e-11, 'cm-2 s-1 MeV-1')),
]

MODEL_TEST_DATA_3FHL = [
    (352, PowerLaw, u.Quantity(5.79746841775092e-12, 'cm-2 s-1 GeV-1')),
    (1444, LogParabola, u.Quantity(2.056998292908196e-12, 'cm-2 s-1 GeV-1')),
]

CRAB_NAMES_3FGL = ['Crab', '3FGL J0534.5+2201', '1FHL J0534.5+2201',
                   '2FGL J0534.5+2201', 'PSR J0534+2200', '0FGL J0534.6+2201']
CRAB_NAMES_1FHL = ['Crab', '1FHL J0534.5+2201', '2FGL J0534.5+2201', 'PSR J0534+2200',
                   'Crab']
CRAB_NAMES_2FHL = ['Crab', '3FGL J0534.5+2201i', '1FHL J0534.5+2201',
                   'TeV J0534+2200']
CRAB_NAMES_3FHL = ['Crab Pulsar', '3FHL J0534.5+2201', 'PSR J0534+2200',
                   '3FGL J0534.5+2201i']


@requires_data('gammapy-extra')
class TestFermi3FGLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FGL()
        # Use 3FGL J0534.5+2201 (Crab) as a test source
        cls.source_name = '3FGL J0534.5+2201'
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_index(self):
        assert self.source.index == 621

    def test_data(self):
        assert_allclose(self.source.data['Signif_Avg'], 30.669872283935547)

    def test_pprint(self):
        # TODO: add assert on output
        self.source.pprint()

    @pytest.mark.xfail
    def _test_plot_lightcurve(self):
        self.source.plot_lightcurve()

    def test_str(self):
        ss = str(self.source)
        assert '3FGL J0534.5+2201' in ss # Source name
        assert '83.637 deg' in ss # RA

    @pytest.mark.parametrize('index, model_type, desired, desired_err', MODEL_TEST_DATA_3FGL)
    def test_spectral_model(self, index, model_type, desired, desired_err):
        energy = u.Quantity(1, 'GeV')
        model = self.cat[index].spectral_model
        assert isinstance(model, model_type)
        actual = model(energy)
        assert_quantity_allclose(actual, desired)

    @requires_dependency('uncertainties')
    @pytest.mark.parametrize('index, model_type, desired, desired_err', MODEL_TEST_DATA_3FGL)
    def test_spectral_model_error(self, index, model_type, desired, desired_err):
        energy = u.Quantity(1, 'GeV')
        model = self.cat[index].spectral_model
        actual = model.evaluate_error(energy)
        assert_quantity_allclose(actual[1], desired_err)

    def test_flux_points(self):
        flux_points = self.source.flux_points

        assert len(flux_points.table) == 5
        assert 'flux_ul' in flux_points.table.colnames

        desired = [8.174943e-03, 7.676263e-04, 6.119782e-05, 3.350906e-06, 1.308784e-08]
        assert_allclose(flux_points.table['dnde'].data, desired, rtol=1e-5)

    def test_lightcurve(self):
        lc = self.source.lightcurve
        assert len(lc) == 48
        assert set(['TIME_MIN', 'TIME_MAX', 'FLUX', 'FLUX_ERR']).issubset(lc.colnames)

        point = lc[0]

        assert point['TIME_MIN'].fits == '2008-08-02T00:33:19.000(UTC)'
        assert point['TIME_MAX'].fits == '2008-09-01T10:31:04.625(UTC)'
        assert_quantity_allclose(point['FLUX'], 2.38471262e-06 * u.Unit('cm-2 s-1'))
        assert_quantity_allclose(point['FLUX_ERR'], 8.07127023e-08 * u.Unit('cm-2 s-1'))

    @pytest.mark.parametrize('name', CRAB_NAMES_3FGL)
    def test_crab_alias(self, name):
        assert str(self.cat['Crab']) == str(self.cat[name])


@requires_data('gammapy-extra')
class TestFermi1FHLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog1FHL()
        # Use 1FHL J0534.5+2201 (Crab) as a test source
        cls.source_name = '1FHL J0534.5+2201'
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_spectral_model(self):
        model = self.source.spectral_model
        energy = u.Quantity(100, 'GeV')
        desired = u.Quantity(4.7717464131e-12, 'cm-2 GeV-1 s-1')
        assert_quantity_allclose(model(energy), desired)

    def test_flux_points(self):
        # test flux point on  PKS 2155-304
        src = self.cat['1FHL J0153.1+7515']
        flux_points = src.flux_points
        actual = flux_points.table['flux']
        desired = [5.523017e-11, np.nan, np.nan] * u.Unit('cm-2 s-1')
        assert_quantity_allclose(actual, desired)

        actual = flux_points.table['flux_ul']
        desired = [np.nan, 2.081589e-11, 1.299698e-11] * u.Unit('cm-2 s-1')
        assert_quantity_allclose(actual, desired, rtol=1e-5)


@requires_data('gammapy-extra')
class TestFermi2FHLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog2FHL()
        # Use 2FHL J0534.5+2201 (Crab) as a test source
        cls.source_name = '2FHL J0534.5+2201'
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_spectral_model(self):
        model = self.source.spectral_model
        energy = u.Quantity(100, 'GeV')
        desired = u.Quantity(6.8700477298e-12, 'cm-2 GeV-1 s-1')
        assert_quantity_allclose(model(energy), desired)

    def test_flux_points(self):
        # test flux point on  PKS 2155-304
        src = self.cat['PKS 2155-304']
        flux_points = src.flux_points
        actual = flux_points.table['flux']
        desired = [2.866363e-10, 6.118736e-11, np.nan] * u.Unit('cm-2 s-1')
        assert_quantity_allclose(actual, desired)

        actual = flux_points.table['flux_ul']
        desired = [np.nan, np.nan, 6.470300e-12] * u.Unit('cm-2 s-1')
        assert_quantity_allclose(actual, desired)


@requires_data('gammapy-extra')
class TestFermi3FHLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FHL()
        # Use 3FHL J0534.5+2201 (Crab) as a test source
        cls.source_name = '3FHL J0534.5+2201'
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_index(self):
        assert self.source.index == 352

    def test_data(self):
        assert_allclose(self.source.data['Signif_Avg'], 168.64082)

    def test_pprint(self):
        self.source.pprint()

    def test_str(self):
        ss = str(self.source)
        assert 'Source: 3FHL J0534.5+2201' in ss
        assert 'RA (J2000)  : 83.63' in ss

    @pytest.mark.parametrize('index, model_type, desired', MODEL_TEST_DATA_3FHL)
    def test_spectral_model(self, index, model_type, desired):
        energy = u.Quantity(100, 'GeV')
        model = self.cat[index].spectral_model
        assert isinstance(model, model_type)
        actual = model(energy)
        assert_quantity_allclose(actual, desired)

    def test_flux_points(self):
        flux_points = self.source.flux_points

        assert len(flux_points.table) == 5
        assert 'flux_ul' in flux_points.table.colnames

        desired = [5.12440652532e-07, 7.37024993524e-08, 9.04493849264e-09, 7.68135443661e-10, 4.30737078315e-11]
        assert_allclose(flux_points.table['dnde'].data, desired, rtol=1e-5)

    @pytest.mark.parametrize('name', CRAB_NAMES_3FHL)
    def test_crab_alias(self, name):
        assert str(self.cat['Crab Pulsar']) == str(self.cat[name])


@requires_data('gammapy-extra')
class TestSourceCatalog3FGL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FGL()

    def test_main_table(self):
        assert len(self.cat.table) == 3034

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 25


@requires_data('gammapy-extra')
class TestSourceCatalog1FHL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog1FHL()

    def test_main_table(self):
        assert len(self.cat.table) == 514

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 18

    @pytest.mark.parametrize('name', CRAB_NAMES_1FHL)
    def test_crab_alias(self, name):
        assert str(self.cat['Crab']) == str(self.cat[name])


@requires_data('gammapy-extra')
class TestSourceCatalog2FHL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog2FHL()

    def test_main_table(self):
        assert len(self.cat.table) == 360

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 25

    @pytest.mark.parametrize('name', CRAB_NAMES_2FHL)
    def test_crab_alias(self, name):
        assert str(self.cat['Crab']) == str(self.cat[name])


@requires_data('gammapy-extra')
class TestSourceCatalog3FHL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FHL()

    def test_main_table(self):
        assert len(self.cat.table) == 1558

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 55
