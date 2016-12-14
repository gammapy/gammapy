# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.units import Quantity
from astropy.tests.helper import pytest, assert_quantity_allclose
from ..fermi import SourceCatalog3FGL, SourceCatalog2FHL
from ...spectrum.models import PowerLaw, LogParabola, ExponentialCutoffPowerLaw3FGL
from ...utils.testing import requires_data, requires_dependency

MODEL_TEST_DATA = [(0, PowerLaw, Quantity(1.4351261e-9, 'GeV-1 s -1 cm-2')),
                   (4, LogParabola, Quantity(8.3828599e-10, 'GeV-1 s -1 cm-2')),
                   (55, ExponentialCutoffPowerLaw3FGL, Quantity(1.8666925e-09, 'GeV-1 s-1 cm-2'))]

CRAB_NAMES_3FGL = ['Crab', '3FGL J0534.5+2201', '1FHL J0534.5+2201',
                   '2FGL J0534.5+2201', 'PSR J0534+2200', '0FGL J0534.6+2201']
CRAB_NAMES_2FHL = ['Crab', '3FGL J0534.5+2201i', '1FHL J0534.5+2201',
                   'TeV J0534+2200']


@requires_data('gammapy-extra')
class TestSourceCatalog3FGL:
    def test_main_table(self):
        cat = SourceCatalog3FGL()
        assert len(cat.table) == 3034

    def test_extended_sources(self):
        cat = SourceCatalog3FGL()
        table = cat.extended_sources_table
        assert len(table) == 25


@requires_data('gammapy-extra')
class TestFermi3FGLObject:
    def setup(self):
        self.cat = SourceCatalog3FGL()
        # Use 3FGL J0534.5+2201 (Crab) as a test source
        self.source_name = '3FGL J0534.5+2201'
        self.source = self.cat[self.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_index(self):
        assert self.source.index == 621

    def test_data(self):
        assert_allclose(self.source.data['Signif_Avg'], 30.669872283935547)

    def test_pprint(self):
        self.source.pprint()

    @pytest.mark.xfail
    def _test_plot_lightcurve(self):
        self.source.plot_lightcurve()

    def test_str(self):
        ss = str(self.source)
        assert 'Source: 3FGL J0534.5+2201' in ss
        assert 'RA (J2000)  : 83.63' in ss

    @pytest.mark.parametrize('index, model_type, desired', MODEL_TEST_DATA)
    def test_spectral_model(self, index, model_type, desired):
        energy = Quantity(1, 'GeV')
        model = self.cat[index].spectral_model
        assert isinstance(model, model_type)
        actual = model(energy)
        assert_quantity_allclose(actual, desired)

    def test_flux_points(self):
        flux_points = self.source.flux_points

        assert len(flux_points.table) == 5

        desired = [8.174943e-03, 7.676263e-04, 6.119782e-05, 3.350906e-06,
                   1.308784e-08]
        assert_allclose(flux_points.table['dnde'].data, desired, rtol=1E-5)

    @pytest.mark.parametrize('name', CRAB_NAMES_3FGL)
    def test_crab_alias(self, name):
        assert str(self.cat['Crab']) == str(self.cat[name])


@requires_data('gammapy-extra')
class TestSourceCatalog2FHL:
    def setup(self):
        self.cat = SourceCatalog2FHL()

    def test_main_table(self):
        assert len(self.cat.table) == 360

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 25

    @pytest.mark.parametrize('name', CRAB_NAMES_2FHL)
    def test_crab_alias(self, name):
        assert str(self.cat['Crab']) == str(self.cat[name])


@requires_data('gammapy-extra')
class TestFermi2FHLObject:
    def setup(self):
        cat = SourceCatalog2FHL()
        # Use 2FHL J0534.5+2201 (Crab) as a test source
        self.source_name = '2FHL J0534.5+2201'
        self.source = cat[self.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_spectral_model(self):
        model = self.source.spectral_model
        energy = Quantity(100, 'GeV')
        desired = Quantity(6.8700477298e-12, 'cm-2 GeV-1 s-1')
        assert_quantity_allclose(model(energy), desired)

    @requires_dependency('uncertainties')
    def test_spectrum(self):
        spectrum = self.source.spectrum
        assert "Fit result info" in str(spectrum)
