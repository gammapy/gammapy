# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.units import Quantity
from astropy.tests.helper import pytest, assert_quantity_allclose
from ..fermi import SourceCatalog3FGL, SourceCatalog2FHL
from ...spectrum.models import PowerLaw, ExponentialCutoffPowerLaw, LogParabola
from ...utils.testing import requires_data



MODEL_TEST_DATA = [(0, PowerLaw, Quantity(1.4351261e-9, 'GeV-1 s -1 cm-2')),
                   (4, LogParabola, Quantity(8.3828599e-10, 'GeV-1 s -1 cm-2')),
                   (55, ExponentialCutoffPowerLaw, Quantity(7.4397387e-10, 'GeV-1 s -1 cm-2'))]

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

        assert len(flux_points) == 5

        desired = [5.10239849e-03, 4.79114673e-04, 3.81966743e-05,
                   2.09147089e-06, 8.16878884e-09]
        assert_allclose(flux_points['DIFF_FLUX'], desired)



    def test_flux_points_integral(self):
        assert len(self.source.flux_points_integral) == 5



@requires_data('gammapy-extra')
class TestSourceCatalog2FHL:
    def setup(self):
        self.cat = SourceCatalog2FHL()

    def test_main_table(self):
        assert len(self.cat.table) == 360

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 25


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


############
# Old stuff:
#
# from ..fermi import fetch_fermi_catalog, fetch_fermi_extended_sources
#
#
# # TODO: refactor (currently skipped)
# @requires_data('gammapy-extra')
# def _test_fetch_fermi_catalog():
#     n_hdu = len(fetch_fermi_catalog('3FGL'))
#     assert n_hdu == 6
#
#     n_sources = len(fetch_fermi_catalog('3FGL', 'LAT_Point_Source_Catalog'))
#     assert n_sources == 3034
#
#     n_hdu = len(fetch_fermi_catalog('2FGL'))
#     assert n_hdu == 5
#
#     n_sources = len(fetch_fermi_catalog('2FGL', 'LAT_Point_Source_Catalog'))
#     assert n_sources == 1873
#
#
# # TODO: refactor (currently skipped)
# @requires_data('gammapy-extra')
# def _test_fetch_fermi_extended_sources():
#     assert len(fetch_fermi_extended_sources('3FGL')) == 26
#     assert len(fetch_fermi_extended_sources('2FGL')) == 12
#     assert len(fetch_fermi_extended_sources('1FHL')) == 23
