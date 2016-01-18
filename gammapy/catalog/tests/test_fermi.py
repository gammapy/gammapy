# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...utils.testing import requires_data
from astropy.tests.helper import pytest
from ..fermi import SourceCatalog3FGL, SourceCatalog2FHL


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
        cat = SourceCatalog3FGL()
        # Use 3FGL J0534.5+2201 (Crab) as a test source
        self.source_name = '3FGL J0534.5+2201'
        self.source = cat[self.source_name]

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

    @pytest.mark.xfail
    def test_plot_spectrum(self):
        self.source.plot_spectrum()

    def test_print_info(self):
        self.source.print_info()


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
