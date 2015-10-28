# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import requires_data
from astropy.tests.helper import pytest
from ..fermi import fetch_fermi_catalog, fetch_fermi_extended_sources


# TODO: refactor (currently skipped)
@requires_data('gammapy-extra')
def _test_fetch_fermi_catalog():
    n_hdu = len(fetch_fermi_catalog('3FGL'))
    assert n_hdu == 6

    n_sources = len(fetch_fermi_catalog('3FGL', 'LAT_Point_Source_Catalog'))
    assert n_sources == 3034

    n_hdu = len(fetch_fermi_catalog('2FGL'))
    assert n_hdu == 5

    n_sources = len(fetch_fermi_catalog('2FGL', 'LAT_Point_Source_Catalog'))
    assert n_sources == 1873


# TODO: refactor (currently skipped)
@requires_data('gammapy-extra')
def _test_fetch_fermi_extended_sources():
    assert len(fetch_fermi_extended_sources('3FGL')) == 26
    assert len(fetch_fermi_extended_sources('2FGL')) == 12
    assert len(fetch_fermi_extended_sources('1FHL')) == 23

# @requires_data('gammapy-extra')
# def test_get_source_catalog():
#     cat = get_source_catalog('3fgl')

    # source = cat['3FGL J0000.1+6545']
    # assert source.name == '3FGL J0000.1+6545'
    # assert_allclose(source.significance, 3.5)
    # assert_allclose(source.row_index, 42)
    #
    # source = cat.source_index(42)
    # assert source.name == '3FGL J0000.1+6545'
    # assert_allclose(source.significance, 3.5)

# @requires_data('gammapy-extra')
# class TestFermi3FGLObject:
#
#     test_source = '3FGL J0000.1+6545'
#
#     def test_plot_lightcurve(self):
#         source = Fermi3FGLObject(self.test_source)
#         source.plot_lightcurve()
#
#     def test_plot_spectrum(self):
#         source = Fermi3FGLObject(self.test_source)
#         source.plot_spectrum()
#
#     def test_info(self):
#         source = Fermi3FGLObject(self.test_source)
#         info = source.info()
#         assert self.test_source in info
