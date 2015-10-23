# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.tests.helper import remote_data
from ..registry import get_source_catalog


@remote_data
def test_get_source_catalog():
    cat = get_source_catalog('3FGL')

    source = cat['3FGL J0000.1+6545']
    assert source.name == '3FGL J0000.1+6545'
    assert_allclose(source.significance, 3.5)
    assert_allclose(source.row_index, 42)

    source = cat.source_index(42)
    assert source.name == '3FGL J0000.1+6545'
    assert_allclose(source.significance, 3.5)


# @remote_data
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
