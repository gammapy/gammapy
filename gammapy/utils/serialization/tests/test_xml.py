# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.utils.data import get_pkg_data_filename
from ...scripts import make_path
from ....cube import SourceLibrary

def test_xml_to_source_library():
    filename = get_pkg_data_filename('data/fermi_model.xml')
    sourcelib = SourceLibrary.from_xml(filename)
    assert len(sourcelib.skymodels) == 4

    # TODO: add more tests

