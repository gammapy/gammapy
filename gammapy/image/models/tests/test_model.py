# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.utils.data import get_pkg_data_filename
from ....extern import xmltodict


def test_model_xml_read_write():
    filename = get_pkg_data_filename('data/fermi_model.xml')
    sources = xmltodict.parse(open(filename).read())
    sources = sources['source_library']['source']
    assert sources[0]['@name'] == '3C 273'
    assert sources[0]['spectrum']['parameter'][1]['@name'] == 'Index'
    assert sources[0]['spectrum']['parameter'][1]['@value'] == '-2.1'
