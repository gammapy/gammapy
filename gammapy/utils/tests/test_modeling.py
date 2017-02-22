# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ..testing import requires_data
from ..modeling import Parameter, SourceLibrary


def test_parameter():
    data = dict(name='spam', val=99., unit='ham')
    par = Parameter.from_dict(data)
    xml = par.to_xml()
    assert xml == '        <parameter name="spam" value="99.0" unit="ham"/>'


@requires_data('gammapy-extra')
def test_source_library_old_xml_file():
    filename = '$GAMMAPY_EXTRA/test_datasets/models/shell.xml'
    sources = SourceLibrary.read(filename)

    assert len(sources.source_list) == 2

    source = sources.source_list[0]
    assert source.source_name == 'CrabShell'
    assert_allclose(source.spectral_model.parameters['Index'].value, 2.48)

    xml = sources.to_xml()
    assert 'sources' in xml


@requires_data('gammapy-extra')
def test_source_library_new_xml_file():
    filename = '$GAMMAPY_EXTRA/test_datasets/models/ctadc_skymodel_gps_sources_bright.xml'
    sources = SourceLibrary.read(filename)

    assert len(sources.source_list) == 47

    source = sources.source_list[0]
    assert source.source_name == 'HESS J0632+057'
    assert_allclose(source.spectral_model.parameters['Index'].value, -2.5299999713897705)

    xml = sources.to_xml()
    assert 'sources' in xml

    # TODO: add more asserts for other spectral & spatial models
    # (or split those out into separate XML strings or files for better unit testing)
