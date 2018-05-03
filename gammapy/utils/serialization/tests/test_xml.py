# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...testing import requires_data
from ...scripts import make_path
from ....cube import SourceLibrary
from ....spectrum import models as spectral
from ....image import models as spatial
from ...serialization import xml_to_source_library, UnknownModelError
import pytest

@requires_data('gammapy-extra')
def test_xml_to_source_library():
    filename = '$GAMMAPY_EXTRA/test_datasets/models/fermi_model.xml'
    sourcelib = SourceLibrary.from_xml(filename)

    assert len(sourcelib.skymodels) == 4

    model1 = sourcelib.skymodels[0]
    assert isinstance(model1.spectral_model, spectral.PowerLaw)
    assert isinstance(model1.spatial_model, spatial.SkyPointSource)

    pars1 = model1.parameters
    assert pars1['index'].value == -2.1
    assert pars1['index'].unit == ''
    assert pars1['index'].parmax == -1.0
    assert pars1['index'].parmin == -5.0
    assert pars1['index'].frozen == False

    assert pars1['lon_0'].value == 187.25
    assert pars1['lon_0'].unit == 'deg'
    assert pars1['lon_0'].parmax == 360
    assert pars1['lon_0'].parmin == -360
    assert pars1['lon_0'].frozen == True

    model3 = sourcelib.skymodels[2]
    assert isinstance(model3.spectral_model, spectral.PowerLaw)
    assert isinstance(model3.spatial_model, spatial.SkyDiffuseMap)

    pars3 = model3.parameters
    assert pars3['index'].value == 0
    assert pars3['index'].unit == ''
    assert pars3['index'].parmax == 1
    assert pars3['index'].parmin == -1
    assert pars3['index'].frozen == True

    assert pars3['norm'].value == 1.0
    assert pars3['norm'].unit == ''
    assert pars3['norm'].parmax == 1e3
    assert pars3['norm'].parmin == 1e-3
    assert pars3['norm'].frozen == True

    # TODO: Test Evaluate combined model as soon as '+' works for SkyModel


@requires_data('gammapy-extra')
@pytest.mark.parametrize('filenames',[[
     '$GAMMAPY_EXTRA/test_datasets/models/fermi_model.xml',
     '$GAMMAPY_EXTRA/test_datasets/models/shell.xml',
]])
def test_models(filenames, tmpdir):
    outfile = tmpdir / 'models_out.xml'
    for filename in filenames:
        sourcelib = SourceLibrary.from_xml(filename)
        sourcelib.to_xml(outfile)

        sourcelib_roundtrip = SourceLibrary.from_xml(outfile)

        for model, model_roundtrip in zip(sourcelib.skymodels,
                                          sourcelib_roundtrip.skymodels):
            assert str(model) == str(model_roundtrip)


def test_xml_errors():
    xml = '<?xml version="1.0" standalone="no"?>\n'
    xml += '<source_library title="broken source library">\n'
    xml += '    <source name="CrabShell" type="ExtendedSource">\n'
    xml += '        <spatialModel type="ElefantShapedSource">\n'
    xml += '            <parameter name="RA" value="1" scale="1" min="1" max="1" free="1"/>\n'
    xml += '        </spatialModel>\n'
    xml += '    </source>\n'
    xml += '</source_library>'

    with pytest.raises(UnknownModelError):
        model = xml_to_source_library(xml)

    # TODO: Think about a more elaborate XML validation scheme
