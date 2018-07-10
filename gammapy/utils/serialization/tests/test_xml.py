# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...testing import requires_data, requires_dependency
from ...scripts import make_path
from ....cube import SourceLibrary
from ....spectrum import models as spectral
from ....image import models as spatial
from ...serialization import xml_to_source_library, UnknownModelError
import pytest

def test_basic():

    xml_str = '''<?xml version="1.0" encoding="utf-8"?>
    <source_library title="source library">
        <source name="3C 273" type="PointSource">
            <spectrum type="PowerLaw">;i
                <parameter free="1" max="1000.0" min="0.001" name="Prefactor" scale="1e-09" value="10"></parameter>
                <parameter free="1" max="-1.0" min="-5.0" name="Index" scale="1.0" value="-2.1"></parameter>
                <parameter free="0" max="2000.0" min="30.0" name="Scale" scale="1.0" value="100.0"></parameter>
            </spectrum>
            <spatialModel type="PointSource">
                <parameter free="0" max="360" min="-360" name="RA" scale="1.0" value="187.25"></parameter>
                <parameter free="0" max="90" min="-90" name="DEC" scale="1.0" value="2.17"></parameter>
            </spatialModel>
        </source>
    </source_library>
    '''

    sourcelib = xml_to_source_library(xml_str)

    assert len(sourcelib.skymodels) == 1

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

def test_complex():

    xml_str = '''<?xml version="1.0" encoding="utf-8"?>
    <source_library title="source library">
        <source name="Source 1" type="PointSource">
            <spectrum type="PowerLaw">
                <parameter free="1" max="1000.0" min="0.001" name="Prefactor" scale="1e-09" value="10" />
                <parameter free="1" max="1.0" min="5.0" name="Index" scale="1.0" value="2.1"/>
                <parameter free="0" max="2000.0" min="30.0" name="Scale" scale="1.0" value="100.0"/>
            </spectrum>
            <spatialModel type="PointSource">
                <parameter free="0" max="360" min="-360" name="GLON" scale="1.0" value="0.5"/>
                <parameter free="0" max="90" min="-90" name="GLAT" scale="1.0" value="1.0"/>
            </spatialModel>
        </source>
        <source name="Source 2" type="RadialGaussian">
            <spectrum type="PowerLaw">
                <parameter free="1" max="1000.0" min="0.001" name="Prefactor" scale="1e-09" value="10" />
                <parameter free="1" max="1.0" min="5.0" name="Index" scale="1.0" value="2.1"/>
                <parameter free="0" max="2000.0" min="30.0" name="Scale" scale="1.0" value="100.0"/>
            </spectrum>
            <spatialModel type="RadialGaussian">
                <parameter free="0" max="360" min="-360" name="GLON" scale="1.0" value="1.0"/>
                <parameter free="0" max="90" min="-90" name="GLAT" scale="1.0" value="0.5"/>
                <parameter free="1" max="10" min="0.01" name="Sigma" scale="1" value="0.2"/>
            </spatialModel>
        </source>
        <source name="Source 3" type="RadialDisk">
            <spectrum type="PowerLaw">
                <parameter free="1" max="1000.0" min="0.001" name="Prefactor" scale="1e-09" value="10" />
                <parameter free="1" max="-1.0" min="-5.0" name="Index" scale="1.0" value="-2.1"/>
                <parameter free="0" max="2000.0" min="30.0" name="Scale" scale="1.0" value="100.0"/>
            </spectrum>
            <spatialModel type="RadialDisk">
                <parameter free="0" max="360" min="-360" name="GLON" scale="1.0" value="358"/>
                <parameter free="0" max="90" min="-90" name="GLAT" scale="1.0" value="-0.5"/>
                <parameter free="1" max="10" min="0.01" name="Radius" scale="1" value="0.5"/>                
            </spatialModel>
        </source>
        <source name="Source 4" type="RadialShell">
            <spectrum type="PowerLaw">
                <parameter free="1" max="1000.0" min="0.001" name="Prefactor" scale="1e-09" value="10" />
                <parameter free="1" max="-1.0" min="-5.0" name="Index" scale="-1.0" value="-2.1"/>
                <parameter free="0" max="2000.0" min="30.0" name="Scale" scale="1.0" value="100.0"/>
            </spectrum>
            <spatialModel type="RadialShell">
                <parameter free="0" max="360" min="-360" name="GLON" scale="1.0" value="0.0"/>
                <parameter free="0" max="90" min="-90" name="GLAT" scale="1.0" value="0.0"/>
                <parameter free="1" max="10" min="0.01" name="Radius" scale="1" value="0.8"/>                
                <parameter free="1" max="10" min="0.01" name="Width" scale="1" value="0.1"/>                
            </spatialModel>
        </source>          
    </source_library>
    '''

    sourcelib = xml_to_source_library(xml_str)

    assert len(sourcelib.skymodels) == 4

    model1 = sourcelib.skymodels[0]
    assert isinstance(model1.spectral_model, spectral.PowerLaw)
    assert isinstance(model1.spatial_model, spatial.SkyPointSource)

    pars1 = model1.parameters
    assert pars1['index'].value == 2.1
    assert pars1['index'].unit == ''
    assert pars1['index'].parmax == 1.0
    assert pars1['index'].parmin == 5.0
    assert pars1['index'].frozen == False

    assert pars1['lon_0'].value == 0.5
    assert pars1['lon_0'].unit == 'deg'
    assert pars1['lon_0'].parmax == 360
    assert pars1['lon_0'].parmin == -360
    assert pars1['lon_0'].frozen == True

    assert pars1['lat_0'].value == 1.0
    assert pars1['lat_0'].unit == 'deg'
    assert pars1['lat_0'].parmax == 90
    assert pars1['lat_0'].parmin == -90
    assert pars1['lat_0'].frozen == True

    model2 = sourcelib.skymodels[1]
    assert isinstance(model2.spectral_model, spectral.PowerLaw)
    assert isinstance(model2.spatial_model, spatial.SkyGaussian)

    pars2 = model2.parameters
    assert pars2['index'].value == 2.1
    assert pars2['index'].unit == ''
    assert pars2['index'].parmax == 1.0
    assert pars2['index'].parmin == 5.0
    assert pars2['index'].frozen == False

    assert pars2['lon_0'].value == 1.0
    assert pars2['lon_0'].unit == 'deg'
    assert pars2['lon_0'].parmax == 360
    assert pars2['lon_0'].parmin == -360
    assert pars2['lon_0'].frozen == True

    assert pars2['lat_0'].value == 0.5
    assert pars2['lat_0'].unit == 'deg'
    assert pars2['lat_0'].parmax == 90
    assert pars2['lat_0'].parmin == -90
    assert pars2['lat_0'].frozen == True

    assert pars2['sigma'].value == 0.2
    assert pars2['sigma'].unit == 'deg'
    assert pars2['sigma'].parmax == 10
    assert pars2['sigma'].parmin == 0.01
    assert pars2['sigma'].frozen == False

@pytest.mark.xfail(reason='Need to update model regsitry')
@requires_data('gammapy-extra')
@requires_dependency('scipy')
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
