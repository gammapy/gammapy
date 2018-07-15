# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from ...testing import requires_data, requires_dependency
from ....cube import SourceLibrary
from ....spectrum import models as spectral
from ....image import models as spatial
from ...serialization import xml_to_source_library, UnknownModelError


@requires_data('gammapy-extra')
def test_complex():
    xml_str = '''<?xml version="1.0" encoding="utf-8"?>
    <source_library title="source library">
        <source name="Source 0" type="PointSource">
            <spectrum type="PowerLaw">
                <parameter free="1" max="1000.0" min="0.001" name="Prefactor" scale="1e-09" value="10" />
                <parameter free="1" max="-1.0" min="-5.0" name="Index" scale="1.0" value="-2.1"/>
                <parameter free="0" max="2000.0" min="30.0" name="Scale" scale="1.0" value="100.0"/>
            </spectrum>
            <spatialModel type="PointSource">
                <parameter free="0" max="360" min="-360" name="GLON" scale="1.0" value="0.5"/>
                <parameter free="0" max="90" min="-90" name="GLAT" scale="1.0" value="1.0"/>
            </spatialModel>
        </source>
        <source name="Source 1" type="SkyDirFunction">
            <spectrum type="PowerLaw">
                <parameter free="1" max="1000.0" min="0.001" name="Prefactor" scale="1e-09" value="10" />
                <parameter free="1" max="-1.0" min="-5.0" name="Index" scale="1.0" value="-2.1"/>
                <parameter free="0" max="2000.0" min="30.0" name="Scale" scale="1.0" value="100.0"/>
            </spectrum>
            <spatialModel type="SkyDirFunction">
                <parameter free="0" max="360" min="-360" name="GLON" scale="1.0" value="0.5"/>
                <parameter free="0" max="90" min="-90" name="GLAT" scale="1.0" value="1.0"/>
            </spatialModel>
        </source>              
        <source name="Source 2" type="RadialGaussian">
            <spectrum type="ExponentialCutoffPowerLaw">
                <parameter name="Prefactor" value="2.48" error="0" scale="1e-18" min="1e-07" max="1000" free="1" />
                <parameter name="Index" value="-2.2" error="0" scale="1" min="-5" max="-1" free="-1" />
                <parameter name="CutoffEnergy" value="100" error="0" scale="1" min="0.01" max="1000" free="1" />
                <parameter name="PivotEnergy" value="1" scale="1000000" min="0.01" max="1000" free="0" />
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
                <parameter free="1" name="Index" scale="1.0" value="-2.1"/>
                <parameter free="0" name="Scale" scale="1.0" value="100.0"/>
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
        <source name="RXJ1713" type="DiffuseSource">
            <spectrum type="ExponentialCutoffPowerLaw">
                <parameter name="Prefactor" scale="1e-17" value="2.3" min="1e-07" max="1000.0" free="1"/>
                <parameter name="Index" scale="-1" value="2.06" min="0.0" max="+5.0" free="1"/>
                <parameter name="PivotEnergy" scale="1e6" value="1.0" min="0.01" max="1000.0" free="0"/>
                <parameter name="CutoffEnergy" scale="1e6" value="12.9" min="0.01" max="1000.0" free="1"/>
            </spectrum>
            <spatialModel type="DiffuseMap" file="$GAMMAPY_EXTRA/datasets/catalogs/fermi/Extended_archive_v18/Templates/RXJ1713_2016_250GeV.fits">
                <parameter name="Prefactor" value="1" scale="1" min="0.001" max="1000" free="0"/>
            </spatialModel>
        </source>
        <source name="IEM" type="DiffuseSource">
            <spectrum type="ConstantValue">
                <parameter name="Value" value="1" error="0" scale="1" min="1e-05" max="100000" free="1" />
            </spectrum>
            <spatialModel type="MapCubeFunction" file="$GAMMAPY_EXTRA/datasets/vela_region/gll_iem_v05_rev1_cutout.fits">
                <parameter name="Normalization" value="1" scale="1" min="0.001" max="1000" free="0" />
            </spatialModel>
        </source>                            
    </source_library>
    '''

    sourcelib = SourceLibrary.from_xml(xml_str)

    assert len(sourcelib.skymodels) == 7

    model0 = sourcelib.skymodels[0]
    assert isinstance(model0.spectral_model, spectral.PowerLaw)
    assert isinstance(model0.spatial_model, spatial.SkyPointSource)

    model1 = sourcelib.skymodels[1]
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

    model2 = sourcelib.skymodels[2]
    assert isinstance(model2.spectral_model, spectral.ExponentialCutoffPowerLaw)
    assert isinstance(model2.spatial_model, spatial.SkyGaussian)

    pars2 = model2.parameters
    assert pars2['sigma'].unit == 'deg'
    assert pars2['lambda_'].value == 0.01
    assert pars2['lambda_'].unit == 'MeV-1'
    assert pars2['lambda_'].parmin == 100
    assert pars2['lambda_'].parmax == 0.001
    assert pars2['index'].value == 2.2
    assert pars2['index'].unit == ''
    assert pars2['index'].parmax == 1.0
    assert pars2['index'].parmin == 5.0

    model3 = sourcelib.skymodels[3]
    assert isinstance(model3.spatial_model, spatial.SkyDisk)
    pars3 = model3.parameters
    assert pars3['r_0'].unit == 'deg'

    model4 = sourcelib.skymodels[4]
    assert isinstance(model4.spatial_model, spatial.SkyShell)
    pars4 = model4.parameters
    assert pars4['radius'].unit == 'deg'
    assert pars4['width'].unit == 'deg'

    model5 = sourcelib.skymodels[5]
    assert isinstance(model5.spatial_model, spatial.SkyDiffuseMap)

    model6 = sourcelib.skymodels[6]
    assert isinstance(model6.spatial_model, spatial.SkyDiffuseMap)


@pytest.mark.xfail(reason='Need to update model regsitry')
@requires_data('gammapy-extra')
@requires_dependency('scipy')
@pytest.mark.parametrize('filenames', [[
    '$GAMMAPY_EXTRA/test_datasets/models/fermi_model.xml',
    '$GAMMAPY_EXTRA/test_datasets/models/shell.xml',
]])
def test_models(filenames, tmpdir):
    outfile = tmpdir / 'models_out.xml'
    for filename in filenames:
        sourcelib = SourceLibrary.read(filename)
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
        xml_to_source_library(xml)

    # TODO: Think about a more elaborate XML validation scheme
