# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ...testing import requires_data, requires_dependency
from ....spectrum import models as spectral
from ....image import models as spatial
from ....cube.models import SkyModels
from ...serialization import xml_to_sky_models, UnknownModelError


def test_from_xml():
    xml = """<?xml version="1.0" encoding="utf-8"?>
        <source_library title="source library">
            <source name="3C 273" type="PointSource">
                <spectrum type="PowerLaw">;i
                    <parameter free="1" max="1000.0" min="0.001" name="Prefactor" scale="1e-09" value="10"></parameter>
                    <parameter free="1" max="-1.0" min="-5.0" name="Index" scale="1.0" value="-2.1"></parameter>
                    <parameter free="0" max="2000.0" min="30.0" name="Scale" scale="1.0" value="100.0"></parameter>
                </spectrum>
                <spatialModel type="SkyDirFunction">
                    <parameter free="0" max="360" min="-360" name="RA" scale="1.0" value="187.25"></parameter>
                    <parameter free="0" max="90" min="-90" name="DEC" scale="1.0" value="2.17"></parameter>
                </spatialModel>
            </source>
        </source_library>
        """
    sky_models = SkyModels.from_xml(xml)
    sky_model = sky_models.skymodels[0]
    assert_allclose(sky_model.parameters["lon_0"].value, 187.25)


def test_xml_errors():
    xml = '<?xml version="1.0" standalone="no"?>\n'
    xml += '<source_library title="broken source library">\n'
    xml += '    <source name="CrabShell" type="ExtendedSource">\n'
    xml += '        <spatialModel type="ElefantShapedSource">\n'
    xml += '            <parameter name="RA" value="1" scale="1" min="1" max="1" free="1"/>\n'
    xml += "        </spatialModel>\n"
    xml += "    </source>\n"
    xml += "</source_library>"

    with pytest.raises(UnknownModelError):
        xml_to_sky_models(xml)

    # TODO: Think about a more elaborate XML validation scheme


@requires_data("gammapy-extra")
@requires_dependency("scipy")
def test_complex():
    filename = "$GAMMAPY_EXTRA/test_datasets/models/examples.xml"
    sourcelib = SkyModels.read(filename)

    assert len(sourcelib.skymodels) == 7

    model0 = sourcelib.skymodels[0]
    assert isinstance(model0.spectral_model, spectral.PowerLaw)
    assert isinstance(model0.spatial_model, spatial.SkyPointSource)

    model1 = sourcelib.skymodels[1]
    assert isinstance(model1.spectral_model, spectral.PowerLaw)
    assert isinstance(model1.spatial_model, spatial.SkyPointSource)

    pars1 = model1.parameters
    assert pars1["index"].value == 2.1
    assert pars1["index"].unit == ""
    assert pars1["index"].max is np.nan
    assert pars1["index"].min is np.nan
    assert pars1["index"].frozen is False

    assert pars1["lon_0"].value == 0.5
    assert pars1["lon_0"].unit == "deg"
    assert pars1["lon_0"].max == 360
    assert pars1["lon_0"].min == -360
    assert pars1["lon_0"].frozen is True

    assert pars1["lat_0"].value == 1.0
    assert pars1["lat_0"].unit == "deg"
    assert pars1["lat_0"].max == 90
    assert pars1["lat_0"].min == -90
    assert pars1["lat_0"].frozen is True

    model2 = sourcelib.skymodels[2]
    assert isinstance(model2.spectral_model, spectral.ExponentialCutoffPowerLaw)
    assert isinstance(model2.spatial_model, spatial.SkyGaussian)

    pars2 = model2.parameters
    assert pars2["sigma"].unit == "deg"
    assert pars2["lambda_"].value == 0.01
    assert pars2["lambda_"].unit == "MeV-1"
    assert pars2["lambda_"].min is np.nan
    assert pars2["lambda_"].max is np.nan
    assert pars2["index"].value == 2.2
    assert pars2["index"].unit == ""
    assert pars2["index"].max is np.nan
    assert pars2["index"].min is np.nan

    model3 = sourcelib.skymodels[3]
    assert isinstance(model3.spatial_model, spatial.SkyDisk)
    pars3 = model3.parameters
    assert pars3["r_0"].unit == "deg"

    model4 = sourcelib.skymodels[4]
    assert isinstance(model4.spatial_model, spatial.SkyShell)
    pars4 = model4.parameters
    assert pars4["radius"].unit == "deg"
    assert pars4["width"].unit == "deg"

    model5 = sourcelib.skymodels[5]
    assert isinstance(model5.spatial_model, spatial.SkyDiffuseMap)

    model6 = sourcelib.skymodels[6]
    assert isinstance(model6.spatial_model, spatial.SkyDiffuseMap)


@pytest.mark.xfail(reason="Need to improve XML read")
@requires_data("gammapy-extra")
@requires_dependency("scipy")
@pytest.mark.parametrize(
    "filename",
    [
        "$GAMMAPY_EXTRA/test_datasets/models/fermi_model.xml",
        "$GAMMAPY_EXTRA/test_datasets/models/shell.xml",
    ],
)
def test_models(filename, tmpdir):
    outfile = tmpdir / "models_out.xml"
    sourcelib = SkyModels.read(filename)
    sourcelib.to_xml(outfile)

    sourcelib_roundtrip = SkyModels.from_xml(outfile)

    for model, model_roundtrip in zip(
        sourcelib.skymodels, sourcelib_roundtrip.skymodels
    ):
        assert str(model) == str(model_roundtrip)


@pytest.mark.xfail(reason="Need to improve XML read")
@requires_data("gammapy-extra")
def test_sky_models_old_xml_file():
    filename = "$GAMMAPY_EXTRA/test_datasets/models/shell.xml"
    sources = SkyModels.read(filename)

    assert len(sources.source_list) == 2

    source = sources.source_list[0]
    assert source.source_name == "CrabShell"
    assert_allclose(source.spectral_model.parameters["Index"].value, 2.48)

    xml = sources.to_xml()
    assert "sources" in xml


@pytest.mark.xfail(reason="Need to improve XML read")
@requires_data("gammapy-extra")
def test_sky_models_new_xml_file():
    filename = (
        "$GAMMAPY_EXTRA/test_datasets/models/ctadc_skymodel_gps_sources_bright.xml"
    )
    sources = SkyModels.read(filename)

    assert len(sources.source_list) == 47

    source = sources.source_list[0]
    assert source.source_name == "HESS J0632+057"
    assert_allclose(
        source.spectral_model.parameters["Index"].value, -2.5299999713897705
    )

    xml = sources.to_xml()
    assert "sources" in xml
