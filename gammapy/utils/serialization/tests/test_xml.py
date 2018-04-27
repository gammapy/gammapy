# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...scripts import make_path
from ....cube import SourceLibrary
from ....spectrum import models as spectral
from ....image import models as spatial

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

    # TODO: Test Evaluate combined model as soon as '+' works for SkyModel

