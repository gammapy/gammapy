# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from astropy.units import Unit
import numpy as np
from numpy.testing import assert_allclose
from ...testing import requires_data
from ....spectrum import models as spectral
from ....image import models as spatial
from ....cube.models import SkyModels
from ...scripts import read_yaml, write_yaml
from ...serialization import models_to_dict, dict_to_models
from astropy.utils.data import get_pkg_data_filename


@requires_data()
def test_yaml_io_3d():
    filename = get_pkg_data_filename("data/examples.yaml")
    models = dict_to_models(filename)

    assert len(models) == 3

    model0 = models[0]
    assert isinstance(model0.spectral_model, spectral.ExponentialCutoffPowerLaw)
    assert isinstance(model0.spatial_model, spatial.SkyPointSource)

    pars0 = model0.parameters
    assert pars0["index"].value == 2.1
    assert pars0["index"].unit == ""
    assert np.isnan(pars0["index"].max)
    assert np.isnan(pars0["index"].min)
    assert pars0["index"].frozen is False

    assert pars0["lon_0"].value == -50.0
    assert pars0["lon_0"].unit == "deg"
    assert pars0["lon_0"].max == 180.0
    assert pars0["lon_0"].min == -180.0
    assert pars0["lon_0"].frozen is True

    assert pars0["lat_0"].value == -0.05
    assert pars0["lat_0"].unit == "deg"
    assert pars0["lat_0"].max == 90.0
    assert pars0["lat_0"].min == -90.0
    assert pars0["lat_0"].frozen is True

    assert pars0["lambda_"].value == 0.06
    assert pars0["lambda_"].unit == "TeV-1"
    assert np.isnan(pars0["lambda_"].min)
    assert np.isnan(pars0["lambda_"].max)

    model1 = models[1]
    assert isinstance(model1.spectral_model, spectral.PowerLaw)
    assert isinstance(model1.spatial_model, spatial.SkyDisk)

    pars1 = model1.parameters
    assert pars1["index"].value == 2.2
    assert pars1["index"].unit == ""
    assert pars1["lat_0"].scale == 1.0
    assert pars1["lat_0"].factor == pars1["lat_0"].value

    assert np.isnan(pars1["index"].max)
    assert np.isnan(pars1["index"].min)

    assert pars1["r_0"].unit == "deg"

    model2 = models[2]
    assert model2.spectral_model.energy.data.tolist() == [34.171, 44.333, 57.517]
    assert model2.spectral_model.energy.unit == "MeV"
    assert model2.spectral_model.values.data.tolist() == [
        2.52894e-06,
        1.2486e-06,
        6.14648e-06,
    ]
    assert model2.spectral_model.values.unit == "1 / (cm2 MeV s sr)"

    assert isinstance(model2.spectral_model, spectral.TableModel)
    assert isinstance(model2.spatial_model, spatial.SkyDiffuseMap)

    assert model2.spatial_model.parameters["norm"].value == 1.0
    assert model2.spectral_model.parameters["norm"].value == 2.1
    # TODO problem of duplicate parameter name between SkyDiffuseMap and TableModel
    # assert model2.parameters["norm"].value == 2.1 # fail

    models_dict = models_to_dict(models)
    write_yaml(models_dict, "$GAMMAPY_DATA/tests/models/examples_write.yaml")
    dict_to_models("$GAMMAPY_DATA/tests/models/examples_write.yaml")

    models_dict = models_to_dict(models, selection="simple")
    write_yaml(models_dict, "$GAMMAPY_DATA/tests/models/examples_write_cut.yaml")
    dict_to_models("$GAMMAPY_DATA/tests/models/examples_write_cut.yaml")
