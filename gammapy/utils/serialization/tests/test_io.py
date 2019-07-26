# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
from astropy.utils.data import get_pkg_data_filename
from ...testing import requires_data
from ....spectrum import models as spectral
from ....image import models as spatial
from ....cube.models import SkyModels
from ...scripts import read_yaml
from ...serialization import dict_to_models


@requires_data()
def test_dict_to_skymodels(tmpdir):
    filename = get_pkg_data_filename("data/examples.yaml")
    models_data = read_yaml(filename)
    models = dict_to_models(models_data)

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
    assert_allclose(model2.spectral_model.energy.data, [34.171, 44.333, 57.517])
    assert model2.spectral_model.energy.unit == "MeV"
    assert_allclose(
        model2.spectral_model.values.data, [2.52894e-06, 1.2486e-06, 6.14648e-06]
    )
    assert model2.spectral_model.values.unit == "1 / (cm2 MeV s sr)"

    assert isinstance(model2.spectral_model, spectral.TableModel)
    assert isinstance(model2.spatial_model, spatial.SkyDiffuseMap)

    assert model2.spatial_model.parameters["norm"].value == 1.0
    assert model2.spectral_model.parameters["norm"].value == 2.1
    # TODO problem of duplicate parameter name between SkyDiffuseMap and TableModel
    # assert model2.parameters["norm"].value == 2.1 # fail


# TODO: test background model serialisation


@requires_data()
def test_sky_models_io(tmpdir):
    # TODO: maybe change to a test case where we create a model programatically?
    filename = get_pkg_data_filename("data/examples.yaml")
    models = SkyModels.from_yaml(filename)

    filename = str(tmpdir / "io_example.yaml")
    models.to_yaml(filename)
    SkyModels.from_yaml(filename)
    # TODO: add asserts to check content

    models.to_yaml(filename, selection="simple")
    SkyModels.from_yaml(filename)
    # TODO add assert to check content

    # TODO: not sure if we should just round-trip, or if we should
    # check YAML file content (e.g. against a ref file in the repo)
    # or check serialised dict content
