# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.modeling.models import Model, Parameter, Parameters
from gammapy.datasets import Datasets
from gammapy.utils.testing import requires_data


class MyModel(Model):
    """Simple model example"""
    x = Parameter("x", 1, "cm")
    y = Parameter("y", 2)


class CoModel(Model):
    """Compound model example"""
    norm = Parameter("norm", 42, "cm")

    def __init__(self, m1, m2, norm=norm.quantity):
        self.m1 = m1
        self.m2 = m2
        super().__init__(norm=norm)

    @property
    def parameters(self):
        return Parameters([self.norm]) + self.m1.parameters + self.m2.parameters


class WrapperModel(Model):
    """Wrapper compound model.

    Dynamically generated parameters in `__init__`,
    and a parameter name conflict with the wrapped
    model, both have a parameter called "y".
    """

    def __init__(self, m1, a=1, y=99):
        self.m1 = m1
        a = Parameter("a", a)
        y = Parameter("y", y)
        self.default_parameters = Parameters([a, y])
        super().__init__(a=a, y=y)

    @property
    def parameters(self):
        return Parameters([self.a, self.y]) + self.m1.parameters


def test_model_class():
    assert isinstance(MyModel.parameters, property)
    assert MyModel.x.name == "x"
    assert MyModel.default_parameters["x"] is MyModel.x


def test_model_class_par_init():
    x = Parameter("x", 4, "cm")
    y = Parameter("y", 10)

    model = MyModel(x=x, y=y)

    assert x is model.x
    assert y is model.y


def test_model_init():
    m = MyModel()
    assert m.x.name == "x"
    assert m.x.value == 1
    assert m.x is m.parameters[0]
    assert m.y is m.parameters[1]
    assert m.parameters is not MyModel.default_parameters

    m = MyModel(x="99 cm")
    assert m.x.value == 99
    assert m.y.value == 2

    # Currently we always convert to the default unit of a parameter
    # TODO: discuss if this is the behaviour we want, or if we instead
    # should change to the user-set unit, as long as it's compatible
    m = MyModel(x=99 * u.m)
    assert_allclose(m.x.value, 9900)
    assert m.x.unit == "cm"

    with pytest.raises(u.UnitConversionError):
        MyModel(x=99)

    with pytest.raises(u.UnitConversionError):
        MyModel(x=99 * u.s)


def test_wrapper_model():
    outer = MyModel()
    m = WrapperModel(outer)

    assert isinstance(m.a, Parameter)
    assert m.y.value == 99

    assert m.parameters.names == ["a", "y", "x", "y"]


def test_model_parameter():
    m = MyModel(x="99 cm")
    assert isinstance(m.x, Parameter)
    assert m.x.value == 99
    assert m.x.unit == "cm"

    with pytest.raises(TypeError):
        m.x = 99

    with pytest.raises(TypeError):
        m.x = 99 * u.cm


# TODO: implement parameter linking. Not working ATM!
@pytest.mark.xfail()
def test_model_parameter_link():
    # Assigning a parameter should create a link
    m = MyModel()
    par = MyModel.x.copy()
    m.x = par
    assert isinstance(m.x, Parameter)
    assert m.x is par
    # model.parameters should be in sync with attributes
    assert m.x is m.parameters["x"]


def test_model_copy():
    m = MyModel()

    m2 = m.copy()

    # Models should be independent
    assert m.parameters is not m2.parameters
    assert m.parameters[0] is not m2.parameters[0]


def test_model_create():
    spectral_model = Model.create(
        "pl-2", model_type="spectral", amplitude="1e-10 cm-2 s-1", index=3
    )
    assert "PowerLaw2SpectralModel" in spectral_model.tag
    assert_allclose(spectral_model.index.value, 3)


def test_compound_model():
    m1 = MyModel()
    m2 = MyModel(x=10 * u.cm, y=20)
    m = CoModel(m1, m2)
    assert len(m.parameters) == 5
    assert m.parameters.names == ["norm", "x", "y", "x", "y"]
    assert_allclose(m.parameters.values, [42, 1, 2, 10, 20])


def test_parameter_link_init():
    m1 = MyModel()
    m2 = MyModel(y=m1.y)

    assert m1.y is m2.y

    m1.y.value = 100
    assert_allclose(m2.y.value, 100)


def test_parameter_link():
    m1 = MyModel()
    m2 = MyModel()

    m2.y = m1.y

    m1.y.value = 100
    assert_allclose(m2.y.value, 100)


@requires_data()
def test_models_management(tmp_path):
    path = "$GAMMAPY_DATA/tests/models"
    filedata = "gc_example_datasets.yaml"
    filemodel = "gc_example_models.yaml"

    datasets = Datasets.read(path, filedata, filemodel)

    model1 = datasets.models[0].copy(name="model1", datasets_names=None)
    model2 = datasets.models[0].copy(name="model2", datasets_names=[datasets[1].name])
    model3 = datasets.models[0].copy(name="model3", datasets_names=[datasets[0].name])

    model1b = datasets.models[0].copy(name="model1", datasets_names=None)
    model1b.spectral_model.amplitude.value *= 2

    names0 = datasets[0].models.names
    names1 = datasets[1].models.names

    datasets[0].models.append(model1)
    _ = datasets[0].models + model2
    assert datasets[0].models.names == names0 + ["model1", "model2"]
    assert datasets[0].models["model1"].datasets_names is None
    assert datasets[0].models["model2"].datasets_names == [
        datasets[1].name,
        datasets[0].name,
    ]
    assert datasets[1].models.names == names1 + ["model1", "model2"]

    # TODO consistency check at datasets level ?
    # or force same Models for each dataset._models on datasets init ?
    # here we have the right behavior: model1 and model2 are also added to dataset1
    # because serialization create a global model object shared by all datasets
    # if that was not the case we could have inconsistancies
    # such as model1.datasets_names == None added only to dataset1
    # user can still create such inconsistancies if they define datasets
    # with diferent Models objects for each dataset.

    del datasets[0].models["model1"]
    assert datasets[0].models.names == names0 + ["model2"]

    datasets[0].models.remove(model2)
    assert datasets[0].models.names == names0

    datasets.models.append(model2)
    assert model2 in datasets.models
    assert model2 in datasets[1].models
    assert datasets[0].models.names == names0 + ["model2"]

    datasets[0].models.extend([model1, model3])
    assert datasets[0].models.names == names0 + ["model2", "model1", "model3"]

    for m in [model1, model2, model3]:
        datasets.models.remove(m)
    assert datasets[0].models.names == names0
    assert datasets[1].models.names == names1
    datasets.models.extend([model1, model2, model3])
    assert datasets[0].models.names == names0 + ["model1", "model2", "model3"]
    assert datasets[1].models.names == names1 + ["model1", "model2"]

    for m in [model1, model2, model3]:
        datasets.models.remove(m)
    _ = datasets.models + [model1, model2]
    assert datasets[0].models.names == names0 + ["model1", "model2"]
    assert datasets[1].models.names == names1 + ["model1", "model2"]

    datasets[0].models["model2"] = model3
    assert datasets[0].models.names == names0 + ["model1", "model3"]
    assert datasets[1].models.names == names1 + ["model1"]

    datasets.models.remove(model1)
    datasets[0].models = model1
    _ = datasets.models  # auto-update models

    npred1 = datasets[0].npred().data.sum()
    datasets.models.remove(model1)
    npred0 = datasets[0].npred().data.sum()
    datasets.models.append(model1b)
    npred1b = datasets[0].npred().data.sum()
    assert npred1b != npred1
    assert npred1b != npred0
    assert_allclose(npred1b, 5157.137554, rtol=1e-5)

    datasets.models.remove(model1b)
    _ = datasets.models  # auto-update models
    newmodels = [datasets.models[0].copy() for k in range(48)]
    datasets.models.extend(newmodels)

    datasets[0].use_cache = False
    nocache = datasets[0].npred().data.sum()
    datasets[0].use_cache = True
    assert_allclose(datasets[0].npred().data.sum(), nocache)
