# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.modeling import Model, Parameter, Parameters


class MyModel(Model):
    """Simple model example"""
    x = Parameter("x", 1)
    y = Parameter("y", 2)


# TODO: change example to also hold parameters?
class CoModel(Model):
    """Compound model example"""
    norm = Parameter("norm", 99)

    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
        super().__init__()

    @property
    def parameters(self):
        return self.parameters + self.m1.parameters + self.m2.parameters


def test_model_class():
    assert isinstance(MyModel.parameters, property)
    assert MyModel.x.name == "x"
    assert MyModel.default_parameters["x"] is MyModel.x


def test_model_init():
    m = MyModel()
    assert m.x.name == "x"
    assert m.x.value == 1
    assert m.x is m.parameters[0]
    assert m.y is m.parameters[1]
    assert m.parameters is not MyModel.default_parameters

    m = MyModel(x=99)
    assert m.x.value == 99
    assert m.y.value == 2


def test_model_copy():
    m = MyModel()

    m2 = m.copy()

    # Models should be independent
    assert m.parameters is not m2.parameters
    assert m.parameters[0] is not m2.parameters[0]


def test_model_create():
    spectral_model = Model.create(
        "PowerLaw2SpectralModel", amplitude="1e-10 cm-2 s-1", index=3
    )
    assert spectral_model.tag == "PowerLaw2SpectralModel"
    assert_allclose(spectral_model.index.value, 3)


def test_compound_model():
    m1 = MyModel()
    m2 = MyModel(x=10, y=20)
    m = CoModel(m1, m2)
    assert len(m.parameters) == 4
    assert m.parameters.names == ["x", "y", "x", "y"]
    assert_allclose(m.parameters.values, [1, 2, 10, 20])
