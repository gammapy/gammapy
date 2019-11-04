# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.modeling import Model, Parameter


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
        return self._parameters + self.m1.parameters + self.m2.parameters


class WrapperModel(Model):
    """Wrapper compound model.

    Dynamically generated parameters in `__init__`,
    and a parameter name conflict with the wrapped
    model, both have a parameter called "y".
    """

    def __init__(self, m1, a=1, y=99):
        self.m1 = m1
        parameters = [Parameter("a", a), Parameter("y", y)]
        super()._init_from_parameters(parameters)

    @property
    def parameters(self):
        return self._parameters + self.m1.parameters


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
        "PowerLaw2SpectralModel", amplitude="1e-10 cm-2 s-1", index=3
    )
    assert spectral_model.tag == "PowerLaw2SpectralModel"
    assert_allclose(spectral_model.index.value, 3)


def test_compound_model():
    m1 = MyModel()
    m2 = MyModel(x=10 * u.cm, y=20)
    m = CoModel(m1, m2)
    assert len(m.parameters) == 5
    assert m.parameters.names == ["norm", "x", "y", "x", "y"]
    assert_allclose(m.parameters.values, [42, 1, 2, 10, 20])
