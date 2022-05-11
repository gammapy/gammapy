# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import Model, ModelBase, Models, SkyModel
from gammapy.utils.testing import mpl_plot_check, requires_data


class MyModel(ModelBase):
    """Simple model example"""

    x = Parameter("x", 1, "cm")
    y = Parameter("y", 2)


class CoModel(ModelBase):
    """Compound model example"""

    norm = Parameter("norm", 42, "cm")

    def __init__(self, m1, m2, norm=norm.quantity):
        self.m1 = m1
        self.m2 = m2
        super().__init__(norm=norm)

    @property
    def parameters(self):
        return Parameters([self.norm]) + self.m1.parameters + self.m2.parameters


class WrapperModel(ModelBase):
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
    assert_allclose(m.x.value, 99)
    assert m.x.unit == "m"

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
    assert_allclose(m.parameters.value, [42, 1, 2, 10, 20])


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
def test_set_parameters_from_table():
    # read gammapy models
    models = Models.read("$GAMMAPY_DATA/tests/models/gc_example_models.yaml")

    tab = models.to_parameters_table()
    tab["value"][0] = 3.0
    tab["min"][0] = -10
    tab["max"][0] = 10
    tab["frozen"][0] = True
    tab["name"][0] = "index2"
    tab["frozen"][1] = True

    models.update_parameters_from_table(tab)

    d = models.parameters.to_dict()
    assert d[0]["value"] == 3.0
    assert d[0]["min"] == -10
    assert d[0]["max"] == 10
    assert d[0]["frozen"]
    assert d[0]["name"] == "index"

    assert d[1]["frozen"]


@requires_data()
def test_plot_models(caplog):
    models = Models.read("$GAMMAPY_DATA/tests/models/gc_example_models.yaml")

    with mpl_plot_check():
        models.plot_positions()
        models.plot_regions()

    assert models.wcs_geom.data_shape == (171, 147)

    regions = models.to_regions()
    assert len(regions) == 3

    p1 = Model.create(
        "pl-2",
        model_type="spectral",
    )
    g1 = Model.create("gauss", model_type="spatial")
    p2 = Model.create(
        "pl-2",
        model_type="spectral",
    )
    m1 = SkyModel(spectral_model=p1, spatial_model=g1, name="m1")
    m2 = SkyModel(spectral_model=p2, name="m2")
    models = Models([m1, m2])

    models.plot_regions()
    assert "WARNING" in [_.levelname for _ in caplog.records]
    assert "Skipping model m2 - no spatial component present" in [
        _.message for _ in caplog.records
    ]


def test_positions():
    p1 = Model.create(
        "pl",
        model_type="spectral",
    )
    g1 = Model.create("gauss", model_type="spatial")
    m1 = SkyModel(spectral_model=p1, spatial_model=g1, name="m1")
    g3 = Model.create("gauss", model_type="spatial", frame="galactic")
    m3 = SkyModel(spectral_model=p1, spatial_model=g3, name="m3")
    models = Models([m1, m3])
    pos = models.positions
    assert_allclose(pos.galactic[0].l.value, 96.337, rtol=1e-3)


def test_parameter_name():
    with pytest.raises(RuntimeError):

        class MyTestModel:
            par = Parameter("wrong-name", value=3)

        _ = MyTestModel()
