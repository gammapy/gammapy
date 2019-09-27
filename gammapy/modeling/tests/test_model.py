# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.modeling import Model, make_model, Parameter, Parameters


class MyModel(Model):
    def __init__(self):
        self.parameters = Parameters(
            [Parameter("x", 2), Parameter("y", 3e2), Parameter("z", 4e-2)]
        )


def test_model():
    m = MyModel()

    m2 = m.copy()

    # Models should be independent
    assert m.parameters is not m2.parameters
    assert m.parameters[0] is not m2.parameters[0]


def test_make_model():
    spectral_model = make_model("PowerLaw2SpectralModel", amplitude="1e-10 cm-2 s-1", index=3)
    assert spectral_model.tag == "PowerLaw2SpectralModel"
    assert_allclose(spectral_model.index.value, 3)
