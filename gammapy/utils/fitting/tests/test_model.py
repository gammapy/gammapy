# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ..parameter import Parameter, Parameters
from ..model import Model


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
