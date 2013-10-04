# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from ..pulsar import Pulsar, ModelPulsar


def test_Pulsar():
    # TODO: get verified test cases
    pulsar = Pulsar(P=1, Pdot=1)
    assert_allclose(pulsar.L, 3.9478417604357427e+46)


def test_ModelPulsar():
    # TODO: get verified test cases
    pulsar = ModelPulsar()
    assert_allclose(pulsar.L(0), 3.855314219175528e+31)
