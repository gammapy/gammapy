# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from ..isrf import Schlickeiser, Galprop


@pytest.skip('TODO')
class TestSchlickeiser:

    def test_omega_g_over_b(self):
        """ Check that CMB has the energy density it is
        supposed to have accoding to its temperature """
        actual = Schlickeiser()._omega_g_over_b('CMB')
        self.assertAlmostEqual(actual, 1, places=2)

    def test_call(self):
        """ Check that we roughly get the same value
        as in Fig. 3.9 of Hillert's diploma thesis.
        TODO: The check should be made against a published
        value instead """
        actual = Schlickeiser()(1e-3)
        self.assertAlmostEqual(actual / 189946, 1, places=5)


class TestGalprop:

    def test_call(self):
        Galprop()

