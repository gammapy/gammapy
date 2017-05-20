# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from ...image.models import Shell2D
from ...spectrum.models import PowerLaw
from ..models import CombinedModel3D


class TestCombinedModel3D:
    def setup(self):
        self.spatial_model = Shell2D(
            amplitude=1, x_0=3, y_0=4, r_in=5, width=6, normed=True,
        )
        self.spectral_model = PowerLaw(
            index=2, amplitude=1 * u.Unit('cm-2 s-1 TeV-1'), reference=1 * u.Unit('TeV'),
        )
        self.cube = CombinedModel3D(self.spatial_model, self.spectral_model)

    def test_repr(self):
        expected = (
            "CombinedModel3D("
            "spatial_model=<Shell2D(amplitude=1.0, x_0=3.0, y_0=4.0, r_in=5.0, width=6.0)>, "
            "spectral_model=PowerLaw())"
        )
        assert repr(self.cube) == expected

    # TODO: change to a test case with a "known good answer" of the output
    # The current one we have here hasn't been validated.
    def test_evaluate_scalar(self):
        lon = 3 * u.deg
        lat = 4 * u.deg
        energy = 1 * u.TeV
        actual = self.cube.evaluate(lon, lat, energy)
        # At the moment evaluate returns a 3D array for scalar inputs
        # TODO: do we want a scalar output here or is the array output OK?
        expected = [[[7.798132206]]] * u.Unit('cm-2 s-1 TeV-1 sr-1')
        assert_quantity_allclose(actual, expected)

    # TODO: shape of lat has to be the same as lon here
    # Broadcasting of lon against lat isn't implemented yet
    def test_evaluate_array(self):
        lon = 3 * u.deg * np.ones(shape=(3, 4))
        lat = 4 * u.deg * np.ones(shape=(3, 4))
        energy = [1, 1, 1, 1, 1] * u.TeV
        actual = self.cube.evaluate(lon, lat, energy)
        expected = 7.798132206 * u.Unit('cm-2 s-1 TeV-1 sr-1')
        assert_quantity_allclose(actual, expected)
        assert actual.shape == (5, 3, 4)
