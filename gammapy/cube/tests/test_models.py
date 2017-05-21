# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_dependency
from ...image.models import Shell2D
from ...spectrum.models import PowerLaw
from ..core import SkyCube
from ..models import CombinedModel3D


def make_ref_cube():
    return SkyCube.empty(
        mode='edges', enumbins=10, emin=0.1, emax=10, eunit='TeV',
        nxpix=200, nypix=100, binsz=0.1, xref=0, yref=0, proj='TAN', coordsys='GAL',
    )


class TestCombinedModel3D:
    def setup(self):
        self.spatial_model = Shell2D(
            amplitude=1, x_0=3, y_0=4, r_in=5, width=6, normed=True,
        )
        self.spectral_model = PowerLaw(
            index=2, amplitude=1 * u.Unit('cm-2 s-1 TeV-1'), reference=1 * u.Unit('TeV'),
        )
        self.model = CombinedModel3D(self.spatial_model, self.spectral_model)

    def test_repr(self):
        assert 'CombinedModel3D' in repr(self.model)

    def test_evaluate_scalar(self):
        lon = 3 * u.deg
        lat = 4 * u.deg
        energy = 1 * u.TeV

        actual = self.model.evaluate(lon, lat, energy)

        expected = [[[7.798132206]]] * u.Unit('cm-2 s-1 TeV-1 sr-1')
        assert_quantity_allclose(actual, expected)

    def test_evaluate_array(self):
        lon = 3 * u.deg * np.ones(shape=(3, 4))
        lat = 4 * u.deg * np.ones(shape=(3, 4))
        energy = [1, 1, 1, 1, 1] * u.TeV

        actual = self.model.evaluate(lon, lat, energy)

        expected = 7.798132206 * u.Unit('cm-2 s-1 TeV-1 sr-1')
        assert_quantity_allclose(actual, expected)
        assert actual.shape == (5, 3, 4)

    @requires_dependency('scipy')
    def test_evaluate_cube(self):
        ref_cube = make_ref_cube()
        cube = self.model.evaluate_cube(ref_cube)

        assert cube.data.shape == (10, 100, 200)
        assert cube.data.unit == 'cm-2 s-1 TeV-1 sr-1'
        assert_allclose(cube.data.value.sum(), 10262401.621666815)

        actual = cube.lookup(position=SkyCoord(0.1, 0.1, frame='galactic', unit='deg'), energy=1 * u.TeV)
        assert_quantity_allclose(actual, 17.444501 * u.Unit('cm-2 s-1 TeV-1 sr-1'))

        # TODO: fix the bug for the following test case!
        # Because spatial models are evaluated in cartesian x/y,
        # the switch from lon = 0 to 360 deg leads to an incorrect result! (sources clipped at lon=0 line!)
        actual = cube.lookup(position=SkyCoord(-0.1, -0.1, frame='galactic', unit='deg'), energy=1 * u.TeV)
        assert_quantity_allclose(actual, 0 * u.Unit('cm-2 s-1 TeV-1 sr-1'))
