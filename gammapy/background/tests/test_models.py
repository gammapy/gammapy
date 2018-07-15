# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.table import Table
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from ...utils.testing import requires_dependency, requires_data, assert_quantity_allclose
from ...utils.energy import EnergyBounds
from ...data import ObservationTable, DataStore, EventList
from ...background.models import _compute_pie_fraction, _select_events_outside_pie
from ...background import GaussianBand2D, FOVCubeBackgroundModel, EnergyOffsetBackgroundModel


@requires_dependency('scipy')
class TestGaussianBand2D:
    def setup(self):
        table = Table()
        table['GLON'] = [-30, -10, 10, 20] * u.deg
        table['Surface_Brightness'] = [0, 1, 10, 0] * u.Unit('cm-2 s-1 sr-1')
        table['GLAT'] = [-1, 0, 1, 0] * u.deg
        table['Width'] = [0.4, 0.5, 0.3, 1.0] * u.deg
        self.table = table
        self.model = GaussianBand2D(table)

    def test_evaluate(self):
        x = np.linspace(-100, 20, 5)
        y = np.linspace(-2, 2, 7)
        x, y = np.meshgrid(x, y)
        coords = SkyCoord(x, y, unit='deg', frame='galactic')
        image = self.model.evaluate(coords)
        desired = 1.223962643740966 * u.Unit('cm-2 s-1 sr-1')
        assert_quantity_allclose(image.sum(), desired)

    def test_parvals(self):
        glon = Angle(10, unit='deg')
        assert_quantity_allclose(self.model.peak_brightness(glon), 10 * u.Unit('cm-2 s-1 sr-1'))
        assert_quantity_allclose(self.model.peak_latitude(glon), 1 * u.deg)
        assert_quantity_allclose(self.model.width(glon), 0.3 * u.deg)
