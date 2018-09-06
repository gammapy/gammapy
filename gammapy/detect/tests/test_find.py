# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from ...utils.testing import requires_dependency
from ...maps import Map
from ..find import find_peaks


@requires_dependency("scipy")
class TestFindPeaks:
    def test_simple(self):
        """Test a simple example"""
        image = Map.create(npix=(10, 5), unit="s")
        image.data[3, 3] = 11
        image.data[3, 4] = 10
        image.data[3, 5] = 12
        image.data[3, 6] = np.nan
        image.data[0, 9] = 1e20

        table = find_peaks(image, threshold=3)

        assert len(table) == 3
        assert table["value"].unit == "s"
        assert table["ra"].unit == "deg"
        assert table["dec"].unit == "deg"

        row = table[0]
        assert tuple((row["x"], row["y"])) == (9, 0)
        assert_allclose(row["value"], 1e20)
        assert_allclose(row["ra"], 359.55)
        assert_allclose(row["dec"], -0.2)

        row = table[1]
        assert tuple((row["x"], row["y"])) == (5, 3)
        assert_allclose(row["value"], 12)

    def test_no_peak(self):
        image = Map.create(npix=(10, 5))
        image.data[3, 5] = 12

        table = find_peaks(image, threshold=12.1)
        assert len(table) == 0

    def test_constant(self):
        image = Map.create(npix=(10, 5))

        table = find_peaks(image, threshold=3)
        assert len(table) == 0
