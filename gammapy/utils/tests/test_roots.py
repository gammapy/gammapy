# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.utils.roots import find_roots


class TestFindRoots:
    lower_bound = -3 * np.pi * u.rad
    upper_bound = 0 * u.rad

    def f(self, x):
        return np.cos(x)

    def h(self, x):
        return x**3 - 1

    def test_methods(self):

        methods = ["brentq", "secant"]
        for method in methods:
            roots, res = find_roots(
                self.f,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                method=method,
            )
            assert roots.unit == u.rad
            assert_allclose(2 * roots.value / np.pi, np.array([-5.0, -3.0, -1.0]))
            assert np.all([sol.converged for sol in res])

            roots, res = find_roots(
                self.h,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                method=method,
            )
            assert np.isnan(roots[0])
            assert res[0].iterations == 0

    def test_invalid_method(self):
        with pytest.raises(ValueError, match='Unknown solver "xfail"'):
            find_roots(
                self.f,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                method="xfail",
            )
