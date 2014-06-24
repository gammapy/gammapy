# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.utils.compat.odict import OrderedDict
from ..flux_point import FluxPoints


@pytest.mark.xfail
def test_FluxPoints_from_dict():
    data = OrderedDict(x=[1, 2], y=[3, 4])
    points = FluxPoints.from_dict(data)
    assert points.y[0] == 3


@pytest.mark.xfail
def test_FluxPoints_from_ascii():
    points = FluxPoints.from_ascii('input/crab_hess_spec.txt')
    # Check row # 13 (when starting counting at 0)
    # against values from ascii file
    # 7.145e+00 1.945e-13 2.724e-14 2.512e-14
    assert_allclose(points.x[13], 7.145e+00)
    assert_allclose(points.y[13], 1.945e-13)
    assert_allclose(points.eyl[13], 2.724e-14)
    assert_allclose(points.eyh[13], 2.512e-14)
