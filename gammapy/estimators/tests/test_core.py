# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.estimators import ESTIMATOR_REGISTRY
from gammapy.maps import MapAxis, Maps, RegionGeom, WcsNDMap
from gammapy.modeling.models import PowerLawSpectralModel


def test_estimator_registry():
    assert "Estimator" in str(ESTIMATOR_REGISTRY)
