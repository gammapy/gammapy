# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.maps import MapAxis, WcsNDMap, RegionGeom, Maps
from gammapy.estimators import ESTIMATOR_REGISTRY


def test_estimator_registry():
    assert "Estimator" in str(ESTIMATOR_REGISTRY)
