# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Unit tests for the Covariance class"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.modeling import Covariance, Parameter, Parameters
from gammapy.utils.testing import mpl_plot_check


@pytest.fixture
def covariance_diagonal():
    x = Parameter("x", 1, error=0.1)
    y = Parameter("y", 2, error=0.2)
    z = Parameter("z", 3, error=0.3)

    parameters = Parameters([x, y, z])
    return Covariance(parameters=parameters)


@pytest.fixture
def covariance(covariance_diagonal):
    x = covariance_diagonal.parameters["x"]
    y = covariance_diagonal.parameters["y"]
    parameters = Parameters([x, y])
    data = np.ones((2, 2))
    return Covariance(parameters=parameters, data=data)


def test_str(covariance_diagonal):
    assert "0.01" in str(covariance_diagonal)


def test_shape(covariance_diagonal):
    assert_allclose(covariance_diagonal.shape, (3, 3))


def test_set_data(covariance_diagonal):
    data = np.ones((2, 2))
    with pytest.raises(ValueError):
        covariance_diagonal.data = data


def test_set_subcovariance(covariance_diagonal, covariance):
    covariance_diagonal.set_subcovariance(covariance)
    assert_allclose(covariance_diagonal.data[:2, :2], np.ones((2, 2)))


def test_get_subcovariance(covariance_diagonal, covariance):
    covar = covariance_diagonal.get_subcovariance(covariance.parameters)
    assert_allclose(np.diag(covar), [0.1**2, 0.2**2])


def test_scipy_mvn(covariance):
    mvn = covariance.scipy_mvn
    value = mvn.pdf(2)
    assert_allclose(value, 0.2489, rtol=1e-3)


def test_plot_correlation(covariance_diagonal):
    with mpl_plot_check():
        covariance_diagonal.plot_correlation()
