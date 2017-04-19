# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest, assert_quantity_allclose
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from ..testing import requires_dependency
from ..nddata import NDDataArray, BinnedDataAxis, DataAxis, sqrt_space


def get_test_arrays():
    data_arrays = list()
    data_arrays.append(dict(
        tag='1D linear',
        data=np.arange(10),
        axes=[
            DataAxis(np.arange(10), name='x-axis')
        ],
        linear_val={'x-axis': 8.5},
        linear_res=8.5,
    ))
    data_arrays.append(dict(
        tag='2D log-linear',
        data=np.arange(12).reshape(3, 4) * u.cm * u.cm,
        axes=[
            BinnedDataAxis.logspace(1, 10, 3, unit=u.TeV, name='energy',
                                    interpolation_mode='log'),
            DataAxis([0.2, 0.3, 0.4, 0.5] * u.deg, name='offset')
        ],
        linear_val={'energy': 4.54 * u.TeV, 'offset': 0.23 * u.deg},
        linear_res=6.184670234285248 * u.cm ** 2,
    ))
    return data_arrays


@pytest.mark.parametrize('config', get_test_arrays())
@requires_dependency('scipy')
def test_nddata(config):
    tester = NDDataArrayTester(config)
    tester.test_all()


class NDDataArrayTester:
    def __init__(self, config):
        self.config = config
        self.data = config['data']
        self.axes = config['axes']
        self.nddata = NDDataArray(axes=self.axes, data=self.data)

    def test_all(self):
        self.test_basic()
        self.test_wrong_init()
        self.test_find_node()
        self.test_evaluate_nodes()
        self.test_linear_interpolation()
        self.test_return_shape()
        self.test_1d()
        self.test_2d()

    def test_basic(self):
        assert self.axes[0].name in str(self.nddata)

    def test_wrong_init(self):
        wrong_data = np.arange(8).reshape(4, 2)
        with pytest.raises(ValueError):
            nddata = NDDataArray(axes=self.axes, data=wrong_data)

    def test_find_node(self):
        kwargs = {}
        for axis in self.nddata.axes:
            kwargs[axis.name] = axis.nodes[1] * 1.24
        node = self.nddata.find_node(**kwargs)
        actual = self.data[node]
        desired = self.nddata.evaluate(method='nearest', **kwargs)
        assert_quantity_allclose(actual, desired)

    def test_evaluate_nodes(self):
        nodes = [2, 3, 1]
        kwargs = dict()
        for dim, axis in enumerate(self.axes):
            kwargs[axis.name] = axis.nodes[nodes[dim]]

        actual = self.nddata.evaluate(method='nearest', **kwargs)
        desired = self.nddata.data[tuple(nodes[0:dim + 1])]
        assert_allclose(actual, desired)

        actual = self.nddata.evaluate(method='linear', **kwargs)
        desired = self.nddata.data[tuple(nodes[0:dim + 1])]
        assert_allclose(actual, desired)

    def test_linear_interpolation(self):
        actual = self.nddata.evaluate(method='linear',
                                      **self.config['linear_val'])
        desired = self.config['linear_res']
        assert_quantity_allclose(actual, desired)

    def test_return_shape(self):
        # Case 0; no kwargs
        actual = self.nddata.evaluate().shape
        desired = self.data.shape
        assert actual == desired

    def test_1d(self):
        if self.nddata.dim != 1:
            return

        # Case 1: scalar input
        kwargs = {self.axes[0].name: self.axes[0].nodes[2] * 0.75}
        actual = self.nddata.evaluate(**kwargs).shape
        desired = ()
        assert actual == desired

        # Case 2: array input
        kwargs = {self.axes[0].name: self.axes[0].nodes[0:2] * 0.75}
        actual = self.nddata.evaluate(**kwargs).shape
        desired = np.zeros(2).shape
        assert actual == desired

    def test_2d(self):
        if self.nddata.dim != 2:
            return

        # Case 1: axis1 = scalar, axis2 = array
        kwargs = {self.axes[0].name: self.axes[0].lo[1] * 0.75,
                  self.axes[1].name: self.axes[1].nodes[0:-1] * 1.1}
        actual = self.nddata.evaluate(**kwargs).shape
        desired = np.zeros(len(self.axes[1].nodes) - 1).shape
        assert actual == desired

        # Case 2: axis1 = array, axis2 = array
        kwargs = {self.axes[0].name: self.axes[0].lo[1:3] * 0.75,
                  self.axes[1].name: self.axes[1].nodes[0:3] * 1.1}
        actual = self.nddata.evaluate(**kwargs).shape
        desired = (2, 3)
        assert actual == desired

        # Case 3: axis1 array, axis2 = 2Darray
        nx, ny = (12, 3)

        # NOTE:  np.linspace does not work with Quantities and numpy 1.10
        eval_field = np.linspace(self.axes[1].nodes[1].value,
                                 self.axes[1].nodes[2].value,
                                 nx * ny).reshape(nx, ny) * self.axes[1].unit
        kwargs = {self.axes[0].name: self.axes[0].lo[0:2],
                  self.axes[1].name: eval_field}
        actual = self.nddata.evaluate(**kwargs).shape
        desired = np.zeros([2, nx, ny]).shape
        assert actual == desired


def test_sqrt_space():
    values = sqrt_space(0, 2, 5)

    assert_allclose(values, [0., 1., 1.41421356, 1.73205081, 2.])
