# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.maps import MapAxis
from gammapy.utils.nddata import NDDataArray


@pytest.fixture(scope="session")
def axis_x():
    return MapAxis.from_nodes([1, 3, 6], name="x")


@pytest.fixture(scope="session")
def axis_energy():
    return MapAxis.from_bounds(
        0.1, 1000, 2, unit=u.TeV, name="energy", interp="log", node_type="edges"
    )


@pytest.fixture(scope="session")
def axis_offset():
    return MapAxis.from_nodes([0.2, 0.3, 0.4, 0.5] * u.deg, name="offset")


@pytest.fixture(scope="session")
def nddata_1d(axis_x):
    return NDDataArray(
        axes=[axis_x],
        data=[1, -1, 2],
        interp_kwargs=dict(bounds_error=False, fill_value=None),
    )


@pytest.fixture(scope="session")
def nddata_2d(axis_energy, axis_offset):
    return NDDataArray(
        axes=[axis_energy, axis_offset],
        data=np.arange(8).reshape(2, 4) * u.cm * u.cm,
        interp_kwargs=dict(bounds_error=False, fill_value=None),
    )


class TestNDDataArray:
    def test_init_error(self):
        with pytest.raises(ValueError):
            NDDataArray(
                axes=[MapAxis.from_nodes([1, 3, 6], name="x")],
                data=np.arange(8).reshape(4, 2),
            )

    def test_str(self, nddata_1d):
        assert "x" in str(nddata_1d)

    def test_evaluate_shape_1d(self, nddata_1d):
        # Scalar input
        out = nddata_1d.evaluate(x=1.5)
        assert out.shape == (1,)

        # Array input
        out = nddata_1d.evaluate(x=[0, 1.5])
        assert out.shape == (2,)

        # No input
        out = nddata_1d.evaluate()
        assert out.shape == (3,)

    def test_evaluate_2d(self, nddata_2d):
        # Case 1: axis1 = scalar, axis2 = array
        out = nddata_2d.evaluate(energy=1 * u.TeV, offset=[0, 0] * u.deg)
        assert out.shape == (2,)

        # Case 2: axis1 = array, axis2 = array
        offset, energy = [0, 0] * u.deg, [1, 1, 1] * u.TeV
        out = nddata_2d.evaluate(energy=energy[:, np.newaxis], offset=offset)
        assert out.shape == (3, 2)

        # Case 3: axis1 array, axis2 = 2Darray
        offset, energy = [0, 0] * u.deg, np.zeros((12, 3)) * u.TeV
        out = nddata_2d.evaluate(energy=energy[:, :, np.newaxis], offset=offset)
        assert out.shape == (12, 3, 2)

    @pytest.mark.parametrize("shape", [(2,), (3, 2), (4, 2, 3)])
    def test_evaluate_2d_shape(self, nddata_2d, shape):
        points = dict(
            energy=np.ones(shape) * 1 * u.TeV, offset=np.ones(shape) * 0.3 * u.deg
        )
        out = nddata_2d.evaluate(**points)
        assert out.shape == shape
        assert_allclose(out.value, 1)

        points = dict(
            energy=np.ones(shape) * 100 * u.TeV, offset=np.ones(shape) * 0.3 * u.deg
        )
        out = nddata_2d.evaluate(**points)
        assert_allclose(out.value, 5)

    def test_evaluate_1d_linear(self, nddata_1d):
        # This should test all cases of interest:
        # - evaluate outside node array, i.e. extrapolate: x=0
        # - evaluate on a given node: x=1
        # - evaluate in between nodes: x=2
        # - check that values < 0 are clipped to 0: x=3
        out = nddata_1d.evaluate(x=[0, 1, 2, 3], method="linear")
        assert_allclose(out, [2, 1, 0, 0])

    def test_evaluate_on_nodes(self, nddata_2d):
        # evaluating on interpolation nodes should give back the interpolation values
        out = nddata_2d.evaluate()
        assert_allclose(out, nddata_2d.data)
