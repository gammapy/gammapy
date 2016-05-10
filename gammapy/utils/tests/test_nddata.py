import pytest
import random
import numpy as np
from astropy.units import Unit as u
from ...utils.testing import requires_dependency
from ..nddata import NDDataArray, DataAxis, BinnedDataAxis

def get_2d_histo():

    hist = NDDataArray()
    x_axis = BinnedDataAxis.linspace(0, 100, 10, 'm')
    y_axis = DataAxis(np.arange(1, 6), 'kg')
    y_axis.name = 'weight'

    hist.add_axis(x_axis)
    hist.add_axis(y_axis)

    data = [random.expovariate(hist.axes[1][_].value + 1) for _ in range(10)]
    val = np.arange(1, 6)
    d = np.array(data)
    data_2d = np.tensordot(val, d, axes=0)
    hist.data = data_2d

    return hist

@requires_dependency('scipy')
def test_1d_histo():

    hist = NDDataArray()
    assert hist.dim == 0

    x_axis = BinnedDataAxis.linspace(0, 100, 10, 'm')
    hist.add_axis(x_axis)
    assert hist.dim == 1

    # default name of first axis is 'x'
    assert hist.axes[0].name == 'x'
    assert hist.axis_names == ['x']
    with pytest.raises(ValueError):
        hist.get_axis('energy')
    assert (x_axis == hist.get_axis('x')).all

    # add wrong data
    data = np.arange(15)
    with pytest.raises(ValueError):
        hist.data = data

    # add correct data
    data = [random.expovariate(hist.axes[0][_].value + 1) for _ in range(10)]
    hist.data = data

    # Find nodes on x-axis
    with pytest.raises(ValueError):
        hist.find_node(x=[14 * u('s')])

    idx = hist.get_axis('x').find_node(12 * u('m'))
    assert idx[0] == 1
    idx = hist.get_axis('x').find_node(1200 * u('cm'))
    assert idx[0] == 1
    vals = [13 * u('m'), 2500 * u('cm'), 600 * u('dm')]
    idx = hist.get_axis('x').find_node(vals)
    assert idx[0] == np.array([1, 2, 6]).all()

    with pytest.raises(ValueError):
        hist.find_node(energy=5)

    # Find nodes using array input
    idx = hist.find_node(x=[12 * u('m'), 67 * u('m')])
    assert idx[0][0] == 1

    # Use hand-written nearest neighbbour interpolation
    eval_data = hist.evaluate_nearest(x=[32.52 * u('m')])
    assert eval_data == data[3]

    eval_data = hist.evaluate_nearest(
        x=[32.52 * u('m'), 12 * u('m'), 61.1512 * u('m')])
    assert (eval_data == np.asarray(data)[np.array([3, 1, 6])]).all()

    # Interpolation (test only nearest here to check if setup works)
    hist.add_linear_interpolator()

    interp_data = hist.evaluate(
        x=[32.52 * u('m'), 12 * u('m'), 61.1512 * u('m')],
        method='nearest')

    # Should give same result als evaluate_nearest
    assert (interp_data == eval_data).all()


def test_2d_histo():
    hist = get_2d_histo()
    y_axis = DataAxis(np.arange(1, 6), 'kg')

    assert hist.axis_names == ['weight', 'x']
    assert (hist.get_axis('weight') == y_axis).all()

    # Data in wrong axis order (have to reset data first)
    hist._data = None
    data = [random.expovariate(hist.axes[1][_].value + 1) for _ in range(10)]
    val = np.arange(1, 6)
    d = np.array(data)
    data_2d = np.tensordot(d, val, axes=0)
    assert data_2d.shape == (10, 5)

    with pytest.raises(ValueError):
        hist.data = data_2d

    # Correct data
    data_2d = data_2d.transpose()
    hist.data = data_2d

    nodes = hist.find_node(x=[12 * u('m'), 23 * u('m')],
                           weight=[1.2, 4.3, 3.5] * u('kg'))

    assert len(nodes) == 2
    assert len(nodes[0]) == 3
    assert len(nodes[1]) == 2
    assert nodes[1][1] == 2
    assert nodes[0][2] == 2

    nodes = hist.find_node(x=[16 * u('m')])
    assert len(nodes) == 2
    assert nodes[0][4] == 4

    eval_data = hist.evaluate_nearest(x=12 * u('m'), weight=3.2 * u('kg'))
    assert eval_data == data_2d[2, 1]

    eval_data = hist.evaluate_nearest(x=[12, 34] * u('m'),
                                      weight=[3.2, 2, 2.4] * u('kg'))
    assert eval_data.shape == (3, 2)

    eval_data = hist.evaluate_nearest(weight=[3.2, 2, 2.4] * u('kg'))
    assert eval_data.shape == (3, 10)


def test_fits_io(tmpdir):
    hist = get_2d_histo()

    f = tmpdir / 'test_file.fits'

    hist.write(str(f), format='fits', overwrite=True)
    hist2 = NDDataArray.read(str(f))

    assert (hist2.axes[0] == hist.axes[0]).all()
    assert (hist2.axes[1] == hist.axes[1]).all()
    assert (hist2.data == hist.data).all()


@requires_dependency('scipy')
def test_interpolation():
    hist = get_2d_histo()
    hist.add_linear_interpolator()

    interp_data = hist.evaluate(x=[12, 34] * u('m'),
                                weight=[3.2, 2, 2.4] * u('kg'))
    assert interp_data.shape == (3, 2)

    interp_data2 = hist.evaluate(x=[1200, 3400] * u('cm'),
                                 weight=[3200, 2000, 2400] * u('g'))

    assert (interp_data == interp_data2).all()

    interp_data = hist.evaluate(x=[77] * u('m'))

    assert interp_data.shape == (5, 1)

    # check that nodes are evaluated correctly
    interp_data = hist.evaluate()
    assert (interp_data == hist.data).all()

    # check if hand-made nearest neighbour interpolation works
    interp_data = hist.evaluate(x=[12, 34] * u('m'),
                                weight=[3.2, 2, 2.4] * u('kg'),
                                method='nearest')

    old_data = hist.evaluate_nearest(x=[12, 34] * u('m'),
                                     weight=[3.2, 2, 2.4] * u('kg'))

    assert (interp_data == old_data).all()

    # check that log interpolation works
    hist.get_axis('x').log_interpolation = True
    hist.add_linear_interpolator()

    interp_data = hist.evaluate()

    assert (interp_data == hist.data).all()
